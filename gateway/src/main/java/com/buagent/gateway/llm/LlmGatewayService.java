package com.buagent.gateway.llm;

import com.buagent.gateway.app.dto.LlmQueryRequest;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.netty.channel.ChannelOption;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.buffer.DataBuffer;
import org.springframework.core.io.buffer.DataBufferUtils;
import org.springframework.core.io.buffer.DefaultDataBufferFactory;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.client.reactive.ReactorClientHttpConnector;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.ClientResponse;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.netty.http.client.HttpClient;

import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.time.OffsetDateTime;
import java.time.ZoneOffset;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * 模型代理网关的核心服务。
 *
 * <p>它负责把内部的 LLM 请求转成上游 OpenAI-compatible 请求，并把上游的 SSE
 * 流式响应转换回网关统一事件格式。</p>
 */
@Service
public class LlmGatewayService {

    private static final Logger logger = LoggerFactory.getLogger(LlmGatewayService.class);

    private static final String HEADER_AUTHORIZATION = "Authorization";
    private static final String HEADER_REQUEST_ID = "X-Request-Id";
    private static final String PROVIDER_OPENAI = "openai";

    private final ObjectMapper objectMapper;
    private final LlmGatewayProperties properties;
    private final WebClient webClient;
    private final DefaultDataBufferFactory dataBufferFactory;
    private final OpenAiRequestBuilder requestBuilder;

    public LlmGatewayService(
        ObjectMapper objectMapper,
        LlmGatewayProperties properties,
        WebClient.Builder webClientBuilder
    ) {
        this.objectMapper = objectMapper;
        this.properties = properties;
        this.webClient = webClientBuilder
            .clientConnector(new ReactorClientHttpConnector(buildHttpClient(properties)))
            .build();
        this.dataBufferFactory = new DefaultDataBufferFactory();
        this.requestBuilder = new OpenAiRequestBuilder();
    }

    private HttpClient buildHttpClient(LlmGatewayProperties properties) {
        HttpClient httpClient = HttpClient.create()
            .option(ChannelOption.CONNECT_TIMEOUT_MILLIS, properties.getConnectTimeoutMillis());
        if (properties.getResponseTimeoutMillis() > 0) {
            httpClient = httpClient.responseTimeout(
                Duration.ofMillis(properties.getResponseTimeoutMillis())
            );
        }
        return httpClient;
    }

    /**
     * 网关对外的流式查询入口。
     *
     * <p>这里延迟到订阅时才解析路由、组装请求和访问上游，避免在 Reactor 链路创建阶段
     * 提前执行有副作用的网络逻辑。</p>
     */
    public Flux<DataBuffer> queryStream(
        LlmQueryRequest request,
        HttpHeaders inboundHeaders
    ) {
        return Flux.defer(() -> {
            if (request == null) {
                throw new IllegalArgumentException("LLM query request is required");
            }
            ResolvedRoute route = resolveRoute(request.getModel());
            Map<String, Object> payload = requestBuilder.build(
                request,
                route.getUpstreamModel(),
                route.getMaxOutputTokens()
            );
            // 下游只看到统一的 SSE 事件；上游请求格式
            // 由 OpenAiRequestBuilder 负责适配。
            Flux<DataBuffer> upstream = webClient.post()
                .uri(route.getCompletionUrl())
                .accept(MediaType.TEXT_EVENT_STREAM)
                .contentType(MediaType.APPLICATION_JSON)
                .headers(headers -> applyHeaders(route, inboundHeaders, headers))
                .bodyValue(payload)
                .exchangeToFlux(this::handleResponse);
            return applyTotalTimeout(upstream);
        }).onErrorResume(exception -> {
            logger.warn("LLM gateway stream failed: {}", exception.getMessage());
            return Flux.just(errorDataBuffer(
                "LLM gateway stream failed: " + exception.getMessage()
            ));
        });
    }

    private Flux<DataBuffer> applyTotalTimeout(Flux<DataBuffer> stream) {
        int totalTimeoutMillis = properties.getTotalTimeoutMillis();
        if (totalTimeoutMillis <= 0) {
            return stream;
        }
        // responseTimeout 只限制两次网络 read 之间的空闲时间；
        // 这里额外限制整条 LLM 流从订阅到结束的总耗时。
        Mono<Long> timeoutSignal = Mono.delay(Duration.ofMillis(totalTimeoutMillis));
        return stream.takeUntilOther(timeoutSignal.flatMap(ignored ->
            Mono.error(new TotalStreamTimeoutException(totalTimeoutMillis))
        ));
    }

    /**
     * 处理上游 HTTP 响应。
     *
     * <p>错误响应会转换成一条 error SSE；正常响应则逐块读取上游 SSE，
     * 再通过 LlmStreamEventTransformer 做协议转换。</p>
     */
    private Flux<DataBuffer> handleResponse(ClientResponse response) {
        if (response.statusCode().isError()) {
            int statusCode = response.statusCode().value();
            return response.bodyToMono(String.class)
                .defaultIfEmpty("")
                .flatMapMany(body -> {
                    logger.warn(
                        "LLM upstream returned error. statusCode={}, bodyLength={}",
                        statusCode,
                        body.length()
                    );
                    return Flux.just(errorDataBuffer("LLM upstream returned HTTP " + statusCode));
                });
        }

        return Flux.defer(() -> {
            // Transformer 持有流式解析状态，必须按请求新建，
            // 不能跨请求复用。
            LlmStreamEventTransformer transformer = new LlmStreamEventTransformer(objectMapper);
            return response.bodyToFlux(DataBuffer.class)
                .concatMap(dataBuffer -> {
                    Flux<DataBuffer> events = toDataBuffers(transformer.accept(readUtf8(dataBuffer)));
                    if (transformer.shouldAbortUpstream()) {
                        // Transformer 已经给下游产出了 error + done。这里抛内部异常只为
                        // 短路上游订阅，避免继续读取不可恢复的坏流。
                        return events.concatWith(Flux.error(new TerminalStreamException()));
                    }
                    return events;
                })
                .onErrorResume(
                    TerminalStreamException.class,
                    exception -> Flux.empty()
                )
                .concatWith(Flux.defer(() -> toDataBuffers(transformer.finish())));
        });
    }

    private Flux<DataBuffer> toDataBuffers(Iterable<String> eventTexts) {
        return Flux.fromIterable(eventTexts)
            .map(eventText -> dataBufferFactory.wrap(eventText.getBytes(StandardCharsets.UTF_8)));
    }

    private String readUtf8(DataBuffer dataBuffer) {
        try {
            byte[] bytes = new byte[dataBuffer.readableByteCount()];
            dataBuffer.read(bytes);
            return new String(bytes, StandardCharsets.UTF_8);
        } finally {
            DataBufferUtils.release(dataBuffer);
        }
    }

    private void applyHeaders(
        ResolvedRoute route,
        HttpHeaders inboundHeaders,
        HttpHeaders outboundHeaders
    ) {
        String apiKey = route.getApiKey();
        if (!isBlank(apiKey)) {
            // 配置里可以只写裸 token，这里统一补齐
            // Authorization 认证方案。
            outboundHeaders.set(HEADER_AUTHORIZATION, normalizeBearerToken(apiKey));
        }
        forwardHeader(inboundHeaders, outboundHeaders, HEADER_REQUEST_ID);
    }

    private void forwardHeader(
        HttpHeaders inboundHeaders,
        HttpHeaders outboundHeaders,
        String headerName
    ) {
        if (inboundHeaders == null) {
            return;
        }
        String headerValue = inboundHeaders.getFirst(headerName);
        if (isBlank(headerValue)) {
            return;
        }
        outboundHeaders.set(headerName, headerValue);
    }

    /**
     * 把客户端传入的模型名解析成真实上游路由。
     *
     * <p>当配置了 routes 时，模型名被视为网关别名，必须能匹配到一条路由；
     * 未配置 routes 时，则走 defaultBaseUrl/defaultModel 的简单默认路由。</p>
     */
    private ResolvedRoute resolveRoute(String requestedModel) {
        String model = cleanOrNull(requestedModel);
        LlmGatewayProperties.Route configuredRoute = null;
        if (model != null) {
            configuredRoute = properties.getRoutes().get(model);
        }
        if (!properties.getRoutes().isEmpty() && configuredRoute == null) {
            throw new IllegalArgumentException("Unknown gateway model alias: " + requestedModel);
        }

        LlmGatewayProperties.Route route = configuredRoute == null
            ? buildDefaultRoute(model)
            : configuredRoute;
        String provider = cleanOrDefault(route.getProvider(), PROVIDER_OPENAI).toLowerCase();
        if (!PROVIDER_OPENAI.equals(provider)) {
            throw new IllegalArgumentException("Unsupported gateway provider route: " + provider);
        }

        String upstreamModel = cleanOrDefault(route.getUpstreamModel(), model);
        upstreamModel = cleanOrDefault(upstreamModel, properties.getDefaultModel());
        String baseUrl = cleanOrDefault(route.getBaseUrl(), properties.getDefaultBaseUrl());
        if (isBlank(baseUrl)) {
            throw new IllegalArgumentException("LLM upstream base URL is required");
        }
        String apiKey = resolveApiKey(route);
        return new ResolvedRoute(
            upstreamModel,
            completionUrl(baseUrl),
            apiKey,
            route.getMaxOutputTokens()
        );
    }

    private LlmGatewayProperties.Route buildDefaultRoute(String model) {
        LlmGatewayProperties.Route route = new LlmGatewayProperties.Route();
        route.setProvider(PROVIDER_OPENAI);
        route.setUpstreamModel(cleanOrDefault(model, properties.getDefaultModel()));
        route.setBaseUrl(properties.getDefaultBaseUrl());
        route.setApiKey(properties.getDefaultApiKey());
        return route;
    }

    private String resolveApiKey(LlmGatewayProperties.Route route) {
        String configuredApiKey = cleanOrNull(route.getApiKey());
        if (configuredApiKey != null) {
            return configuredApiKey;
        }
        return cleanOrNull(properties.getDefaultApiKey());
    }

    /**
     * 允许配置方填写 baseUrl 或完整的 /chat/completions 地址。
     */
    private String completionUrl(String baseUrl) {
        String cleaned = trimTrailingSlash(baseUrl);
        if (cleaned.endsWith("/chat/completions")) {
            return cleaned;
        }
        return cleaned + "/chat/completions";
    }

    /**
     * 构造网关自身的错误 SSE，保证调用方即使在失败场景也能按流式协议收尾。
     */
    private DataBuffer errorDataBuffer(String message) {
        Map<String, Object> payload = new LinkedHashMap<String, Object>();
        payload.put("type", "error");
        payload.put("timestamp", OffsetDateTime.now(ZoneOffset.UTC).toString());
        payload.put("error", message);
        String eventText;
        try {
            eventText = "data: " + objectMapper.writeValueAsString(payload) + "\n\n: done\n\n";
        } catch (Exception exception) {
            eventText = "data: {\"type\":\"error\",\"error\":\"LLM gateway stream failed\"}"
                + "\n\n: done\n\n";
        }
        return dataBufferFactory.wrap(eventText.getBytes(StandardCharsets.UTF_8));
    }

    private String normalizeBearerToken(String value) {
        String cleaned = value.trim();
        String lowered = cleaned.toLowerCase();
        if (lowered.startsWith("bearer ") || lowered.startsWith("basic ")) {
            return cleaned;
        }
        return "Bearer " + cleaned;
    }

    private String cleanOrDefault(String value, String defaultValue) {
        String cleaned = cleanOrNull(value);
        if (cleaned != null) {
            return cleaned;
        }
        return cleanOrNull(defaultValue);
    }

    private String cleanOrNull(String value) {
        if (value == null || value.trim().isEmpty()) {
            return null;
        }
        return value.trim();
    }

    private String trimTrailingSlash(String value) {
        String cleaned = value.trim();
        while (cleaned.endsWith("/")) {
            cleaned = cleaned.substring(0, cleaned.length() - 1);
        }
        return cleaned;
    }

    private boolean isBlank(String value) {
        return value == null || value.trim().isEmpty();
    }

    /**
     * 内部控制流异常：只表示当前 SSE 解析已终止，需要停止继续消费上游。
     * 它会在本类内被吞掉，不会作为网关错误暴露给调用方。
     */
    private static final class TerminalStreamException extends RuntimeException {
    }

    private static final class TotalStreamTimeoutException extends RuntimeException {

        private TotalStreamTimeoutException(int timeoutMillis) {
            super("LLM gateway stream exceeded total timeout: " + timeoutMillis + " ms");
        }
    }

    /**
     * 解析后的路由快照，避免后续代码反复读取和合并配置对象。
     */
    private static final class ResolvedRoute {

        private final String upstreamModel;
        private final String completionUrl;
        private final String apiKey;
        private final Integer maxOutputTokens;

        private ResolvedRoute(
            String upstreamModel,
            String completionUrl,
            String apiKey,
            Integer maxOutputTokens
        ) {
            this.upstreamModel = upstreamModel;
            this.completionUrl = completionUrl;
            this.apiKey = apiKey;
            this.maxOutputTokens = maxOutputTokens;
        }

        private String getUpstreamModel() {
            return upstreamModel;
        }

        private String getCompletionUrl() {
            return completionUrl;
        }

        private String getApiKey() {
            return apiKey;
        }

        private Integer getMaxOutputTokens() {
            return maxOutputTokens;
        }
    }
}
