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
import reactor.netty.http.client.HttpClient;

import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.time.OffsetDateTime;
import java.time.ZoneOffset;
import java.util.LinkedHashMap;
import java.util.Map;

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
            return webClient.post()
                .uri(route.getCompletionUrl())
                .accept(MediaType.TEXT_EVENT_STREAM)
                .contentType(MediaType.APPLICATION_JSON)
                .headers(headers -> applyHeaders(route, inboundHeaders, headers))
                .bodyValue(payload)
                .exchangeToFlux(this::handleResponse);
        }).onErrorResume(exception -> {
            logger.warn("LLM gateway stream failed: {}", exception.getMessage());
            return Flux.just(errorDataBuffer(
                "LLM gateway stream failed: " + exception.getMessage()
            ));
        });
    }

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
            LlmStreamEventTransformer transformer = new LlmStreamEventTransformer(objectMapper);
            return response.bodyToFlux(DataBuffer.class)
                .concatMap(dataBuffer -> toDataBuffers(transformer.accept(readUtf8(dataBuffer))))
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

    private String completionUrl(String baseUrl) {
        String cleaned = trimTrailingSlash(baseUrl);
        if (cleaned.endsWith("/chat/completions")) {
            return cleaned;
        }
        return cleaned + "/chat/completions";
    }

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
