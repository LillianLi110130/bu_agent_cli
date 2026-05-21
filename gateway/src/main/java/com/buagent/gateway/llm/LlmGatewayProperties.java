package com.buagent.gateway.llm;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * LLM 网关的配置入口，对应 application.yml 中的 gateway.llm 配置段。
 *
 * <p>这里把“客户端请求的模型名”和“真实上游模型配置”分开，便于后续通过模型别名
 * 切换供应商、baseUrl 或 token 限制，而不影响调用方。</p>
 */
@Component
@ConfigurationProperties(prefix = "gateway.llm")
public class LlmGatewayProperties {

    private String defaultModel = "GLM-4.7";
    private String defaultBaseUrl;
    private String defaultApiKey;
    private int connectTimeoutMs = 5000;
    // 流式响应不能默认无限等待；上游连接成功后长时间无数据时需要释放网关资源。
    private int responseTimeoutMs = 120000;
    // 单次上游 LLM 流的总时长上限；防止持续有 chunk 的长流无限占用网关资源。
    private int totalTimeoutMs = 1200000;
    private Map<String, Route> routes = new LinkedHashMap<String, Route>();

    public String getDefaultModel() {
        return defaultModel;
    }

    public void setDefaultModel(String defaultModel) {
        this.defaultModel = defaultModel;
    }

    public String getDefaultBaseUrl() {
        return defaultBaseUrl;
    }

    public void setDefaultBaseUrl(String defaultBaseUrl) {
        this.defaultBaseUrl = defaultBaseUrl;
    }

    public String getDefaultApiKey() {
        return defaultApiKey;
    }

    public void setDefaultApiKey(String defaultApiKey) {
        this.defaultApiKey = defaultApiKey;
    }

    public int getConnectTimeoutMs() {
        return connectTimeoutMs;
    }

    public void setConnectTimeoutMs(int connectTimeoutMs) {
        this.connectTimeoutMs = connectTimeoutMs;
    }

    public int getConnectTimeoutMillis() {
        return connectTimeoutMs;
    }

    public void setConnectTimeoutMillis(int connectTimeoutMillis) {
        this.connectTimeoutMs = connectTimeoutMillis;
    }

    public int getResponseTimeoutMs() {
        return responseTimeoutMs;
    }

    public void setResponseTimeoutMs(int responseTimeoutMs) {
        this.responseTimeoutMs = responseTimeoutMs;
    }

    public int getResponseTimeoutMillis() {
        return responseTimeoutMs;
    }

    public void setResponseTimeoutMillis(int responseTimeoutMillis) {
        this.responseTimeoutMs = responseTimeoutMillis;
    }

    public int getTotalTimeoutMs() {
        return totalTimeoutMs;
    }

    public void setTotalTimeoutMs(int totalTimeoutMs) {
        this.totalTimeoutMs = totalTimeoutMs;
    }

    public int getTotalTimeoutMillis() {
        return totalTimeoutMs;
    }

    public void setTotalTimeoutMillis(int totalTimeoutMillis) {
        this.totalTimeoutMs = totalTimeoutMillis;
    }

    public Map<String, Route> getRoutes() {
        return routes;
    }

    public void setRoutes(Map<String, Route> routes) {
        if (routes == null) {
            this.routes = new LinkedHashMap<String, Route>();
            return;
        }
        this.routes = routes;
    }

    /**
     * 单个模型别名的上游路由配置。
     *
     * <p>例如客户端请求 GLM-4.7，实际可以在这里配置它转发到哪个 OpenAI-compatible
     * 上游地址、上游模型名和 API Key。</p>
     */
    public static class Route {

        private String provider = "openai";
        private String upstreamModel;
        private String baseUrl;
        private String apiKey;
        private Integer maxInputTokens;
        private Integer maxOutputTokens;

        public String getProvider() {
            return provider;
        }

        public void setProvider(String provider) {
            this.provider = provider;
        }

        public String getUpstreamModel() {
            return upstreamModel;
        }

        public void setUpstreamModel(String upstreamModel) {
            this.upstreamModel = upstreamModel;
        }

        public String getBaseUrl() {
            return baseUrl;
        }

        public void setBaseUrl(String baseUrl) {
            this.baseUrl = baseUrl;
        }

        public String getApiKey() {
            return apiKey;
        }

        public void setApiKey(String apiKey) {
            this.apiKey = apiKey;
        }

        public Integer getMaxInputTokens() {
            return maxInputTokens;
        }

        public void setMaxInputTokens(Integer maxInputTokens) {
            this.maxInputTokens = maxInputTokens;
        }

        public Integer getMaxOutputTokens() {
            return maxOutputTokens;
        }

        public void setMaxOutputTokens(Integer maxOutputTokens) {
            this.maxOutputTokens = maxOutputTokens;
        }
    }
}
