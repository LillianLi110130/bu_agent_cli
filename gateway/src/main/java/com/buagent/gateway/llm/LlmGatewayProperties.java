package com.buagent.gateway.llm;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

import java.util.LinkedHashMap;
import java.util.Map;

@Component
@ConfigurationProperties(prefix = "gateway.llm")
public class LlmGatewayProperties {

    private String defaultModel = "GLM-4.7";
    private String defaultBaseUrl;
    private String defaultApiKey;
    private int connectTimeoutMs = 5000;
    private int responseTimeoutMs = 0;
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
