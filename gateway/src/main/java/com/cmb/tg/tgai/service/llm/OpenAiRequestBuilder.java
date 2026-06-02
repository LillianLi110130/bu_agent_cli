package com.cmb.tg.tgai.service.llm;

import com.cmb.tg.tgai.service.message.dto.LlmQueryRequest;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * 将网关内部的 LLM 请求转换成 OpenAI-compatible chat completions 请求体。
 *
 * <p>这个类只负责请求结构适配，不负责路由选择、鉴权和网络发送。</p>
 */
class OpenAiRequestBuilder {

    /**
     * 构造上游请求体，并强制使用流式返回。
     */
    Map<String, Object> build(
        LlmQueryRequest request,
        String upstreamModel,
        Integer maxOutputTokens
    ) {
        Map<String, Object> payload = new LinkedHashMap<String, Object>();
        payload.put("model", upstreamModel);
        payload.put("messages", buildMessages(request.getMessages()));
        List<Map<String, Object>> tools = buildTools(request.getTools());
        if (!tools.isEmpty()) {
            payload.put("tools", tools);
            Object toolChoice = buildToolChoice(request.getToolChoice());
            if (toolChoice != null) {
                payload.put("tool_choice", toolChoice);
            }
            payload.put("parallel_tool_calls", Boolean.TRUE);
        }
        if (maxOutputTokens != null && maxOutputTokens.intValue() > 0) {
            payload.put("max_completion_tokens", maxOutputTokens);
        }
        payload.put("stream", Boolean.TRUE);
        // 要求上游在流末返回 usage，便于网关把 token 统计透传给调用方。
        payload.put(
            "stream_options",
            Collections.<String, Object>singletonMap("include_usage", Boolean.TRUE)
        );
        return payload;
    }

    /**
     * 过滤掉无效 message，并保留 OpenAI 协议允许的字段。
     */
    private List<Map<String, Object>> buildMessages(List<Map<String, Object>> messages) {
        if (messages == null || messages.isEmpty()) {
            return Collections.emptyList();
        }

        List<Map<String, Object>> result = new ArrayList<Map<String, Object>>();
        for (Map<String, Object> message : messages) {
            if (message == null) {
                continue;
            }
            Map<String, Object> normalized = buildMessage(message);
            if (!normalized.isEmpty()) {
                result.add(normalized);
            }
        }
        return result;
    }

    private Map<String, Object> buildMessage(Map<String, Object> message) {
        String role = asText(message.get("role"));
        if (isBlank(role)) {
            return Collections.emptyMap();
        }

        Map<String, Object> result = new LinkedHashMap<String, Object>();
        result.put("role", role);
        if (message.containsKey("content")) {
            result.put("content", normalizeContent(message.get("content"), role));
        }
        copyIfPresent(message, result, "name");
        copyIfPresent(message, result, "refusal");
        copyIfPresent(message, result, "tool_call_id");
        if ("assistant".equals(role)) {
            Object toolCalls = normalizeToolCalls(message.get("tool_calls"));
            if (toolCalls != null) {
                result.put("tool_calls", toolCalls);
            }
        }
        return result;
    }

    /**
     * content 可能是纯文本，也可能是多模态 parts；这里统一成上游能接受的形态。
     */
    private Object normalizeContent(Object content, String role) {
        if (!(content instanceof List)) {
            return content;
        }

        @SuppressWarnings("unchecked")
        List<Object> parts = (List<Object>) content;
        List<Map<String, Object>> normalizedParts = new ArrayList<Map<String, Object>>();
        for (Object item : parts) {
            if (!(item instanceof Map)) {
                continue;
            }
            @SuppressWarnings("unchecked")
            Map<String, Object> part = (Map<String, Object>) item;
            String type = asText(part.get("type"));
            Map<String, Object> normalized = normalizeContentPart(part, type, role);
            if (!normalized.isEmpty()) {
                normalizedParts.add(normalized);
            }
        }
        if (normalizedParts.isEmpty()) {
            return "assistant".equals(role) ? null : "";
        }
        return normalizedParts;
    }

    private Map<String, Object> normalizeContentPart(
        Map<String, Object> part,
        String type,
        String role
    ) {
        Map<String, Object> normalized = new LinkedHashMap<String, Object>();
        if ("text".equals(type)) {
            normalized.put("type", "text");
            normalized.put("text", part.get("text"));
            return normalized;
        }
        if ("image_url".equals(type) && "user".equals(role)) {
            normalized.put("type", "image_url");
            normalized.put("image_url", part.get("image_url"));
            return normalized;
        }
        if ("document".equals(type) && "user".equals(role)) {
            // 当前上游接口不直接接收 PDF 二进制，
            // 先用文本占位保留“有附件”这一事实。
            normalized.put("type", "text");
            normalized.put("text", "[PDF document attached]");
            return normalized;
        }
        if ("refusal".equals(type) && "assistant".equals(role)) {
            normalized.put("type", "refusal");
            normalized.put("refusal", part.get("refusal"));
        }
        return normalized;
    }

    /**
     * 规范化历史 assistant 消息中的 tool_calls，供多轮工具调用继续上下文。
     */
    private Object normalizeToolCalls(Object value) {
        if (!(value instanceof List)) {
            return null;
        }
        @SuppressWarnings("unchecked")
        List<Object> rawToolCalls = (List<Object>) value;
        List<Map<String, Object>> toolCalls = new ArrayList<Map<String, Object>>();
        for (Object item : rawToolCalls) {
            if (!(item instanceof Map)) {
                continue;
            }
            @SuppressWarnings("unchecked")
            Map<String, Object> rawToolCall = (Map<String, Object>) item;
            Map<String, Object> toolCall = new LinkedHashMap<String, Object>();
            copyIfPresent(rawToolCall, toolCall, "id");
            Object type = rawToolCall.get("type") == null ? "function" : rawToolCall.get("type");
            toolCall.put("type", type);
            copyIfPresent(rawToolCall, toolCall, "function");
            if (toolCall.containsKey("function")) {
                toolCalls.add(toolCall);
            }
        }
        return toolCalls.isEmpty() ? null : toolCalls;
    }

    /**
     * 兼容两种工具定义。
     *
     * <p>已经是 OpenAI function tool 的直接透传；简化格式则包装成
     * OpenAI 要求的 type/function 结构。</p>
     */
    private List<Map<String, Object>> buildTools(List<Map<String, Object>> tools) {
        if (tools == null || tools.isEmpty()) {
            return Collections.emptyList();
        }

        List<Map<String, Object>> result = new ArrayList<Map<String, Object>>();
        for (Map<String, Object> tool : tools) {
            if (tool == null) {
                continue;
            }
            if ("function".equals(tool.get("type")) && tool.get("function") instanceof Map) {
                result.add(new LinkedHashMap<String, Object>(tool));
                continue;
            }
            Map<String, Object> function = new LinkedHashMap<String, Object>();
            copyIfPresent(tool, function, "name");
            copyIfPresent(tool, function, "description");
            copyIfPresent(tool, function, "parameters");
            copyIfPresent(tool, function, "strict");
            if (!function.containsKey("name")) {
                continue;
            }

            Map<String, Object> openAiTool = new LinkedHashMap<String, Object>();
            openAiTool.put("type", "function");
            openAiTool.put("function", function);
            result.add(openAiTool);
        }
        return result;
    }

    /**
     * tool_choice 支持 auto/required/none，也支持直接指定某个函数名。
     */
    private Object buildToolChoice(String toolChoice) {
        if (isBlank(toolChoice)) {
            return null;
        }
        if ("auto".equals(toolChoice)
            || "required".equals(toolChoice)
            || "none".equals(toolChoice)) {
            return toolChoice;
        }

        Map<String, Object> function = new LinkedHashMap<String, Object>();
        function.put("name", toolChoice);
        Map<String, Object> result = new LinkedHashMap<String, Object>();
        result.put("type", "function");
        result.put("function", function);
        return result;
    }

    private void copyIfPresent(
        Map<String, Object> source,
        Map<String, Object> target,
        String key
    ) {
        if (!source.containsKey(key) || source.get(key) == null) {
            return;
        }
        target.put(key, source.get(key));
    }

    private String asText(Object value) {
        return value == null ? null : String.valueOf(value);
    }

    private boolean isBlank(String value) {
        return value == null || value.trim().isEmpty();
    }
}
