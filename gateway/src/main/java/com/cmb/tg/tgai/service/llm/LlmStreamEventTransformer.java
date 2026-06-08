package com.cmb.tg.tgai.service.llm;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.time.OffsetDateTime;
import java.time.ZoneOffset;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * 上游 LLM SSE 流到网关统一事件流的转换器。
 *
 * <p>这个类是有状态的：一次请求创建一个实例，持续缓存尚未成行的文本和未完成的
 * tool call，直到上游结束或网关主动收尾。</p>
 */
class LlmStreamEventTransformer {

    private final ObjectMapper objectMapper;
    // 网络分块不保证刚好按 SSE 行切开，
    // 未遇到换行的尾部内容会暂存在这里。
    private final StringBuilder pendingText = new StringBuilder();
    // tool call 的 arguments 通常分多段返回，
    // 先缓存，等模型结束时再统一输出。
    private final Map<String, ToolCallBuffer> toolCalls =
        new LinkedHashMap<String, ToolCallBuffer>();
    private final Map<String, String> toolCallIndexAliases = new LinkedHashMap<String, String>();
    private int anonymousToolCallCounter = 0;
    private boolean doneEmitted = false;
    private boolean terminalCommentEmitted = false;
    // terminal 表示下游 SSE 已经完成收尾；后续上游 chunk 必须被忽略，
    // 避免出现 error/done 之后继续输出 text 或 tool_call 的协议错乱。
    private boolean terminal = false;
    // abortUpstream 只用于不可恢复的上游协议错误。它要求 service 停止继续消费
    // 当前上游 Flux，而不是安静地丢弃后续 chunk 直到连接自然结束。
    private boolean abortUpstream = false;
    private String lastStopReason;

    LlmStreamEventTransformer(ObjectMapper objectMapper) {
        this.objectMapper = objectMapper;
    }

    /**
     * 接收一段上游流文本，只处理其中已经完整到达的 SSE 行。
     */
    List<String> accept(String text) {
        if (terminal) {
            return Collections.emptyList();
        }
        if (text == null || text.isEmpty()) {
            return Collections.emptyList();
        }

        pendingText.append(text);
        List<String> events = new ArrayList<String>();
        int lineEnd;
        while ((lineEnd = indexOfLineFeed(pendingText)) >= 0) {
            String line = pendingText.substring(0, lineEnd);
            pendingText.delete(0, lineEnd + 1);
            events.addAll(processLine(removeTrailingCarriageReturn(line)));
        }
        return events;
    }

    /**
     * 上游 Flux 完成时的兜底收尾，确保残留文本、tool call 和 done 事件不会丢失。
     */
    List<String> finish() {
        List<String> events = new ArrayList<String>();
        if (!terminal && pendingText.length() > 0) {
            String line = pendingText.toString();
            pendingText.setLength(0);
            events.addAll(processLine(removeTrailingCarriageReturn(line)));
        }
        if (!doneEmitted) {
            events.addAll(flushToolCalls());
            events.add(buildDoneEvent(lastStopReason));
            doneEmitted = true;
        }
        if (!terminalCommentEmitted) {
            events.add(": done\n\n");
            terminalCommentEmitted = true;
        }
        terminal = true;
        return events;
    }

    boolean shouldAbortUpstream() {
        return abortUpstream;
    }

    /**
     * 只处理 SSE 的 data 行；注释行、空行和其他字段当前不参与业务转换。
     */
    private List<String> processLine(String line) {
        String trimmedLine = line == null ? "" : line.trim();
        if (trimmedLine.isEmpty() || trimmedLine.startsWith(":")) {
            return Collections.emptyList();
        }
        if (!trimmedLine.startsWith("data:")) {
            return Collections.emptyList();
        }

        String data = trimmedLine.substring("data:".length()).trim();
        if (data.isEmpty() || ",".equals(data)) {
            return Collections.emptyList();
        }
        if ("[DONE]".equals(data)) {
            return emitDoneFromProviderSignal();
        }
        return processJsonData(data);
    }

    /**
     * 解析 OpenAI-compatible 的流式 JSON。
     *
     * <p>根据上游字段转换成 text、thinking、tool_call、usage 或 done 事件。</p>
     */
    private List<String> processJsonData(String data) {
        List<String> events = new ArrayList<String>();
        try {
            JsonNode root = objectMapper.readTree(data);
            String finishReason = processChoices(root.path("choices"), events);
            if (finishReason != null) {
                lastStopReason = finishReason;
                events.addAll(flushToolCalls());
                events.addAll(buildUsageEvents(root.path("usage")));
                events.add(buildDoneEvent(finishReason));
                doneEmitted = true;
                return events;
            }
            events.addAll(buildUsageEvents(root.path("usage")));
            return events;
        } catch (Exception exception) {
            // 上游 JSON 已经无法解析，本次流的协议语义不可继续信任。
            // 这里立即输出 error 和终止注释，并让 service 取消后续上游消费。
            events.add(buildErrorEvent(
                "Invalid LLM upstream stream payload: " + exception.getMessage()
            ));
            events.add(": done\n\n");
            doneEmitted = true;
            terminalCommentEmitted = true;
            terminal = true;
            abortUpstream = true;
            return events;
        }
    }

    /**
     * 当前网关只消费 choices[0]，因为调用方期望的是一条连续的助手回复流。
     */
    private String processChoices(JsonNode choices, List<String> events) {
        if (!choices.isArray() || choices.size() == 0) {
            return null;
        }

        JsonNode choice = choices.get(0);
        JsonNode delta = choice.path("delta");
        addThinkingEvent(delta, events);
        addTextEvent(delta, events);
        addToolCallBuffers(delta.path("tool_calls"));

        JsonNode finishReason = choice.get("finish_reason");
        if (finishReason != null && !finishReason.isNull()) {
            return finishReason.asText();
        }
        return null;
    }

    /**
     * 兼容不同供应商的推理字段命名，并统一输出为 thinking 事件。
     */
    private void addThinkingEvent(JsonNode delta, List<String> events) {
        String reasoning = firstNonBlankText(
            delta.get("reasoning"),
            delta.get("reasoning_content"),
            delta.get("thinking")
        );
        if (!isBlank(reasoning)) {
            Map<String, Object> payload = baseEvent("thinking");
            payload.put("content", reasoning);
            events.add(serializeEvent(payload));
        }
    }

    private void addTextEvent(JsonNode delta, List<String> events) {
        JsonNode content = delta.get("content");
        if (content == null || content.isNull()) {
            return;
        }
        String text = content.asText();
        if (text.isEmpty()) {
            return;
        }
        Map<String, Object> payload = baseEvent("text");
        payload.put("content", text);
        events.add(serializeEvent(payload));
    }

    private void addToolCallBuffers(JsonNode toolCallsNode) {
        if (!toolCallsNode.isArray()) {
            return;
        }
        for (JsonNode toolCallNode : toolCallsNode) {
            appendToolCallBuffer(toolCallNode);
        }
    }

    /**
     * 把一段 tool call chunk 归并到同一个 buffer。
     *
     * <p>有些上游首个 chunk 带 id，后续 chunk 只带 index，所以这里同时维护
     * id 和 index 两种定位方式。</p>
     */
    private void appendToolCallBuffer(JsonNode toolCallNode) {
        String id = textOrNull(toolCallNode.get("id"));
        String index = textOrNull(toolCallNode.get("index"));
        String key = resolveToolCallKey(id, index);
        ToolCallBuffer buffer = toolCalls.get(key);
        if (buffer == null) {
            buffer = new ToolCallBuffer(defaultToolCallId(id, index, key));
            toolCalls.put(key, buffer);
        }
        if (!isBlank(id)) {
            buffer.id = id;
        }
        if (!isBlank(index)) {
            toolCallIndexAliases.put(index, key);
        }

        JsonNode functionNode = toolCallNode.path("function");
        String name = textOrNull(functionNode.get("name"));
        if (!isBlank(name)) {
            buffer.name = name;
        }
        String arguments = textOrNull(functionNode.get("arguments"));
        if (arguments != null) {
            buffer.arguments.append(arguments);
        }
    }

    /**
     * 统一网关的 token 统计字段，调用方不用关心上游 usage 的细节结构。
     */
    private List<String> buildUsageEvents(JsonNode usage) {
        if (usage == null || usage.isMissingNode() || usage.isNull()) {
            return Collections.emptyList();
        }

        Map<String, Object> usagePayload = new LinkedHashMap<String, Object>();
        usagePayload.put("prompt_tokens", intOrZero(usage.get("prompt_tokens")));
        usagePayload.put("prompt_cached_tokens", cachedPromptTokens(usage));
        usagePayload.put("prompt_cache_creation_tokens", null);
        usagePayload.put("prompt_image_tokens", null);
        usagePayload.put("completion_tokens", completionTokens(usage));
        usagePayload.put("total_tokens", intOrZero(usage.get("total_tokens")));

        Map<String, Object> payload = baseEvent("usage");
        payload.put("usage", usagePayload);
        return Collections.singletonList(serializeEvent(payload));
    }

    private List<String> emitDoneFromProviderSignal() {
        if (doneEmitted) {
            return Collections.emptyList();
        }
        List<String> events = new ArrayList<String>();
        events.addAll(flushToolCalls());
        events.add(buildDoneEvent(lastStopReason));
        doneEmitted = true;
        terminal = true;
        return events;
    }

    /**
     * 将缓存的 tool call 转成事件。只有拿到工具名的调用才会输出。
     */
    private List<String> flushToolCalls() {
        if (toolCalls.isEmpty()) {
            return Collections.emptyList();
        }

        List<String> events = new ArrayList<String>();
        for (ToolCallBuffer buffer : toolCalls.values()) {
            if (isBlank(buffer.name)) {
                continue;
            }
            Map<String, Object> payload = baseEvent("tool_call");
            payload.put("tool", buffer.name);
            payload.put("args", parseToolArguments(buffer.arguments.toString()));
            payload.put("tool_call_id", buffer.id);
            payload.put("display_name", buffer.name);
            events.add(serializeEvent(payload));
        }
        toolCalls.clear();
        toolCallIndexAliases.clear();
        return events;
    }

    /**
     * tool arguments 理论上是 JSON 对象；解析失败时保留原始字符串，避免信息丢失。
     */
    private Map<String, Object> parseToolArguments(String arguments) {
        if (isBlank(arguments)) {
            return new LinkedHashMap<String, Object>();
        }
        try {
            Map<String, Object> parsed = objectMapper.readValue(
                arguments,
                new TypeReference<Map<String, Object>>() {
                }
            );
            return parsed;
        } catch (Exception exception) {
            Map<String, Object> fallback = new LinkedHashMap<String, Object>();
            fallback.put("_raw", arguments);
            return fallback;
        }
    }

    private String buildDoneEvent(String stopReason) {
        Map<String, Object> payload = baseEvent("done");
        payload.put("stop_reason", stopReason);
        return serializeEvent(payload);
    }

    private String buildErrorEvent(String message) {
        Map<String, Object> payload = baseEvent("error");
        payload.put("error", message);
        return serializeEvent(payload);
    }

    private Map<String, Object> baseEvent(String type) {
        Map<String, Object> payload = new LinkedHashMap<String, Object>();
        payload.put("type", type);
        payload.put("timestamp", OffsetDateTime.now(ZoneOffset.UTC).toString());
        return payload;
    }

    private String serializeEvent(Map<String, Object> payload) {
        try {
            return "data: " + objectMapper.writeValueAsString(payload) + "\n\n";
        } catch (Exception exception) {
            return "data: {\"type\":\"error\",\"error\":\"LLM event serialization failed\"}\n\n";
        }
    }

    /**
     * 根据上游 chunk 中可用的 id/index 选择稳定的缓存 key。
     */
    private String resolveToolCallKey(String id, String index) {
        if (!isBlank(id)) {
            String key = "id:" + id;
            if (!isBlank(index)) {
                toolCallIndexAliases.put(index, key);
            }
            return key;
        }
        if (!isBlank(index) && toolCallIndexAliases.containsKey(index)) {
            return toolCallIndexAliases.get(index);
        }
        if (!isBlank(index)) {
            return "index:" + index;
        }
        String key = "anonymous:" + anonymousToolCallCounter;
        anonymousToolCallCounter++;
        return key;
    }

    private String defaultToolCallId(String id, String index, String key) {
        if (!isBlank(id)) {
            return id;
        }
        if (!isBlank(index)) {
            return "call_" + index;
        }
        return key.replace(":", "_");
    }

    private Integer cachedPromptTokens(JsonNode usage) {
        JsonNode details = usage.path("prompt_tokens_details");
        JsonNode cachedTokens = details.get("cached_tokens");
        if (cachedTokens == null || cachedTokens.isNull()) {
            return null;
        }
        return Integer.valueOf(cachedTokens.asInt());
    }

    private int completionTokens(JsonNode usage) {
        int completionTokens = intOrZero(usage.get("completion_tokens"));
        JsonNode details = usage.path("completion_tokens_details");
        JsonNode reasoningTokens = details.get("reasoning_tokens");
        if (reasoningTokens == null || reasoningTokens.isNull()) {
            return completionTokens;
        }
        return completionTokens + reasoningTokens.asInt();
    }

    private int intOrZero(JsonNode node) {
        if (node == null || node.isNull()) {
            return 0;
        }
        return node.asInt();
    }

    private String firstNonBlankText(JsonNode first, JsonNode second, JsonNode third) {
        String firstText = textOrNull(first);
        if (!isBlank(firstText)) {
            return firstText;
        }
        String secondText = textOrNull(second);
        if (!isBlank(secondText)) {
            return secondText;
        }
        return textOrNull(third);
    }

    private String textOrNull(JsonNode node) {
        if (node == null || node.isNull() || node.isMissingNode()) {
            return null;
        }
        return node.asText();
    }

    private int indexOfLineFeed(StringBuilder builder) {
        for (int i = 0; i < builder.length(); i++) {
            if (builder.charAt(i) == '\n') {
                return i;
            }
        }
        return -1;
    }

    private String removeTrailingCarriageReturn(String value) {
        if (value != null && value.endsWith("\r")) {
            return value.substring(0, value.length() - 1);
        }
        return value;
    }

    private boolean isBlank(String value) {
        return value == null || value.trim().isEmpty();
    }

    /**
     * 保存一个 tool call 在流式传输过程中的中间状态。
     */
    private static final class ToolCallBuffer {

        private String id;
        private String name;
        private final StringBuilder arguments = new StringBuilder();

        private ToolCallBuffer(String id) {
            this.id = id;
        }
    }
}
