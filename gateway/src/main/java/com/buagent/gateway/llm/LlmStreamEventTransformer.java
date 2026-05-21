package com.buagent.gateway.llm;

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

class LlmStreamEventTransformer {

    private final ObjectMapper objectMapper;
    private final StringBuilder pendingText = new StringBuilder();
    private final Map<String, ToolCallBuffer> toolCalls =
        new LinkedHashMap<String, ToolCallBuffer>();
    private final Map<String, String> toolCallIndexAliases = new LinkedHashMap<String, String>();
    private int anonymousToolCallCounter = 0;
    private boolean doneEmitted = false;
    private boolean terminalCommentEmitted = false;
    private String lastStopReason;

    LlmStreamEventTransformer(ObjectMapper objectMapper) {
        this.objectMapper = objectMapper;
    }

    List<String> accept(String text) {
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

    List<String> finish() {
        List<String> events = new ArrayList<String>();
        if (pendingText.length() > 0) {
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
        return events;
    }

    private List<String> processLine(String line) {
        String trimmedLine = line == null ? "" : line.trim();
        if (trimmedLine.isEmpty() || trimmedLine.startsWith(":")) {
            return Collections.emptyList();
        }
        if (!trimmedLine.startsWith("data:")) {
            return Collections.emptyList();
        }

        String data = trimmedLine.substring("data:".length()).trim();
        if ("[DONE]".equals(data)) {
            return emitDoneFromProviderSignal();
        }
        return processJsonData(data);
    }

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
            events.add(buildErrorEvent(
                "Invalid LLM upstream stream payload: " + exception.getMessage()
            ));
            events.add(": done\n\n");
            doneEmitted = true;
            terminalCommentEmitted = true;
            return events;
        }
    }

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
        return events;
    }

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

    private static final class ToolCallBuffer {

        private String id;
        private String name;
        private final StringBuilder arguments = new StringBuilder();

        private ToolCallBuffer(String id) {
            this.id = id;
        }
    }
}
