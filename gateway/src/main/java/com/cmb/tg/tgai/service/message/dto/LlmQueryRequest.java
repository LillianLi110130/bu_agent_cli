package com.cmb.tg.tgai.service.message.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;
import java.util.Map;

/**
 * 网关对外暴露的 LLM 查询请求。
 *
 * <p>字段尽量贴近 OpenAI chat completions 协议，方便客户端复用已有请求结构。</p>
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
public class LlmQueryRequest {

    /**
     * 客户端请求的模型名。在网关配置了 routes 时，它通常是一个模型别名。
     */
    private String model;

    /**
     * 对话消息列表，内部元素保留为 Map，便于兼容文本、多模态和工具调用消息。
     */
    private List<Map<String, Object>> messages;

    /**
     * 可用工具列表，支持 OpenAI function tool 结构或网关侧的简化结构。
     */
    private List<Map<String, Object>> tools;

    /**
     * 工具选择策略：auto、required、none，或指定某个工具名。
     */
    @JsonProperty("tool_choice")
    private String toolChoice;

    /**
     * 预留给调用方透传请求级上下文，当前 Java gateway 不直接消费。
     */
    private Map<String, Object> metadata;
}
