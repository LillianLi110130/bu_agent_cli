package com.buagent.gateway.app.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;
import java.util.Map;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class LlmQueryRequest {

    private String model;

    private List<Map<String, Object>> messages;

    private List<Map<String, Object>> tools;

    @JsonProperty("tool_choice")
    private String toolChoice;

    private Map<String, Object> metadata;
}
