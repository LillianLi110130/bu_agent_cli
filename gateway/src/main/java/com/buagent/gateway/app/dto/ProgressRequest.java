package com.buagent.gateway.app.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class ProgressRequest {
    @JsonProperty("session_key")
    private String sessionKey;

    @JsonProperty("worker_id")
    private String workerId;

    private String content;

    private String source;
}
