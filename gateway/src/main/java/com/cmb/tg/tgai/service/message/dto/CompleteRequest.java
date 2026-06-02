package com.cmb.tg.tgai.service.message.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class CompleteRequest {
    @JsonProperty("session_key")
    private String sessionKey;

    @JsonProperty("worker_id")
    private String workerId;

    @JsonProperty("delivery_id")
    private String deliveryId;

    @JsonProperty("final_content")
    private String finalContent;

    @JsonProperty("final_status")
    private String finalStatus;

    @JsonProperty("error_code")
    private String errorCode;

    @JsonProperty("error_message")
    private String errorMessage;

    private String source;
}
