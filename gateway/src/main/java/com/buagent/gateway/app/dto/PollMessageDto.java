package com.buagent.gateway.app.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class PollMessageDto {
    @JsonProperty("message_id")
    private Long messageId;

    @JsonProperty("delivery_id")
    private String deliveryId;

    private Long epoch;
    private String content;
}
