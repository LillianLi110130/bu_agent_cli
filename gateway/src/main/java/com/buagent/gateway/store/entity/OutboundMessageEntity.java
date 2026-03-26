package com.buagent.gateway.store.entity;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class OutboundMessageEntity {
    private Long id;
    private String sessionKey;
    private Long epoch;
    private Long inboundMessageId;
    private String content;
    private String status;
    private Long createdAt;
}
