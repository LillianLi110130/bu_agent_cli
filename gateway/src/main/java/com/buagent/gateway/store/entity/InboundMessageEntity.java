package com.buagent.gateway.store.entity;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class InboundMessageEntity {
    private Long id;
    private String sessionKey;
    private Long epoch;
    private String content;
    private String status;
    private String deliveryId;
    private Long leaseExpiresAt;
    private Long createdAt;
}
