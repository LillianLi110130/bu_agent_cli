package com.buagent.gateway.session;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class InFlightDelivery {
    private Long messageId;
    private String deliveryId;
    private String workerId;
    private Long epoch;
    private String content;
    private Long leaseExpiresAt;
}
