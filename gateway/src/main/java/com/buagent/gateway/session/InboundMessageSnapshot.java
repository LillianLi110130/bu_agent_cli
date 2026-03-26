package com.buagent.gateway.session;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class InboundMessageSnapshot {
    private Long messageId;
    private String sessionKey;
    private Long epoch;
    private String content;
}
