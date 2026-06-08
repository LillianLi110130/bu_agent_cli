package com.cmb.tg.tgai.infrastructure.message.entity;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class OutboundMessageEntity {
    private Long id;
    private String sessionKey;
    private String source;
    private String content;
    private String status;
    private LocalDateTime createdAt;
}
