package com.cmb.tg.tgai.service.wecom.dto;

import lombok.Builder;
import lombok.Getter;

import java.time.LocalDateTime;

@Getter
@Builder
public class ChatConversationReplayQuery {

    private String roomId;

    private LocalDateTime startTime;

    private LocalDateTime endTime;
}
