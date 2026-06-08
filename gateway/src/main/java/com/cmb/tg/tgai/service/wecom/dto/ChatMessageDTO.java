package com.cmb.tg.tgai.service.wecom.dto;

import lombok.Builder;
import lombok.Getter;

import java.time.LocalDateTime;

@Getter
@Builder
public class ChatMessageDTO {

    private String msgId;

    private String roomId;

    private String fromUser;

    private String wthrFromCm;

    private String toList;

    private String singleReceiverId;

    private String msgBody;

    private String msgType;

    private String acsKey;

    private Long msgTimeLong;

    private LocalDateTime msgTime;
}
