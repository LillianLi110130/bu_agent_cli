package com.cmb.tg.tgai.infrastructure.wecom.po;

import lombok.Getter;
import lombok.Setter;

import java.time.LocalDateTime;

@Getter
@Setter
public class ChatMessagePO {

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
