package com.cmb.tg.tgai.infrastructure.wecom.po;

import lombok.Getter;
import lombok.Setter;

import java.time.LocalDateTime;

@Getter
@Setter
public class ChatConversationMessagePO {

    private String msgId;

    private String msgType;

    private LocalDateTime msgTime;

    private String fromUser;

    private String fromUserName;

    private String fromUserType;

    private String msgBody;
}
