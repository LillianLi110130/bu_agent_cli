package com.cmb.tg.tgai.service.wecom.dto;

import lombok.Builder;
import lombok.Getter;

import java.time.LocalDateTime;
import java.util.List;

@Getter
@Builder
public class ChatConversationReplayResult {

    private String roomId;

    private String groupName;

    private Long total;

    private List<MessageItem> messages;

    private String transcript;

    @Getter
    @Builder
    public static class MessageItem {

        private String msgId;

        private String msgType;

        private LocalDateTime msgTime;

        private String fromUser;

        private String fromUserName;

        private String fromUserType;

        private String msgBody;

        private String displayText;
    }
}
