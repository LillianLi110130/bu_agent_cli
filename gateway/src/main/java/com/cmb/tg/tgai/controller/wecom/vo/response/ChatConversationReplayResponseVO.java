package com.cmb.tg.tgai.controller.wecom.vo.response;

import com.fasterxml.jackson.annotation.JsonFormat;
import lombok.Builder;
import lombok.Getter;

import java.time.LocalDateTime;
import java.util.List;

@Getter
@Builder
public class ChatConversationReplayResponseVO {

    private String roomId;

    private String groupName;

    private Long total;

    private List<MessageItemVO> messages;

    private String transcript;

    @Getter
    @Builder
    public static class MessageItemVO {

        private String msgId;

        private String msgType;

        @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
        private LocalDateTime msgTime;

        private String fromUser;

        private String fromUserName;

        private String fromUserType;

        private String msgBody;

        private String displayText;
    }
}
