package com.cmb.tg.tgai.controller.wecom.vo.response;

import com.fasterxml.jackson.annotation.JsonFormat;
import lombok.Builder;
import lombok.Getter;

import java.time.LocalDateTime;
import java.util.List;

@Getter
@Builder
public class ChatGroupSearchResponseVO {

    private Long total;

    private List<GroupItemVO> list;

    @Getter
    @Builder
    public static class GroupItemVO {

        private String roomId;

        private String groupName;

        private String groupOwnerName;

        @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
        private LocalDateTime lastMsgTime;
    }
}
