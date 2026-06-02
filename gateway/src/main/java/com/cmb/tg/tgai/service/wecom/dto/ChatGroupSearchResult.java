package com.cmb.tg.tgai.service.wecom.dto;

import lombok.Builder;
import lombok.Getter;

import java.time.LocalDateTime;
import java.util.List;

@Getter
@Builder
public class ChatGroupSearchResult {

    private Long total;

    private List<GroupItem> list;

    @Getter
    @Builder
    public static class GroupItem {

        private String roomId;

        private String groupName;

        private String groupOwnerName;

        private LocalDateTime lastMsgTime;
    }
}
