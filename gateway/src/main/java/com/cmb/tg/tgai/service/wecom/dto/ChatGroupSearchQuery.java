package com.cmb.tg.tgai.service.wecom.dto;

import lombok.Builder;
import lombok.Getter;

@Getter
@Builder
public class ChatGroupSearchQuery {

    private String groupName;

    private String groupOwnerName;
}
