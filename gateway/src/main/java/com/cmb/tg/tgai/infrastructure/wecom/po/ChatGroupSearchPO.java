package com.cmb.tg.tgai.infrastructure.wecom.po;

import lombok.Getter;
import lombok.Setter;

import java.time.LocalDateTime;

@Getter
@Setter
public class ChatGroupSearchPO {

    private String roomId;

    private String groupName;

    private String groupOwnerName;

    private LocalDateTime lastMsgTime;
}
