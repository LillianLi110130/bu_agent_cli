package com.cmb.tg.tgai.infrastructure.wecom.po;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class ChatGroupMemberPO {

    private String roomId;

    private String memberId;

    private String memberRemark;

    private String userType;

    private String joinTime;

    private String joinScene;
}
