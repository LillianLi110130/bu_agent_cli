package com.cmb.tg.tgai.infrastructure.common.rest.wecom.dto.response;

import lombok.Getter;
import lombok.Setter;

import java.util.List;

@Getter
@Setter
public class GroupBaseInfoResponseDTO {

    private String rtnCod;

    private String errMsg;

    private ReturnInfoDTO returnInfo;

    private List<GroupInfoDTO> infBdy;

    @Getter
    @Setter
    public static class ReturnInfoDTO {

        private String returnCode;

        private String errorMsg;
    }

    @Getter
    @Setter
    public static class GroupInfoDTO {

        private String groupNm;

        private String groupOwnerId;

        private String groupOwnerYstUserId;

        private String groupOwnerNm;

        private List<GroupMemberDTO> groupMemberList;
    }

    @Getter
    @Setter
    public static class GroupMemberDTO {

        private String custId;

        private String custNm;

        private String userType;

        private String joinTm;

        private String joinScene;
    }
}
