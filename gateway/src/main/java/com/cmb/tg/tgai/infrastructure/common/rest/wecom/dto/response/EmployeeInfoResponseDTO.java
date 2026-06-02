package com.cmb.tg.tgai.infrastructure.common.rest.wecom.dto.response;

import lombok.Getter;
import lombok.Setter;

import java.util.List;

@Getter
@Setter
public class EmployeeInfoResponseDTO {

    private String rtnCod;

    private String errMsg;

    private ReturnInfoDTO returnInfo;

    private List<EmployeeDTO> infBdy;

    @Getter
    @Setter
    public static class ReturnInfoDTO {

        private String returnCode;

        private String errorMsg;
    }

    @Getter
    @Setter
    public static class EmployeeDTO {

        private String gender;
        private String mobile;
        private Integer mainDept;
        private String telephone;
        private String avatar;
        private String follow;
        private String openUserid;
        private String crmUserId;
        private String mainDeptNm;
        private String crtTm;
        private String qrCodeUrl;
        private String thumbAvatar;
        private String name;
        private String bbkYstIntId;
        private String alias;
        private String position;
        private String id;
        private String sapId;
        private String email;
        private String status;
    }
}
