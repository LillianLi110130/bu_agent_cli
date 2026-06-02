package com.cmb.tg.tgai.infrastructure.common.rest.wecom.dto.response;

import lombok.Getter;
import lombok.Setter;

import java.util.List;

@Getter
@Setter
public class CustomerBaseInfoResponseDTO {

    private String rtnCod;

    private String errMsg;

    private ReturnInfoDTO returnInfo;

    private List<CustomerInfoDTO> infBdy;

    @Getter
    @Setter
    public static class ReturnInfoDTO {

        private String returnCode;

        private String errorMsg;
    }

    @Getter
    @Setter
    public static class CustomerInfoDTO {

        private String custId;

        private String custNm;

        private String custAvatar;

        private List<CustomerAuthDTO> custAuth;
    }

    @Getter
    @Setter
    public static class CustomerAuthDTO {

        private String comId;

        private String comUid;

        private String comNm;

        private String comPosition;
    }
}
