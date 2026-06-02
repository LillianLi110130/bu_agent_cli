package com.cmb.tg.tgai.controller.wecom.vo.response;

import lombok.Builder;
import lombok.Getter;

@Getter
@Builder
public class WecomMessageReceiveResponseVO {

    private String code;

    private String message;
}
