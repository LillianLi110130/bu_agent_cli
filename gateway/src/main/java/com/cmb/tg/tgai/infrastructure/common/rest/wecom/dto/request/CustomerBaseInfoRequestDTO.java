package com.cmb.tg.tgai.infrastructure.common.rest.wecom.dto.request;

import lombok.Builder;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
@Builder
public class CustomerBaseInfoRequestDTO {

    private String bbkId;

    private String custId;
}
