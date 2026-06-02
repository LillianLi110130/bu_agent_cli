package com.cmb.tg.tgai.infrastructure.common.rest.wecom;

import lombok.Getter;
import lombok.Setter;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

@Getter
@Setter
@Component
@ConfigurationProperties(prefix = "wecom-sync")
public class WecomSyncProperties {

    private String appId;

    private String privateKey;

    private String secret;

    private String verify = "SM3withSM2";

    private String groupBaseInfoUrl;

    private String customerBaseInfoUrl;

    private String employeeInfoUrl;
}
