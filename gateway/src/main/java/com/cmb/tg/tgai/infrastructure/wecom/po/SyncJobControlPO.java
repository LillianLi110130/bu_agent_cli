package com.cmb.tg.tgai.infrastructure.wecom.po;

import lombok.Getter;
import lombok.Setter;

import java.time.LocalDateTime;

@Getter
@Setter
public class SyncJobControlPO {

    private String jobName;

    private LocalDateTime lockUntil;
}
