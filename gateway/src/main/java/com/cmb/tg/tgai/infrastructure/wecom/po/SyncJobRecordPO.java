package com.cmb.tg.tgai.infrastructure.wecom.po;

import lombok.Getter;
import lombok.Setter;

import java.time.LocalDateTime;

@Getter
@Setter
public class SyncJobRecordPO {

    private Long id;

    private String jobName;

    private LocalDateTime startTime;

    private LocalDateTime endTime;

    private String runStatus;

    private String errorMsg;
}
