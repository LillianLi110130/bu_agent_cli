package com.cmb.tg.tgai.service.wecom;

import com.cmb.tg.tgai.infrastructure.wecom.mapper.SyncJobControlMapper;
import com.cmb.tg.tgai.infrastructure.wecom.mapper.SyncJobRecordMapper;
import com.cmb.tg.tgai.infrastructure.wecom.po.SyncJobRecordPO;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.concurrent.Callable;

@Slf4j
@Service
@RequiredArgsConstructor
public class SyncJobService {

    private static final int LOCK_MINUTES = 10;

    private static final int MAX_ERROR_MSG_LENGTH = 512;

    private static final String RUNNING = "RUNNING";

    private static final String SUCCESS = "SUCCESS";

    private static final String FAILED = "FAILED";

    private final SyncJobControlMapper syncJobControlMapper;

    private final SyncJobRecordMapper syncJobRecordMapper;

    public void runWithControl(final String jobName, final Callable<Void> callable) {
        int acquired = syncJobControlMapper.acquire(jobName, LOCK_MINUTES);
        if (acquired == 0) {
            log.info("sync job skip, lock not acquired, jobName={}", jobName);
            return;
        }

        SyncJobRecordPO recordPO = new SyncJobRecordPO();
        try {
            recordPO.setJobName(jobName);
            recordPO.setStartTime(LocalDateTime.now());
            recordPO.setRunStatus(RUNNING);
            syncJobRecordMapper.insert(recordPO);

            callable.call();
            recordPO.setEndTime(LocalDateTime.now());
            recordPO.setRunStatus(SUCCESS);
            syncJobRecordMapper.updateStatus(recordPO);
        } catch (Exception ex) {
            log.error("sync job execute failed, jobName={}", jobName, ex);
            recordPO.setEndTime(LocalDateTime.now());
            recordPO.setRunStatus(FAILED);
            String errorMsg = ex.getMessage();
            if (errorMsg != null && errorMsg.length() > MAX_ERROR_MSG_LENGTH) {
                errorMsg = errorMsg.substring(0, MAX_ERROR_MSG_LENGTH);
            }
            recordPO.setErrorMsg(errorMsg);
            syncJobRecordMapper.updateStatus(recordPO);
        } finally {
            syncJobControlMapper.release(jobName);
        }
    }
}
