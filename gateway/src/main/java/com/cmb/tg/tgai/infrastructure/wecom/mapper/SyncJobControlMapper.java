package com.cmb.tg.tgai.infrastructure.wecom.mapper;

import org.apache.ibatis.annotations.Param;

public interface SyncJobControlMapper {

    int acquire(@Param("jobName") String jobName, @Param("lockMinutes") Integer lockMinutes);

    int release(@Param("jobName") String jobName);
}
