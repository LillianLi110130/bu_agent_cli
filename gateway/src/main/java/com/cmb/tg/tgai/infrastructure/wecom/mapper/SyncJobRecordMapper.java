package com.cmb.tg.tgai.infrastructure.wecom.mapper;

import com.cmb.tg.tgai.infrastructure.wecom.po.SyncJobRecordPO;

public interface SyncJobRecordMapper {

    int insert(SyncJobRecordPO po);

    int updateStatus(SyncJobRecordPO po);
}
