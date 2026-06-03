package com.cmb.tg.tgai.infrastructure.message.mapper;

import com.cmb.tg.tgai.infrastructure.message.entity.OnlineWorkerEntity;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

@Mapper
public interface OnlineWorkerMapper {
    OnlineWorkerEntity findByWorkerId(@Param("workerId") String workerId);

    OnlineWorkerEntity findByWorkerIdPrefix(@Param("workerIdPrefix") String workerIdPrefix);

    int insert(OnlineWorkerEntity entity);

    int updateStatus(@Param("workerId") String workerId, @Param("status") String status);
}
