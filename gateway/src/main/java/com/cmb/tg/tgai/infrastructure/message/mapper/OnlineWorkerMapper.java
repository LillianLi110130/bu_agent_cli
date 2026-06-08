package com.cmb.tg.tgai.infrastructure.message.mapper;

import com.cmb.tg.tgai.infrastructure.message.entity.OnlineWorker;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

@Mapper
public interface OnlineWorkerMapper {
    OnlineWorker findByWorkerId(@Param("workerId") String workerId);

    OnlineWorker findByWorkerIdPrefix(@Param("workerIdPrefix") String workerIdPrefix);

    int insert(OnlineWorker entity);

    int updateStatus(@Param("workerId") String workerId, @Param("status") String status);
}
