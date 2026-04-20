package com.buagent.gateway.store.mapper;

import com.buagent.gateway.store.entity.OnlineWorkerEntity;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

@Mapper
public interface OnlineWorkerMapper {
    OnlineWorkerEntity findByWorkerId(@Param("workerId") String workerId);

    int insert(OnlineWorkerEntity entity);

    int updateStatus(@Param("workerId") String workerId, @Param("status") String status);
}
