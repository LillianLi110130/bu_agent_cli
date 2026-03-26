package com.buagent.gateway.store.mapper;

import com.buagent.gateway.store.entity.SessionStateEntity;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

@Mapper
public interface SessionStateMapper {
    SessionStateEntity findBySessionKey(@Param("sessionKey") String sessionKey);

    int insert(SessionStateEntity entity);

    int updateCurrentEpoch(@Param("sessionKey") String sessionKey, @Param("currentEpoch") Long currentEpoch);
}
