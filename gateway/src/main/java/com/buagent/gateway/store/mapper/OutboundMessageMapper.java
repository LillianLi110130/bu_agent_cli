package com.buagent.gateway.store.mapper;

import com.buagent.gateway.store.entity.OutboundMessageEntity;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

@Mapper
public interface OutboundMessageMapper {
    int insert(OutboundMessageEntity entity);

    OutboundMessageEntity findLatestBySessionKey(@Param("sessionKey") String sessionKey);

    OutboundMessageEntity findFirstBySessionKeyAndStatus(
        @Param("sessionKey") String sessionKey,
        @Param("status") String status
    );

    OutboundMessageEntity findFirstPendingBySessionKey(@Param("sessionKey") String sessionKey);

    int updateStatus(
        @Param("id") Long id,
        @Param("currentStatus") String currentStatus,
        @Param("nextStatus") String nextStatus
    );
}
