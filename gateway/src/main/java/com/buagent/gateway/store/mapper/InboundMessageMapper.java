package com.buagent.gateway.store.mapper;

import com.buagent.gateway.store.entity.InboundMessageEntity;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

@Mapper
public interface InboundMessageMapper {
    int insert(InboundMessageEntity entity);

    InboundMessageEntity findFirstBySessionKeyAndStatus(
        @Param("sessionKey") String sessionKey,
        @Param("status") String status
    );

    int updateStatus(
        @Param("id") Long id,
        @Param("currentStatus") String currentStatus,
        @Param("nextStatus") String nextStatus
    );
}
