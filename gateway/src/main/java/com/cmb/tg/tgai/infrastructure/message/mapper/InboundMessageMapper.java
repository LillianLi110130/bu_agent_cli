package com.cmb.tg.tgai.infrastructure.message.mapper;

import com.cmb.tg.tgai.infrastructure.message.entity.InboundMessageEntity;
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

    InboundMessageEntity fetchInboundMessage(String sessionKey);
}
