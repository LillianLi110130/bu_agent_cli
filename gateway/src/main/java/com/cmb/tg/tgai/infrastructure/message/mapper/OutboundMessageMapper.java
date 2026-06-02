package com.cmb.tg.tgai.infrastructure.message.mapper;

import com.cmb.tg.tgai.infrastructure.message.entity.OutboundMessageEntity;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

@Mapper
public interface OutboundMessageMapper {
    int insert(OutboundMessageEntity entity);

    OutboundMessageEntity fetchOutboundMessage(String sessionKey);

    int updateCurrentStatus(@Param("id") Long id, @Param("currentStatus") String currentStatus, @Param("nextStatus") String nextStatus);
}
