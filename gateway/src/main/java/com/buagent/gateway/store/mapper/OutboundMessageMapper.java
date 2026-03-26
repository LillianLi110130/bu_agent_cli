package com.buagent.gateway.store.mapper;

import com.buagent.gateway.store.entity.OutboundMessageEntity;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

@Mapper
public interface OutboundMessageMapper {
    int insert(OutboundMessageEntity entity);

    int updateStatus(@Param("id") Long id, @Param("status") String status);
}
