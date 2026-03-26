package com.buagent.gateway.store.mapper;

import com.buagent.gateway.store.entity.InboundMessageEntity;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

@Mapper
public interface InboundMessageMapper {
    int insert(InboundMessageEntity entity);

    int markDelivering(
        @Param("id") Long id,
        @Param("deliveryId") String deliveryId,
        @Param("leaseExpiresAt") Long leaseExpiresAt
    );

    int updateLease(@Param("id") Long id, @Param("leaseExpiresAt") Long leaseExpiresAt);

    int markConsumed(@Param("id") Long id, @Param("deliveryId") String deliveryId);
}
