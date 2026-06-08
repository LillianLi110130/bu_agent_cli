package com.cmb.tg.tgai.infrastructure.wecom.mapper;

import com.cmb.tg.tgai.infrastructure.wecom.po.ChatConversationMessagePO;
import com.cmb.tg.tgai.infrastructure.wecom.po.ChatMessagePO;
import org.apache.ibatis.annotations.Param;

import java.time.LocalDateTime;
import java.util.List;

public interface ChatMessageMapper {

    int batchInsertIgnore(@Param("list") List<ChatMessagePO> list);

    List<String> selectActiveRoomIds(@Param("activeFromTime") LocalDateTime activeFromTime);

    List<String> selectSingleCustomerFromUserIds(@Param("activeFromTime") LocalDateTime activeFromTime);

    List<String> selectSingleEmployeeFromUserIds(@Param("activeFromTime") LocalDateTime activeFromTime);

    List<String> selectSingleReceiverIds(@Param("activeFromTime") LocalDateTime activeFromTime);

    List<ChatConversationMessagePO> selectReplayMessages(
            @Param("roomId") String roomId,
            @Param("startTime") LocalDateTime startTime,
            @Param("endTime") LocalDateTime endTime);
}
