package com.cmb.tg.tgai.infrastructure.wecom.mapper;

import com.cmb.tg.tgai.infrastructure.wecom.po.ChatGroupMemberPO;
import org.apache.ibatis.annotations.Param;

import java.time.LocalDateTime;
import java.util.List;

public interface ChatGroupMemberMapper {

    int deleteByRoomId(@Param("roomId") String roomId);

    int batchInsert(@Param("list") List<ChatGroupMemberPO> list);

    List<String> selectCustomerMemberIdsByActiveRooms(@Param("activeFromTime") LocalDateTime activeFromTime);

    List<String> selectEmployeeMemberIdsByActiveRooms(@Param("activeFromTime") LocalDateTime activeFromTime);
}
