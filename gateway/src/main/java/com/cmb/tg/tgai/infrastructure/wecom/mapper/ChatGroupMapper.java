package com.cmb.tg.tgai.infrastructure.wecom.mapper;

import com.cmb.tg.tgai.infrastructure.wecom.po.ChatGroupPO;
import com.cmb.tg.tgai.infrastructure.wecom.po.ChatGroupSearchPO;
import org.apache.ibatis.annotations.Param;

import java.util.List;

public interface ChatGroupMapper {

    int batchUpsert(List<ChatGroupPO> list);

    List<ChatGroupSearchPO> searchGroups(
            @Param("groupName") String groupName,
            @Param("groupOwnerName") String groupOwnerName);

    ChatGroupPO selectByRoomId(@Param("roomId") String roomId);
}
