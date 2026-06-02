package com.cmb.tg.tgai.service.wecom;

import com.cmb.tg.tgai.infrastructure.wecom.mapper.ChatCustomerAuthMapper;
import com.cmb.tg.tgai.infrastructure.wecom.mapper.ChatGroupMemberMapper;
import com.cmb.tg.tgai.infrastructure.wecom.po.ChatCustomerAuthPO;
import com.cmb.tg.tgai.infrastructure.wecom.po.ChatGroupMemberPO;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.util.CollectionUtils;

import java.util.List;

@Service
@RequiredArgsConstructor
public class SyncPersistenceService {

    private final ChatGroupMemberMapper chatGroupMemberMapper;

    private final ChatCustomerAuthMapper chatCustomerAuthMapper;

    @Transactional(rollbackFor = Exception.class)
    public void refreshGroupMembers(final String roomId, final List<ChatGroupMemberPO> memberPOList) {
        chatGroupMemberMapper.deleteByRoomId(roomId);
        if (CollectionUtils.isEmpty(memberPOList)) {
            return;
        }
        chatGroupMemberMapper.batchInsert(memberPOList);
    }

    @Transactional(rollbackFor = Exception.class)
    public void refreshCustomerAuth(final String custId, final List<ChatCustomerAuthPO> authPOList) {
        chatCustomerAuthMapper.deleteByCustId(custId);
        if (CollectionUtils.isEmpty(authPOList)) {
            return;
        }
        chatCustomerAuthMapper.batchInsert(authPOList);
    }
}
