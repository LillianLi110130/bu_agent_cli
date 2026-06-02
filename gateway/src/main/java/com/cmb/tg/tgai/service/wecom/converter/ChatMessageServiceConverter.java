package com.cmb.tg.tgai.service.wecom.converter;

import com.cmb.tg.tgai.infrastructure.wecom.po.ChatMessagePO;
import com.cmb.tg.tgai.service.wecom.dto.ChatMessageDTO;
import org.springframework.stereotype.Component;

@Component
public class ChatMessageServiceConverter {

    public ChatMessagePO toPo(final ChatMessageDTO dto) {
        ChatMessagePO po = new ChatMessagePO();
        po.setMsgId(dto.getMsgId());
        po.setRoomId(dto.getRoomId());
        po.setFromUser(dto.getFromUser());
        po.setWthrFromCm(dto.getWthrFromCm());
        po.setToList(dto.getToList());
        po.setSingleReceiverId(dto.getSingleReceiverId());
        po.setMsgBody(dto.getMsgBody());
        po.setMsgType(dto.getMsgType());
        po.setAcsKey(dto.getAcsKey());
        po.setMsgTimeLong(dto.getMsgTimeLong());
        po.setMsgTime(dto.getMsgTime());
        return po;
    }
}
