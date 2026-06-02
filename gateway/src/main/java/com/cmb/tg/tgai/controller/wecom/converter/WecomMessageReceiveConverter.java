package com.cmb.tg.tgai.controller.wecom.converter;

import com.cmb.tg.tgai.controller.wecom.vo.request.WecomMessageReceiveRequest;
import com.cmb.tg.tgai.service.wecom.dto.ChatMessageDTO;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;
import org.springframework.util.CollectionUtils;
import org.springframework.util.StringUtils;

import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.util.Collections;
import java.util.List;

@Component
@RequiredArgsConstructor
public class WecomMessageReceiveConverter {

    private static final TypeReference<List<String>> STRING_LIST_TYPE = new TypeReference<List<String>>() {
    };

    private final ObjectMapper objectMapper;

    public ChatMessageDTO toDto(final WecomMessageReceiveRequest request) {
        return ChatMessageDTO.builder()
                .msgId(request.getMsgId())
                .roomId(request.getRoomId())
                .fromUser(request.getFromUser())
                .wthrFromCm(request.getWthrFromCm())
                .toList(request.getToList())
                .singleReceiverId(parseSingleReceiverId(request.getRoomId(), request.getToList()))
                .msgBody(request.getMsgBody())
                .msgType(request.getMsgType())
                .acsKey(request.getAcsKey())
                .msgTimeLong(request.getMsgTimeLong())
                .msgTime(toLocalDateTime(request.getMsgTimeLong()))
                .build();
    }

    private String parseSingleReceiverId(final String roomId, final String toList) {
        if (StringUtils.hasText(roomId)) {
            return null;
        }
        List<String> receiverList = parseReceiverList(toList);
        if (CollectionUtils.isEmpty(receiverList)) {
            return null;
        }
        return receiverList.get(0);
    }

    private List<String> parseReceiverList(final String toList) {
        if (!StringUtils.hasText(toList)) {
            return Collections.emptyList();
        }
        try {
            return objectMapper.readValue(toList, STRING_LIST_TYPE);
        } catch (Exception ex) {
            return Collections.emptyList();
        }
    }

    private LocalDateTime toLocalDateTime(final Long msgTimeLong) {
        return LocalDateTime.ofInstant(Instant.ofEpochMilli(msgTimeLong), ZoneId.systemDefault());
    }
}
