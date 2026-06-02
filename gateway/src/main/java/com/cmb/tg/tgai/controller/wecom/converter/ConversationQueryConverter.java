package com.cmb.tg.tgai.controller.wecom.converter;

import com.cmb.tg.tgai.controller.wecom.vo.request.ChatConversationReplayRequest;
import com.cmb.tg.tgai.controller.wecom.vo.request.ChatGroupSearchRequest;
import com.cmb.tg.tgai.controller.wecom.vo.response.ChatConversationReplayResponseVO;
import com.cmb.tg.tgai.controller.wecom.vo.response.ChatGroupSearchResponseVO;
import com.cmb.tg.tgai.service.wecom.dto.ChatConversationReplayQuery;
import com.cmb.tg.tgai.service.wecom.dto.ChatConversationReplayResult;
import com.cmb.tg.tgai.service.wecom.dto.ChatGroupSearchQuery;
import com.cmb.tg.tgai.service.wecom.dto.ChatGroupSearchResult;
import org.springframework.stereotype.Component;

import java.time.LocalTime;
import java.util.Collections;
import java.util.stream.Collectors;

@Component
public class ConversationQueryConverter {

    public ChatGroupSearchQuery toQuery(final ChatGroupSearchRequest request) {
        return ChatGroupSearchQuery.builder()
                .groupName(request.getGroupName())
                .groupOwnerName(request.getGroupOwnerName())
                .build();
    }

    public ChatConversationReplayQuery toQuery(final ChatConversationReplayRequest request) {
        return ChatConversationReplayQuery.builder()
                .roomId(request.getRoomId())
                .startTime(request.getStartDate().atStartOfDay())
                .endTime(request.getEndDate().atTime(LocalTime.of(23, 59, 59)))
                .build();
    }

    public ChatGroupSearchResponseVO toResponse(final ChatGroupSearchResult result) {
        return ChatGroupSearchResponseVO.builder()
                .total(result.getTotal())
                .list(result.getList() == null ? Collections.emptyList() : result.getList().stream()
                        .map(item -> ChatGroupSearchResponseVO.GroupItemVO.builder()
                                .roomId(item.getRoomId())
                                .groupName(item.getGroupName())
                                .groupOwnerName(item.getGroupOwnerName())
                                .lastMsgTime(item.getLastMsgTime())
                                .build())
                        .collect(Collectors.toList()))
                .build();
    }

    public ChatConversationReplayResponseVO toResponse(final ChatConversationReplayResult result) {
        return ChatConversationReplayResponseVO.builder()
                .roomId(result.getRoomId())
                .groupName(result.getGroupName())
                .total(result.getTotal())
                .messages(result.getMessages() == null ? Collections.emptyList() : result.getMessages().stream()
                        .map(item -> ChatConversationReplayResponseVO.MessageItemVO.builder()
                                .msgId(item.getMsgId())
                                .msgType(item.getMsgType())
                                .msgTime(item.getMsgTime())
                                .fromUser(item.getFromUser())
                                .fromUserName(item.getFromUserName())
                                .fromUserType(item.getFromUserType())
                                .msgBody(item.getMsgBody())
                                .displayText(item.getDisplayText())
                                .build())
                        .collect(Collectors.toList()))
                .transcript(result.getTranscript())
                .build();
    }
}
