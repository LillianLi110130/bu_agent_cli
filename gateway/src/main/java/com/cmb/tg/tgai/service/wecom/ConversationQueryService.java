package com.cmb.tg.tgai.service.wecom;

import com.cmb.tg.tgai.infrastructure.wecom.mapper.ChatGroupMapper;
import com.cmb.tg.tgai.infrastructure.wecom.mapper.ChatMessageMapper;
import com.cmb.tg.tgai.infrastructure.wecom.po.ChatConversationMessagePO;
import com.cmb.tg.tgai.infrastructure.wecom.po.ChatGroupPO;
import com.cmb.tg.tgai.infrastructure.wecom.po.ChatGroupSearchPO;
import com.cmb.tg.tgai.service.wecom.dto.ChatConversationReplayQuery;
import com.cmb.tg.tgai.service.wecom.dto.ChatConversationReplayResult;
import com.cmb.tg.tgai.service.wecom.dto.ChatGroupSearchQuery;
import com.cmb.tg.tgai.service.wecom.dto.ChatGroupSearchResult;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;

import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class ConversationQueryService {

    private static final DateTimeFormatter DATE_TIME_FORMATTER = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

    private final ObjectMapper objectMapper;

    private final ChatGroupMapper chatGroupMapper;

    private final ChatMessageMapper chatMessageMapper;

    public ChatGroupSearchResult searchGroups(final ChatGroupSearchQuery query) {
        List<ChatGroupSearchPO> groupList = chatGroupMapper.searchGroups(query.getGroupName(), query.getGroupOwnerName());
        long total = groupList.size();

        return ChatGroupSearchResult.builder()
                .total(total)
                .list(groupList.stream()
                        .map(item -> ChatGroupSearchResult.GroupItem.builder()
                                .roomId(item.getRoomId())
                                .groupName(item.getGroupName())
                                .groupOwnerName(item.getGroupOwnerName())
                                .lastMsgTime(item.getLastMsgTime())
                                .build())
                        .collect(Collectors.toList()))
                .build();
    }

    public ChatConversationReplayResult replayConversation(final ChatConversationReplayQuery query) {
        ChatGroupPO groupPO = chatGroupMapper.selectByRoomId(query.getRoomId());
        List<ChatConversationMessagePO> messageList =
                chatMessageMapper.selectReplayMessages(query.getRoomId(), query.getStartTime(), query.getEndTime());
        long total = messageList.size();
        String groupName = groupPO == null ? null : groupPO.getGroupName();
        List<ChatConversationReplayResult.MessageItem> messageItemList = messageList.stream()
                .map(item -> ChatConversationReplayResult.MessageItem.builder()
                        .msgId(item.getMsgId())
                        .msgType(item.getMsgType())
                        .msgTime(item.getMsgTime())
                        .fromUser(item.getFromUser())
                        .fromUserName(item.getFromUserName())
                        .fromUserType(item.getFromUserType())
                        .msgBody(item.getMsgBody())
                        .displayText(buildDisplayText(item.getMsgType(), item.getMsgBody()))
                        .build())
                .collect(Collectors.toList());

        return ChatConversationReplayResult.builder()
                .roomId(query.getRoomId())
                .groupName(groupName)
                .total(total)
                .messages(messageItemList)
                .transcript(buildTranscript(groupName, query.getRoomId(), messageItemList))
                .build();
    }

    private String buildTranscript(
            final String groupName,
            final String roomId,
            final List<ChatConversationReplayResult.MessageItem> messageList) {
        StringBuilder builder = new StringBuilder();
        if (StringUtils.hasText(groupName)) {
            builder.append("群名称：").append(groupName).append("\n\n");
        } else if (StringUtils.hasText(roomId)) {
            builder.append("群ID：").append(roomId).append("\n\n");
        }
        for (ChatConversationReplayResult.MessageItem messageVO : messageList) {
            if (messageVO.getMsgTime() != null) {
                builder.append("[")
                        .append(messageVO.getMsgTime().format(DATE_TIME_FORMATTER))
                        .append("]");
            }
            builder.append("[")
                    .append("EMPLOYEE".equals(messageVO.getFromUserType()) ? "员工" : "客户")
                    .append("]");
            builder.append("[")
                    .append(StringUtils.hasText(messageVO.getFromUserName()) ? messageVO.getFromUserName() : messageVO.getFromUser())
                    .append("]")
                    .append("\n");
            builder.append(messageVO.getDisplayText())
                    .append("\n\n");
        }
        return builder.toString().trim();
    }

    private String buildDisplayText(final String msgType, final String msgBody) {
        if ("text".equalsIgnoreCase(msgType)) {
            String textContent = extractTextContent(msgBody);
            if (StringUtils.hasText(textContent)) {
                return textContent;
            }
        }
        if (StringUtils.hasText(msgType)) {
            return "[" + msgType + "消息]";
        }
        return "[其余类型消息]";
    }

    private String extractTextContent(final String msgBody) {
        if (!StringUtils.hasText(msgBody)) {
            return null;
        }
        try {
            JsonNode jsonNode = objectMapper.readTree(msgBody);
            JsonNode contentNode = jsonNode.get("content");
            return contentNode == null || contentNode.isNull() ? null : contentNode.asText();
        } catch (Exception ex) {
            return msgBody;
        }
    }
}
