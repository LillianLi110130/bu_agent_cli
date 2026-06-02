package com.cmb.tg.tgai.controller.wecom;

import com.cmb.tg.tgai.controller.wecom.converter.ConversationQueryConverter;
import com.cmb.tg.tgai.controller.wecom.vo.request.ChatConversationReplayRequest;
import com.cmb.tg.tgai.controller.wecom.vo.request.ChatGroupSearchRequest;
import com.cmb.tg.tgai.controller.wecom.vo.response.ChatConversationReplayResponseVO;
import com.cmb.tg.tgai.controller.wecom.vo.response.ChatGroupSearchResponseVO;
import com.cmb.tg.tgai.service.wecom.ConversationQueryService;
import com.cmb.tg.tgai.service.wecom.dto.ChatConversationReplayResult;
import com.cmb.tg.tgai.service.wecom.dto.ChatGroupSearchResult;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import javax.validation.Valid;

@Slf4j
@Validated
@RestController
@RequiredArgsConstructor
@RequestMapping("/wecom/chat")
public class ConversationQueryController {

    private final ConversationQueryService conversationQueryService;

    private final ConversationQueryConverter conversationQueryConverter;

    @PostMapping("/group/search")
    public ChatGroupSearchResponseVO searchGroups(@RequestBody @Valid final ChatGroupSearchRequest request) {
        log.info("search wecom groups, groupName={}, groupOwnerName={}",
                request.getGroupName(), request.getGroupOwnerName());
        ChatGroupSearchResult result = conversationQueryService.searchGroups(
                conversationQueryConverter.toQuery(request));
        return conversationQueryConverter.toResponse(result);
    }

    @PostMapping("/conversation/replay")
    public ChatConversationReplayResponseVO replayConversation(
            @RequestBody @Valid final ChatConversationReplayRequest request) {
        log.info("query wecom group messages, roomId={}, startDate={}, endDate={}",
                request.getRoomId(), request.getStartDate(), request.getEndDate());
        ChatConversationReplayResult result = conversationQueryService.replayConversation(
                conversationQueryConverter.toQuery(request));
        return conversationQueryConverter.toResponse(result);
    }
}
