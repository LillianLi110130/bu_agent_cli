package com.cmb.tg.tgai.controller.wecom;

import com.cmb.tg.tgai.controller.wecom.converter.WecomMessageReceiveConverter;
import com.cmb.tg.tgai.controller.wecom.vo.request.WecomMessageReceiveRequest;
import com.cmb.tg.tgai.controller.wecom.vo.response.WecomMessageReceiveResponseVO;
import com.cmb.tg.tgai.service.wecom.ChatMessageService;
import com.cmb.tg.tgai.service.wecom.dto.ChatMessageDTO;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.util.CollectionUtils;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import javax.validation.Valid;
import java.util.List;
import java.util.stream.Collectors;

@Validated
@RestController
@Slf4j
@RequiredArgsConstructor
@RequestMapping("/wecom")
public class WecomMessageReceiveController {

    private final WecomMessageReceiveConverter wecomMessageReceiveConverter;

    private final ChatMessageService chatMessageService;

    @PostMapping("/message")
    public WecomMessageReceiveResponseVO receiveMessages(@RequestBody final List<@Valid WecomMessageReceiveRequest> requestList) {
        if (CollectionUtils.isEmpty(requestList)) {
            log.info("receive wecom messages empty");
            return WecomMessageReceiveResponseVO.builder()
                    .code("SUCCESS")
                    .message("处理成功")
                    .build();
        }
        log.info("receive wecom messages, count={}", requestList.size());
        List<ChatMessageDTO> dtoList = requestList.stream()
                .map(wecomMessageReceiveConverter::toDto)
                .collect(Collectors.toList());
        chatMessageService.saveMessages(dtoList);
        return WecomMessageReceiveResponseVO.builder()
                .code("SUCCESS")
                .message("处理成功")
                .build();
    }
}
