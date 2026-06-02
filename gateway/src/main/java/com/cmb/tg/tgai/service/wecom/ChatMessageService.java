package com.cmb.tg.tgai.service.wecom;

import com.cmb.tg.tgai.infrastructure.wecom.mapper.ChatMessageMapper;
import com.cmb.tg.tgai.infrastructure.wecom.po.ChatMessagePO;
import com.cmb.tg.tgai.service.wecom.converter.ChatMessageServiceConverter;
import com.cmb.tg.tgai.service.wecom.dto.ChatMessageDTO;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.stream.Collectors;

@Slf4j
@Service
@RequiredArgsConstructor
public class ChatMessageService {

    private final ChatMessageMapper chatMessageMapper;

    private final ChatMessageServiceConverter chatMessageServiceConverter;

    public void saveMessages(final List<ChatMessageDTO> dtoList) {
        List<ChatMessagePO> poList = dtoList.stream()
                .map(chatMessageServiceConverter::toPo)
                .collect(Collectors.toList());
        int savedCount = chatMessageMapper.batchInsertIgnore(poList);
        log.info("save wecom chat messages success, requestCount={}, savedCount={}", dtoList.size(), savedCount);
    }
}
