package com.buagent.gateway.channel;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

@Component
public class NoopChannelRouter implements ChannelRouter {

    private static final Logger logger = LoggerFactory.getLogger(NoopChannelRouter.class);

    @Override
    public boolean send(String sessionKey, String content) {
        logger.info("Noop send sessionKey={} content={}", sessionKey, content);
        return true;
    }

    @Override
    public boolean sendAttachment(
        String sessionKey,
        String fileName,
        String mimeType,
        Long fileSize,
        String fileContentBase64
    ) {
        logger.info(
            "Noop send attachment sessionKey={} fileName={} mimeType={} fileSize={} base64Length={}",
            sessionKey,
            fileName,
            mimeType,
            fileSize,
            fileContentBase64 == null ? 0 : fileContentBase64.length()
        );
        return true;
    }
}
