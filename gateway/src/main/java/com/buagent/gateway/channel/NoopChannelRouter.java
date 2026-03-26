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
}
