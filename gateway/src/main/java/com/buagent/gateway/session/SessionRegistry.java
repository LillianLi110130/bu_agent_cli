package com.buagent.gateway.session;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import org.springframework.stereotype.Component;

@Component
public class SessionRegistry {

    private final Map<String, SessionMailbox> mailboxes = new ConcurrentHashMap<>();

    public SessionMailbox getOrCreate(String sessionKey, Long currentEpoch) {
        return mailboxes.compute(sessionKey, (key, existing) -> {
            if (existing == null) {
                return new SessionMailbox(sessionKey, currentEpoch);
            }
            existing.setCurrentEpoch(currentEpoch);
            return existing;
        });
    }
}
