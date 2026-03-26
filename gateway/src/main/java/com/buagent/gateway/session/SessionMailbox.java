package com.buagent.gateway.session;

import java.util.ArrayDeque;
import java.util.Deque;

import lombok.Getter;
import lombok.Setter;

@Getter
public class SessionMailbox {

    private final String sessionKey;
    @Setter
    private Long currentEpoch;
    private final Deque<InboundMessageSnapshot> pendingQueue = new ArrayDeque<>();
    @Setter
    private InFlightDelivery inFlightDelivery;
    @Setter
    private PollWaiter activePollWaiter;
    @Setter
    private String ownerWorkerId;
    @Setter
    private Long ownerActiveUntil;

    public SessionMailbox(String sessionKey, Long currentEpoch) {
        this.sessionKey = sessionKey;
        this.currentEpoch = currentEpoch;
    }

    public void enqueue(InboundMessageSnapshot snapshot) {
        pendingQueue.addLast(snapshot);
    }

    public InboundMessageSnapshot pollNextSnapshot() {
        return pendingQueue.pollFirst();
    }

    public int getPendingSize() {
        return pendingQueue.size();
    }

    public void resetForNewEpoch(Long newEpoch) {
        this.currentEpoch = newEpoch;
        this.pendingQueue.clear();
        this.inFlightDelivery = null;
    }
}
