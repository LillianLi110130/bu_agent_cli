package com.buagent.gateway.session;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class SessionMailboxTest {

    @Test
    void shouldEnqueueAndPollInFifoOrder() {
        SessionMailbox mailbox = new SessionMailbox("telegram:123", 1L);
        mailbox.enqueue(new InboundMessageSnapshot(1L, "telegram:123", 1L, "first"));
        mailbox.enqueue(new InboundMessageSnapshot(2L, "telegram:123", 1L, "second"));

        InboundMessageSnapshot first = mailbox.pollNextSnapshot();
        InboundMessageSnapshot second = mailbox.pollNextSnapshot();

        assertNotNull(first);
        assertNotNull(second);
        assertEquals(1L, first.getMessageId());
        assertEquals(2L, second.getMessageId());
        assertEquals(0, mailbox.getPendingSize());
    }

    @Test
    void shouldClearPendingQueueAndInFlightWhenResettingEpoch() {
        SessionMailbox mailbox = new SessionMailbox("telegram:123", 1L);
        mailbox.enqueue(new InboundMessageSnapshot(1L, "telegram:123", 1L, "first"));
        mailbox.setInFlightDelivery(
            new InFlightDelivery(1L, "d-1", "worker-1", 1L, "first", System.currentTimeMillis() + 1000)
        );

        mailbox.resetForNewEpoch(2L);

        assertEquals(2L, mailbox.getCurrentEpoch());
        assertEquals(0, mailbox.getPendingSize());
        assertNull(mailbox.getInFlightDelivery());
    }
}
