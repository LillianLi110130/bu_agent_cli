package com.buagent.gateway.worker;

import com.buagent.gateway.app.dto.CompleteRequest;
import com.buagent.gateway.app.dto.DebugInboundRequest;
import com.buagent.gateway.app.dto.PollRequest;
import com.buagent.gateway.app.dto.PollResponse;
import com.buagent.gateway.app.dto.RenewRequest;
import com.buagent.gateway.app.dto.SimpleOkResponse;
import com.buagent.gateway.app.dto.SendTextRequest;
import com.buagent.gateway.app.dto.UploadAttachmentRequest;
import com.buagent.gateway.channel.ChannelRouter;
import com.buagent.gateway.session.SessionMailbox;
import com.buagent.gateway.session.SessionRegistry;
import com.buagent.gateway.store.entity.InboundMessageEntity;
import com.buagent.gateway.store.entity.OutboundMessageEntity;
import com.buagent.gateway.store.entity.SessionStateEntity;
import com.buagent.gateway.store.mapper.InboundMessageMapper;
import com.buagent.gateway.store.mapper.OutboundMessageMapper;
import com.buagent.gateway.store.mapper.SessionStateMapper;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.web.context.request.async.DeferredResult;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class WorkerGatewayServiceTest {

    @Mock
    private SessionStateMapper sessionStateMapper;

    @Mock
    private InboundMessageMapper inboundMessageMapper;

    @Mock
    private OutboundMessageMapper outboundMessageMapper;

    @Mock
    private ChannelRouter channelRouter;

    private SessionRegistry sessionRegistry;
    private WorkerGatewayService workerGatewayService;

    @BeforeEach
    void setUp() {
        sessionRegistry = new SessionRegistry();
        workerGatewayService = new WorkerGatewayService(
            sessionRegistry,
            sessionStateMapper,
            inboundMessageMapper,
            outboundMessageMapper,
            channelRouter,
            120L,
            90L,
            25000L
        );
    }

    @Test
    void shouldPollOneMessageImmediatelyWhenQueueAlreadyHasData() {
        String sessionKey = "telegram:123";
        when(sessionStateMapper.findBySessionKey(sessionKey)).thenReturn(null);
        doAnswer(invocation -> {
            InboundMessageEntity entity = invocation.getArgument(0);
            entity.setId(101L);
            return 1;
        }).when(inboundMessageMapper).insert(any(InboundMessageEntity.class));
        when(inboundMessageMapper.markDelivering(eq(101L), anyString(), anyLong())).thenReturn(1);

        workerGatewayService.acceptDebugInbound(new DebugInboundRequest(sessionKey, "hello"));

        DeferredResult<PollResponse> deferredResult = workerGatewayService.poll(
            new PollRequest(sessionKey, "worker-1")
        );

        PollResponse pollResponse = (PollResponse) deferredResult.getResult();
        assertNotNull(pollResponse);
        assertEquals(1, pollResponse.getMessages().size());
        assertEquals("hello", pollResponse.getMessages().get(0).getContent());
        assertEquals(1L, pollResponse.getMessages().get(0).getEpoch().longValue());
    }

    @Test
    void shouldWakeWaitingPollWhenInboundMessageArrives() {
        String sessionKey = "telegram:123";
        when(sessionStateMapper.findBySessionKey(sessionKey)).thenReturn(null);
        doAnswer(invocation -> {
            InboundMessageEntity entity = invocation.getArgument(0);
            entity.setId(101L);
            return 1;
        }).when(inboundMessageMapper).insert(any(InboundMessageEntity.class));
        when(inboundMessageMapper.markDelivering(eq(101L), anyString(), anyLong())).thenReturn(1);

        DeferredResult<PollResponse> deferredResult = workerGatewayService.poll(
            new PollRequest(sessionKey, "worker-1")
        );
        assertNull(deferredResult.getResult());

        workerGatewayService.acceptDebugInbound(new DebugInboundRequest(sessionKey, "hello"));

        PollResponse pollResponse = (PollResponse) deferredResult.getResult();
        assertNotNull(pollResponse);
        assertEquals(1, pollResponse.getMessages().size());
        assertEquals("hello", pollResponse.getMessages().get(0).getContent());
    }

    @Test
    void shouldRenewAndCompleteCurrentInFlightDelivery() {
        String sessionKey = "telegram:123";
        when(sessionStateMapper.findBySessionKey(sessionKey)).thenReturn(null);
        doAnswer(invocation -> {
            InboundMessageEntity entity = invocation.getArgument(0);
            entity.setId(101L);
            return 1;
        }).when(inboundMessageMapper).insert(any(InboundMessageEntity.class));
        when(inboundMessageMapper.markDelivering(eq(101L), anyString(), anyLong())).thenReturn(1);
        when(inboundMessageMapper.updateLease(eq(101L), anyLong())).thenReturn(1);
        when(inboundMessageMapper.markConsumed(eq(101L), anyString())).thenReturn(1);
        doAnswer(invocation -> {
            OutboundMessageEntity entity = invocation.getArgument(0);
            entity.setId(201L);
            return 1;
        }).when(outboundMessageMapper).insert(any(OutboundMessageEntity.class));
        when(outboundMessageMapper.updateStatus(eq(201L), eq("SENT"))).thenReturn(1);
        when(channelRouter.send(sessionKey, "done")).thenReturn(true);

        workerGatewayService.acceptDebugInbound(new DebugInboundRequest(sessionKey, "hello"));
        DeferredResult<PollResponse> deferredResult = workerGatewayService.poll(
            new PollRequest(sessionKey, "worker-1")
        );
        PollResponse pollResponse = (PollResponse) deferredResult.getResult();
        String deliveryId = pollResponse.getMessages().get(0).getDeliveryId();

        SimpleOkResponse renewResponse = workerGatewayService.renew(
            new RenewRequest(sessionKey, "worker-1", deliveryId)
        );
        SimpleOkResponse completeResponse = workerGatewayService.complete(
            new CompleteRequest(sessionKey, "worker-1", deliveryId, "done")
        );

        assertTrue(renewResponse.getOk());
        assertTrue(completeResponse.getOk());
        verify(channelRouter).send(sessionKey, "done");
        SessionMailbox mailbox = sessionRegistry.getOrCreate(sessionKey, 1L);
        assertNull(mailbox.getInFlightDelivery());
    }

    @Test
    void shouldPollAndCompleteUsingWorkerMinimalProtocol() {
        String workerId = "worker-1";
        when(sessionStateMapper.findBySessionKey(workerId)).thenReturn(null);
        doAnswer(invocation -> {
            InboundMessageEntity entity = invocation.getArgument(0);
            entity.setId(101L);
            return 1;
        }).when(inboundMessageMapper).insert(any(InboundMessageEntity.class));
        when(inboundMessageMapper.markDelivering(eq(101L), anyString(), anyLong())).thenReturn(1);
        when(inboundMessageMapper.markConsumed(eq(101L), anyString())).thenReturn(1);
        doAnswer(invocation -> {
            OutboundMessageEntity entity = invocation.getArgument(0);
            entity.setId(201L);
            return 1;
        }).when(outboundMessageMapper).insert(any(OutboundMessageEntity.class));
        when(outboundMessageMapper.updateStatus(eq(201L), eq("SENT"))).thenReturn(1);
        when(channelRouter.send(workerId, "done")).thenReturn(true);

        workerGatewayService.acceptDebugInbound(new DebugInboundRequest(workerId, "hello"));

        DeferredResult<PollResponse> deferredResult = workerGatewayService.poll(
            new PollRequest(null, workerId)
        );
        PollResponse pollResponse = (PollResponse) deferredResult.getResult();

        assertNotNull(pollResponse);
        assertEquals(1, pollResponse.getMessages().size());
        assertEquals("hello", pollResponse.getMessages().get(0).getContent());

        SimpleOkResponse completeResponse = workerGatewayService.complete(
            new CompleteRequest(null, workerId, null, "done")
        );

        assertTrue(completeResponse.getOk());
        verify(channelRouter).send(workerId, "done");
    }

    @Test
    void shouldAcknowledgeWorkerOnlineAndOfflineCalls() {
        assertTrue(workerGatewayService.online("worker-1").getOk());
        assertTrue(workerGatewayService.offline("worker-1").getOk());
    }

    @Test
    void shouldSendProactiveTextUsingDedicatedEndpoint() {
        when(channelRouter.send("telegram:123", "hello proactive")).thenReturn(true);

        SimpleOkResponse response = workerGatewayService.sendText(
            new SendTextRequest("telegram:123", "worker-1", "hello proactive")
        );

        assertTrue(response.getOk());
        verify(channelRouter).send("telegram:123", "hello proactive");
    }

    @Test
    void shouldUploadAttachmentUsingDedicatedEndpoint() {
        when(channelRouter.sendAttachment(
            "telegram:123",
            "report.pdf",
            "application/pdf",
            123L,
            "cGRm"
        )).thenReturn(true);

        SimpleOkResponse response = workerGatewayService.uploadAttachment(
            new UploadAttachmentRequest(
                "telegram:123",
                "worker-1",
                "report.pdf",
                "application/pdf",
                123L,
                "cGRm"
            )
        );

        assertTrue(response.getOk());
        verify(channelRouter).sendAttachment(
            "telegram:123",
            "report.pdf",
            "application/pdf",
            123L,
            "cGRm"
        );
    }

    @Test
    void shouldIncrementEpochAndClearMailboxWhenReceivingNewCommand() {
        String sessionKey = "telegram:123";
        SessionStateEntity sessionStateEntity = new SessionStateEntity();
        sessionStateEntity.setSessionKey(sessionKey);
        sessionStateEntity.setCurrentEpoch(1L);

        when(sessionStateMapper.findBySessionKey(sessionKey)).thenReturn(sessionStateEntity);
        when(sessionStateMapper.updateCurrentEpoch(sessionKey, 2L)).thenReturn(1);
        doAnswer(invocation -> {
            InboundMessageEntity entity = invocation.getArgument(0);
            entity.setId(101L);
            return 1;
        }).when(inboundMessageMapper).insert(any(InboundMessageEntity.class));

        workerGatewayService.acceptDebugInbound(new DebugInboundRequest(sessionKey, "hello"));
        workerGatewayService.acceptDebugInbound(new DebugInboundRequest(sessionKey, "/new"));

        SessionMailbox mailbox = sessionRegistry.getOrCreate(sessionKey, 2L);
        assertEquals(2L, mailbox.getCurrentEpoch());
        assertEquals(0, mailbox.getPendingSize());
        assertNull(mailbox.getInFlightDelivery());
    }
}
