package com.buagent.gateway.worker;

import com.buagent.gateway.app.dto.CompleteRequest;
import com.buagent.gateway.app.dto.DebugInboundRequest;
import com.buagent.gateway.app.dto.PollMessageDto;
import com.buagent.gateway.app.dto.PollRequest;
import com.buagent.gateway.app.dto.PollResponse;
import com.buagent.gateway.app.dto.RenewRequest;
import com.buagent.gateway.app.dto.SendTextRequest;
import com.buagent.gateway.app.dto.SimpleOkResponse;
import com.buagent.gateway.app.dto.UploadAttachmentRequest;
import com.buagent.gateway.channel.ChannelRouter;
import com.buagent.gateway.session.InFlightDelivery;
import com.buagent.gateway.session.InboundMessageSnapshot;
import com.buagent.gateway.session.PollWaiter;
import com.buagent.gateway.session.SessionMailbox;
import com.buagent.gateway.session.SessionRegistry;
import com.buagent.gateway.store.entity.InboundMessageEntity;
import com.buagent.gateway.store.entity.OutboundMessageEntity;
import com.buagent.gateway.store.entity.SessionStateEntity;
import com.buagent.gateway.store.mapper.InboundMessageMapper;
import com.buagent.gateway.store.mapper.OutboundMessageMapper;
import com.buagent.gateway.store.mapper.SessionStateMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.web.context.request.async.DeferredResult;
import org.springframework.web.server.ResponseStatusException;

import java.util.Collections;
import java.util.UUID;

@Service
public class WorkerGatewayService {

    private static final Logger logger = LoggerFactory.getLogger(WorkerGatewayService.class);

    private final SessionRegistry sessionRegistry;
    private final SessionStateMapper sessionStateMapper;
    private final InboundMessageMapper inboundMessageMapper;
    private final OutboundMessageMapper outboundMessageMapper;
    private final ChannelRouter channelRouter;
    private final long deliveryLeaseTimeoutMillis;
    private final long ownerActiveTimeoutMillis;
    private final long pollWaitTimeoutMillis;

    public WorkerGatewayService(
        SessionRegistry sessionRegistry,
        SessionStateMapper sessionStateMapper,
        InboundMessageMapper inboundMessageMapper,
        OutboundMessageMapper outboundMessageMapper,
        ChannelRouter channelRouter,
        @Value("${gateway.delivery-lease-timeout-seconds:120}") long deliveryLeaseTimeoutSeconds,
        @Value("${gateway.owner-active-timeout-seconds:90}") long ownerActiveTimeoutSeconds,
        @Value("${gateway.poll-wait-timeout-ms:25000}") long pollWaitTimeoutMillis
    ) {
        this.sessionRegistry = sessionRegistry;
        this.sessionStateMapper = sessionStateMapper;
        this.inboundMessageMapper = inboundMessageMapper;
        this.outboundMessageMapper = outboundMessageMapper;
        this.channelRouter = channelRouter;
        this.deliveryLeaseTimeoutMillis = deliveryLeaseTimeoutSeconds * 1000L;
        this.ownerActiveTimeoutMillis = ownerActiveTimeoutSeconds * 1000L;
        this.pollWaitTimeoutMillis = pollWaitTimeoutMillis;
    }

    public SimpleOkResponse acceptDebugInbound(DebugInboundRequest request) {
        if ("/new".equals(request.getContent())) {
            handleNewSession(request.getSessionKey());
            return new SimpleOkResponse(true);
        }

        long epoch = ensureCurrentEpoch(request.getSessionKey());
        SessionMailbox mailbox = sessionRegistry.getOrCreate(request.getSessionKey(), epoch);

        InboundMessageEntity inboundMessageEntity = new InboundMessageEntity();
        inboundMessageEntity.setSessionKey(request.getSessionKey());
        inboundMessageEntity.setEpoch(epoch);
        inboundMessageEntity.setContent(request.getContent());
        inboundMessageEntity.setStatus("RECEIVED");
        inboundMessageEntity.setCreatedAt(System.currentTimeMillis());
        inboundMessageMapper.insert(inboundMessageEntity);

        PollWaiter waiterToWake = null;
        PollResponse pollResponse = null;
        synchronized (mailbox) {
            mailbox.setCurrentEpoch(epoch);
            mailbox.enqueue(new InboundMessageSnapshot(
                inboundMessageEntity.getId(),
                request.getSessionKey(),
                epoch,
                request.getContent()
            ));
            if (mailbox.getActivePollWaiter() != null) {
                waiterToWake = mailbox.getActivePollWaiter();
                mailbox.setActivePollWaiter(null);
                pollResponse = dispatchNextMessageLocked(mailbox, waiterToWake.getWorkerId());
            }
        }

        if (waiterToWake != null && pollResponse != null) {
            waiterToWake.getDeferredResult().setResult(pollResponse);
        }
        return new SimpleOkResponse(true);
    }

    public DeferredResult<PollResponse> poll(PollRequest request) {
        String sessionKey = resolveSessionKey(request.getSessionKey(), request.getWorkerId());
        long currentEpoch = ensureCurrentEpoch(sessionKey);
        SessionMailbox mailbox = sessionRegistry.getOrCreate(sessionKey, currentEpoch);
        DeferredResult<PollResponse> deferredResult = new DeferredResult<>(pollWaitTimeoutMillis);

        deferredResult.onTimeout(() -> {
            synchronized (mailbox) {
                PollWaiter currentWaiter = mailbox.getActivePollWaiter();
                if (currentWaiter != null && currentWaiter.getDeferredResult() == deferredResult) {
                    mailbox.setActivePollWaiter(null);
                }
            }
            deferredResult.setResult(PollResponse.empty());
        });

        PollResponse immediateResponse = null;
        synchronized (mailbox) {
            mailbox.setCurrentEpoch(currentEpoch);
            long now = System.currentTimeMillis();
            if (!canUseOwner(mailbox, request.getWorkerId(), now)) {
                throw new ResponseStatusException(HttpStatus.CONFLICT, "session owner conflict");
            }
            if (mailbox.getActivePollWaiter() != null) {
                throw new ResponseStatusException(HttpStatus.CONFLICT, "active poll already exists");
            }

            touchOwner(mailbox, request.getWorkerId(), now);
            if (mailbox.getPendingSize() > 0) {
                immediateResponse = dispatchNextMessageLocked(mailbox, request.getWorkerId());
            } else {
                mailbox.setActivePollWaiter(new PollWaiter(request.getWorkerId(), deferredResult));
            }
        }

        if (immediateResponse != null) {
            deferredResult.setResult(immediateResponse);
        }
        return deferredResult;
    }

    public SimpleOkResponse online(String workerId) {
        if (isBlank(workerId)) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "worker_id is required");
        }
        logger.info("Worker marked online, workerId={}", workerId);
        return new SimpleOkResponse(true);
    }

    public SimpleOkResponse offline(String workerId) {
        if (isBlank(workerId)) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "worker_id is required");
        }
        logger.info("Worker marked offline, workerId={}", workerId);
        return new SimpleOkResponse(true);
    }

    public SimpleOkResponse sendText(SendTextRequest request) {
        String sessionKey = resolveSessionKey(request.getSessionKey(), request.getWorkerId());
        boolean sent = channelRouter.send(sessionKey, request.getText());
        return new SimpleOkResponse(sent);
    }

    public SimpleOkResponse uploadAttachment(UploadAttachmentRequest request) {
        String sessionKey = resolveSessionKey(request.getSessionKey(), request.getWorkerId());
        boolean sent = channelRouter.sendAttachment(
            sessionKey,
            request.getFileName(),
            request.getMimeType(),
            request.getFileSize(),
            request.getFileContentBase64()
        );
        return new SimpleOkResponse(sent);
    }

    public SimpleOkResponse renew(RenewRequest request) {
        String sessionKey = resolveSessionKey(request.getSessionKey(), request.getWorkerId());
        long currentEpoch = ensureCurrentEpoch(sessionKey);
        SessionMailbox mailbox = sessionRegistry.getOrCreate(sessionKey, currentEpoch);
        synchronized (mailbox) {
            InFlightDelivery inFlightDelivery = mailbox.getInFlightDelivery();
            if (inFlightDelivery == null) {
                return new SimpleOkResponse(false);
            }
            if (!request.getWorkerId().equals(mailbox.getOwnerWorkerId())) {
                return new SimpleOkResponse(false);
            }
            String deliveryId = resolveDeliveryId(request.getDeliveryId(), inFlightDelivery);
            if (deliveryId == null) {
                return new SimpleOkResponse(false);
            }
            if (!deliveryId.equals(inFlightDelivery.getDeliveryId())) {
                return new SimpleOkResponse(false);
            }

            long leaseExpiresAt = System.currentTimeMillis() + deliveryLeaseTimeoutMillis;
            int updatedRows = inboundMessageMapper.updateLease(inFlightDelivery.getMessageId(), leaseExpiresAt);
            if (updatedRows != 1) {
                return new SimpleOkResponse(false);
            }
            inFlightDelivery.setLeaseExpiresAt(leaseExpiresAt);
            touchOwner(mailbox, request.getWorkerId(), System.currentTimeMillis());
            return new SimpleOkResponse(true);
        }
    }

    public SimpleOkResponse complete(CompleteRequest request) {
        String sessionKey = resolveSessionKey(request.getSessionKey(), request.getWorkerId());
        long currentEpoch = ensureCurrentEpoch(sessionKey);
        SessionMailbox mailbox = sessionRegistry.getOrCreate(sessionKey, currentEpoch);
        synchronized (mailbox) {
            InFlightDelivery inFlightDelivery = mailbox.getInFlightDelivery();
            if (inFlightDelivery == null) {
                return new SimpleOkResponse(false);
            }
            if (!request.getWorkerId().equals(mailbox.getOwnerWorkerId())) {
                return new SimpleOkResponse(false);
            }
            String deliveryId = resolveDeliveryId(request.getDeliveryId(), inFlightDelivery);
            if (deliveryId == null) {
                return new SimpleOkResponse(false);
            }
            if (!deliveryId.equals(inFlightDelivery.getDeliveryId())) {
                return new SimpleOkResponse(false);
            }

            int consumedRows = inboundMessageMapper.markConsumed(
                inFlightDelivery.getMessageId(),
                inFlightDelivery.getDeliveryId()
            );
            if (consumedRows != 1) {
                return new SimpleOkResponse(false);
            }

            OutboundMessageEntity outboundMessageEntity = new OutboundMessageEntity();
            outboundMessageEntity.setSessionKey(sessionKey);
            outboundMessageEntity.setEpoch(inFlightDelivery.getEpoch());
            outboundMessageEntity.setInboundMessageId(inFlightDelivery.getMessageId());
            outboundMessageEntity.setContent(request.getFinalContent());
            outboundMessageEntity.setStatus("PENDING");
            outboundMessageEntity.setCreatedAt(System.currentTimeMillis());
            outboundMessageMapper.insert(outboundMessageEntity);

            boolean sent = channelRouter.send(sessionKey, request.getFinalContent());
            outboundMessageMapper.updateStatus(outboundMessageEntity.getId(), sent ? "SENT" : "FAILED");
            mailbox.setInFlightDelivery(null);
            return new SimpleOkResponse(true);
        }
    }

    private void handleNewSession(String sessionKey) {
        long currentEpoch = ensureCurrentEpoch(sessionKey);
        long newEpoch = currentEpoch + 1L;
        sessionStateMapper.updateCurrentEpoch(sessionKey, newEpoch);
        SessionMailbox mailbox = sessionRegistry.getOrCreate(sessionKey, newEpoch);
        synchronized (mailbox) {
            mailbox.resetForNewEpoch(newEpoch);
        }
    }

    private long ensureCurrentEpoch(String sessionKey) {
        SessionStateEntity sessionStateEntity = sessionStateMapper.findBySessionKey(sessionKey);
        if (sessionStateEntity != null) {
            return sessionStateEntity.getCurrentEpoch();
        }

        SessionStateEntity initialState = new SessionStateEntity(sessionKey, 1L);
        sessionStateMapper.insert(initialState);
        return 1L;
    }

    private boolean canUseOwner(SessionMailbox mailbox, String workerId, long now) {
        if (mailbox.getOwnerWorkerId() == null || mailbox.getOwnerActiveUntil() == null) {
            return true;
        }
        if (workerId.equals(mailbox.getOwnerWorkerId())) {
            return true;
        }
        return mailbox.getOwnerActiveUntil() < now;
    }

    private void touchOwner(SessionMailbox mailbox, String workerId, long now) {
        mailbox.setOwnerWorkerId(workerId);
        mailbox.setOwnerActiveUntil(now + ownerActiveTimeoutMillis);
    }

    private PollResponse dispatchNextMessageLocked(SessionMailbox mailbox, String workerId) {
        InboundMessageSnapshot snapshot = mailbox.pollNextSnapshot();
        if (snapshot == null) {
            return PollResponse.empty();
        }

        String deliveryId = UUID.randomUUID().toString();
        long leaseExpiresAt = System.currentTimeMillis() + deliveryLeaseTimeoutMillis;
        int updatedRows = inboundMessageMapper.markDelivering(snapshot.getMessageId(), deliveryId, leaseExpiresAt);
        if (updatedRows != 1) {
            logger.warn("Failed to mark inbound message as DELIVERING, messageId={}", snapshot.getMessageId());
            return PollResponse.empty();
        }

        mailbox.setInFlightDelivery(new InFlightDelivery(
            snapshot.getMessageId(),
            deliveryId,
            workerId,
            snapshot.getEpoch(),
            snapshot.getContent(),
            leaseExpiresAt
        ));
        touchOwner(mailbox, workerId, System.currentTimeMillis());

        return new PollResponse(Collections.singletonList(
            new PollMessageDto(
                snapshot.getMessageId(),
                deliveryId,
                snapshot.getEpoch(),
                snapshot.getContent()
            )
        ));
    }

    private String resolveSessionKey(String sessionKey, String workerId) {
        if (!isBlank(sessionKey)) {
            return sessionKey;
        }
        if (!isBlank(workerId)) {
            return workerId;
        }
        throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "session_key or worker_id is required");
    }

    private String resolveDeliveryId(String deliveryId, InFlightDelivery inFlightDelivery) {
        if (!isBlank(deliveryId)) {
            return deliveryId;
        }
        if (inFlightDelivery == null || isBlank(inFlightDelivery.getDeliveryId())) {
            return null;
        }
        return inFlightDelivery.getDeliveryId();
    }

    private boolean isBlank(String value) {
        return value == null || value.trim().isEmpty();
    }
}
