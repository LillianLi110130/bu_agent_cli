package com.buagent.gateway.worker;

import com.buagent.gateway.app.dto.CompleteRequest;
import com.buagent.gateway.app.dto.MessageRequest;
import com.buagent.gateway.app.dto.SimpleOkResponse;
import com.buagent.gateway.app.dto.WorkerRequest;
import com.buagent.gateway.store.entity.InboundMessageEntity;
import com.buagent.gateway.store.entity.OnlineWorkerEntity;
import com.buagent.gateway.store.entity.OutboundMessageEntity;
import com.buagent.gateway.store.mapper.InboundMessageMapper;
import com.buagent.gateway.store.mapper.OnlineWorkerMapper;
import com.buagent.gateway.store.mapper.OutboundMessageMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.DisposableBean;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.dao.DuplicateKeyException;
import org.springframework.stereotype.Service;
import org.springframework.web.server.ResponseStatusException;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.io.IOException;
import java.util.Collections;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;

import static org.springframework.http.HttpStatus.CONFLICT;

@Service
public class WorkerGatewayService implements DisposableBean {

    private static final Logger logger = LoggerFactory.getLogger(WorkerGatewayService.class);

    private static final String STATUS_ONLINE = "online";
    private static final String STATUS_OFFLINE = "offline";
    private static final String STATUS_RECEIVED = "RECEIVED";
    private static final String STATUS_CONSUMED = "CONSUMED";
    private static final String EVENT_READY = "ready";
    private static final String EVENT_MESSAGE = "message";
    private static final String EVENT_HEARTBEAT = "heartbeat";
    private static final String EVENT_ERROR = "error";

    private final InboundMessageMapper inboundMessageMapper;
    private final OutboundMessageMapper outboundMessageMapper;
    private final OnlineWorkerMapper onlineWorkerMapper;
    private final long streamHeartbeatIntervalMillis;
    private final long streamEmitterTimeoutMillis;
    private final ScheduledExecutorService heartbeatExecutorService;
    private final ConcurrentMap<String, StreamSession> streamSessions;

    public WorkerGatewayService(
        InboundMessageMapper inboundMessageMapper,
        OutboundMessageMapper outboundMessageMapper,
        OnlineWorkerMapper onlineWorkerMapper,
        @Value("${gateway.stream-heartbeat-interval-ms:10000}") long streamHeartbeatIntervalMillis,
        @Value("${gateway.stream-emitter-timeout-ms:0}") long streamEmitterTimeoutMillis
    ) {
        this.inboundMessageMapper = inboundMessageMapper;
        this.outboundMessageMapper = outboundMessageMapper;
        this.onlineWorkerMapper = onlineWorkerMapper;
        this.streamHeartbeatIntervalMillis = streamHeartbeatIntervalMillis;
        this.streamEmitterTimeoutMillis = streamEmitterTimeoutMillis;
        this.heartbeatExecutorService = Executors.newScheduledThreadPool(1);
        this.streamSessions = new ConcurrentHashMap<String, StreamSession>();
    }

    public SimpleOkResponse online(WorkerRequest request) {
        upsertWorkerStatus(request.getWorkerId(), STATUS_ONLINE);
        logger.info("Worker marked online. workerId={}", request.getWorkerId());
        return new SimpleOkResponse(true);
    }

    public SimpleOkResponse offline(WorkerRequest request) {
        upsertWorkerStatus(request.getWorkerId(), STATUS_OFFLINE);
        removeStreamSession(request.getWorkerId(), streamSessions.get(request.getWorkerId()));
        logger.info("Worker marked offline. workerId={}", request.getWorkerId());
        return new SimpleOkResponse(true);
    }

    public SimpleOkResponse acceptMockMessage(MessageRequest request) {
        validateWorkerIsOnline(request.getWorkerId());
        InboundMessageEntity inboundMessageEntity = new InboundMessageEntity();
        inboundMessageEntity.setSessionKey(resolveSessionKey(request.getWorkerId()));
        inboundMessageEntity.setContent(request.getContent());
        inboundMessageEntity.setStatus(STATUS_RECEIVED);
        inboundMessageMapper.insert(inboundMessageEntity);
        logger.info(
            "Accepted inbound mock message. workerId={}, messageId={}, contentLength={}",
            request.getWorkerId(),
            inboundMessageEntity.getId(),
            request.getContent() == null ? 0 : request.getContent().length()
        );
        dispatchPendingMessages(request.getWorkerId(), false);
        return new SimpleOkResponse(true);
    }

    public SseEmitter stream(String workerId) {
        upsertWorkerStatus(workerId, STATUS_ONLINE);

        SseEmitter emitter = createEmitter(streamEmitterTimeoutMillis);
        StreamSession streamSession = new StreamSession(workerId, emitter);
        logger.info(
            "Opening SSE stream. workerId={}, sessionId={}, timeoutMillis={}",
            workerId,
            streamSession.getSessionId(),
            streamEmitterTimeoutMillis
        );
        StreamSession previousSession = streamSessions.put(workerId, streamSession);
        if (previousSession != null) {
            logger.warn(
                "Replacing existing SSE stream with newer connection. workerId={}, previousSessionId={}, currentSessionId={}",
                workerId,
                previousSession.getSessionId(),
                streamSession.getSessionId()
            );
            sendEventSafely(
                previousSession,
                EVENT_ERROR,
                Collections.<String, Object>singletonMap(
                    "message",
                    "connection replaced by newer stream"
                )
            );
            previousSession.complete();
        }

        registerEmitterCallbacks(workerId, streamSession);
        sendEventSafely(
            streamSession,
            EVENT_READY,
            Collections.<String, Object>singletonMap("worker_id", workerId)
        );
        dispatchPendingMessages(workerId, false);

        ScheduledFuture<?> heartbeatFuture = heartbeatExecutorService.scheduleAtFixedRate(
            new Runnable() {
                @Override
                public void run() {
                    sendHeartbeat(workerId, streamSession);
                }
            },
            streamHeartbeatIntervalMillis,
            streamHeartbeatIntervalMillis,
            TimeUnit.MILLISECONDS
        );
        streamSession.setHeartbeatFuture(heartbeatFuture);
        logger.info(
            "SSE stream opened. workerId={}, sessionId={}, heartbeatIntervalMillis={}",
            workerId,
            streamSession.getSessionId(),
            streamHeartbeatIntervalMillis
        );
        return emitter;
    }

    public SimpleOkResponse complete(CompleteRequest request) {
        OutboundMessageEntity outboundMessageEntity = new OutboundMessageEntity();
        outboundMessageEntity.setSessionKey(resolveSessionKey(request.getWorkerId()));
        outboundMessageEntity.setContent(request.getFinalContent());
        outboundMessageEntity.setStatus(STATUS_RECEIVED);
        outboundMessageMapper.insert(outboundMessageEntity);
        logger.info(
            "Accepted outbound completion. workerId={}, outboundMessageId={}, finalContentLength={}",
            request.getWorkerId(),
            outboundMessageEntity.getId(),
            request.getFinalContent() == null ? 0 : request.getFinalContent().length()
        );
        return new SimpleOkResponse(true);
    }

    @Override
    public void destroy() {
        for (StreamSession streamSession : streamSessions.values()) {
            streamSession.complete();
        }
        streamSessions.clear();
        heartbeatExecutorService.shutdownNow();
    }

    SseEmitter createEmitter(Long timeoutMillis) {
        if (timeoutMillis == null) {
            return new SseEmitter();
        }
        return new SseEmitter(timeoutMillis);
    }

    private void registerEmitterCallbacks(String workerId, StreamSession streamSession) {
        streamSession.getEmitter().onCompletion(new Runnable() {
            @Override
            public void run() {
                logger.info(
                    "SSE stream completed. workerId={}, sessionId={}",
                    workerId,
                    streamSession.getSessionId()
                );
                removeStreamSession(workerId, streamSession);
            }
        });
        streamSession.getEmitter().onTimeout(new Runnable() {
            @Override
            public void run() {
                logger.warn(
                    "SSE stream timed out. workerId={}, sessionId={}",
                    workerId,
                    streamSession.getSessionId()
                );
                removeStreamSession(workerId, streamSession);
                streamSession.getEmitter().complete();
            }
        });
        streamSession.getEmitter().onError(new java.util.function.Consumer<Throwable>() {
            @Override
            public void accept(Throwable throwable) {
                if (isClientDisconnect(throwable)) {
                    logger.info(
                        "SSE stream closed by client. workerId={}, sessionId={}, reason={}",
                        workerId,
                        streamSession.getSessionId(),
                        throwable == null ? "unknown" : throwable.getMessage()
                    );
                } else {
                    logger.warn(
                        "SSE stream closed with error. workerId={}, sessionId={}",
                        workerId,
                        streamSession.getSessionId(),
                        throwable
                    );
                }
                removeStreamSession(workerId, streamSession);
            }
        });
    }

    private void sendHeartbeat(String workerId, StreamSession streamSession) {
        StreamSession currentSession = streamSessions.get(workerId);
        if (currentSession != streamSession) {
            return;
        }
        if (!isWorkerOnline(workerId)) {
            logger.info("Closing local SSE stream because worker is offline in database. workerId={}", workerId);
            removeStreamSession(workerId, streamSession);
            streamSession.complete();
            return;
        }
        dispatchPendingMessages(workerId, true);
    }

    private void dispatchPendingMessages(String workerId, boolean emitHeartbeatWhenEmpty) {
        StreamSession streamSession = streamSessions.get(workerId);
        if (streamSession == null) {
            return;
        }

        boolean delivered = false;
        while (true) {
            InboundMessageEntity inboundMessageEntity = inboundMessageMapper.findFirstBySessionKeyAndStatus(
                resolveSessionKey(workerId),
                STATUS_RECEIVED
            );
            if (inboundMessageEntity == null) {
                break;
            }

            int updated = inboundMessageMapper.updateStatus(
                inboundMessageEntity.getId(),
                STATUS_RECEIVED,
                STATUS_CONSUMED
            );
            if (updated != 1) {
                logger.warn(
                    "Failed to update inbound message status before SSE dispatch. workerId={}, messageId={}",
                    workerId,
                    inboundMessageEntity.getId()
                );
                break;
            }

            delivered = true;
            logger.info(
                "Dispatching inbound message to SSE stream. workerId={}, sessionId={}, messageId={}",
                workerId,
                streamSession.getSessionId(),
                inboundMessageEntity.getId()
            );
            sendEventSafely(
                streamSession,
                EVENT_MESSAGE,
                Collections.<String, Object>singletonMap("content", inboundMessageEntity.getContent())
            );
        }

        if (!delivered && emitHeartbeatWhenEmpty) {
            sendEventSafely(
                streamSession,
                EVENT_HEARTBEAT,
                Collections.<String, Object>singletonMap("ts", System.currentTimeMillis())
            );
        }
    }

    private void sendEventSafely(StreamSession streamSession, String eventName, Map<String, Object> payload) {
        try {
            streamSession.getEmitter().send(
                SseEmitter.event().name(eventName).data(payload)
            );
        } catch (IOException exception) {
            if (isClientDisconnect(exception)) {
                logger.info(
                    "SSE client disconnected during event send. workerId={}, sessionId={}, event={}, reason={}",
                    streamSession.getWorkerId(),
                    streamSession.getSessionId(),
                    eventName,
                    exception.getMessage()
                );
            } else {
                logger.warn(
                    "Failed to send SSE event. workerId={}, sessionId={}, event={}",
                    streamSession.getWorkerId(),
                    streamSession.getSessionId(),
                    eventName,
                    exception
                );
            }
            removeStreamSession(streamSession.getWorkerId(), streamSession);
            streamSession.getEmitter().completeWithError(exception);
        }
    }

    private void removeStreamSession(String workerId, StreamSession expectedSession) {
        if (expectedSession == null) {
            return;
        }
        boolean removed = streamSessions.remove(workerId, expectedSession);
        if (!removed) {
            return;
        }
        expectedSession.cancelHeartbeat();
        logger.info("Removed local SSE stream session. workerId={}, sessionId={}", workerId, expectedSession.getSessionId());
    }

    private void validateWorkerIsOnline(String workerId) {
        if (!isWorkerOnline(workerId)) {
            logger.warn("Rejecting request because worker is offline. workerId={}", workerId);
            throw new ResponseStatusException(CONFLICT, "no_online_worker:" + workerId);
        }
    }

    private boolean isWorkerOnline(String workerId) {
        OnlineWorkerEntity onlineWorkerEntity = onlineWorkerMapper.findByWorkerId(workerId);
        return onlineWorkerEntity != null && STATUS_ONLINE.equalsIgnoreCase(onlineWorkerEntity.getStatus());
    }

    private void upsertWorkerStatus(String workerId, String status) {
        OnlineWorkerEntity onlineWorkerEntity = onlineWorkerMapper.findByWorkerId(workerId);
        if (onlineWorkerEntity == null) {
            try {
                onlineWorkerMapper.insert(new OnlineWorkerEntity(workerId, status, null, null));
                logger.info("Inserted online worker state. workerId={}, status={}", workerId, status);
                return;
            } catch (DuplicateKeyException exception) {
                logger.info("Online worker already exists when inserting. workerId={}", workerId);
            }
        }
        onlineWorkerMapper.updateStatus(workerId, status);
        logger.info("Updated worker state. workerId={}, status={}", workerId, status);
    }

    private String resolveSessionKey(String workerId) {
        return workerId;
    }

    private boolean isClientDisconnect(Throwable throwable) {
        Throwable current = throwable;
        while (current != null) {
            String className = current.getClass().getName();
            String message = current.getMessage();
            if ("org.apache.catalina.connector.ClientAbortException".equals(className)) {
                return true;
            }
            if (message != null) {
                String normalizedMessage = message.toLowerCase();
                if (normalizedMessage.contains("connection reset")
                    || normalizedMessage.contains("broken pipe")
                    || message.contains("你的主机中的软件中止了一个已建立的连接")) {
                    return true;
                }
            }
            current = current.getCause();
        }
        return false;
    }

    private static final class StreamSession {

        private final String sessionId;
        private final String workerId;
        private final SseEmitter emitter;
        private volatile ScheduledFuture<?> heartbeatFuture;

        private StreamSession(String workerId, SseEmitter emitter) {
            this.sessionId = UUID.randomUUID().toString();
            this.workerId = workerId;
            this.emitter = emitter;
        }

        private String getSessionId() {
            return sessionId;
        }

        private String getWorkerId() {
            return workerId;
        }

        private SseEmitter getEmitter() {
            return emitter;
        }

        private void setHeartbeatFuture(ScheduledFuture<?> heartbeatFuture) {
            this.heartbeatFuture = heartbeatFuture;
        }

        private void cancelHeartbeat() {
            ScheduledFuture<?> currentHeartbeatFuture = heartbeatFuture;
            if (currentHeartbeatFuture != null) {
                currentHeartbeatFuture.cancel(true);
            }
        }

        private void complete() {
            cancelHeartbeat();
            emitter.complete();
        }
    }
}
