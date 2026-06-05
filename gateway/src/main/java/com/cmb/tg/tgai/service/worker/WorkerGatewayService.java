package com.cmb.tg.tgai.service.worker;

import com.cmb.tg.tgai.infrastructure.common.holder.TokenContextHolder;
import com.cmb.tg.tgai.service.message.dto.CompleteRequest;
import com.cmb.tg.tgai.service.message.dto.MessageRequest;
import com.cmb.tg.tgai.service.message.dto.ProgressRequest;
import com.cmb.tg.tgai.service.message.dto.SendTextRequest;
import com.cmb.tg.tgai.service.message.dto.SimpleOkResponse;
import com.cmb.tg.tgai.service.message.dto.WorkerRequest;
import com.cmb.tg.tgai.infrastructure.message.entity.InboundMessageEntity;
import com.cmb.tg.tgai.infrastructure.message.entity.OnlineWorkerEntity;
import com.cmb.tg.tgai.infrastructure.message.entity.OutboundMessageEntity;
import com.cmb.tg.tgai.infrastructure.message.mapper.InboundMessageMapper;
import com.cmb.tg.tgai.infrastructure.message.mapper.OnlineWorkerMapper;
import com.cmb.tg.tgai.infrastructure.message.mapper.OutboundMessageMapper;
import lombok.RequiredArgsConstructor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.DisposableBean;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.dao.DuplicateKeyException;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.io.IOException;
import java.time.LocalDateTime;
import java.util.Collections;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;

@Service
@RequiredArgsConstructor
public class WorkerGatewayService implements DisposableBean {

    private static final Logger logger = LoggerFactory.getLogger(WorkerGatewayService.class);

    private static final String STATUS_ONLINE = "online";
    private static final String STATUS_OFFLINE = "offline";
    private static final String STATUS_RECEIVED = "RECEIVED";
    private static final String STATUS_CONSUMED = "CONSUMED";
    private static final String SOURCE_IM = "im";
    private static final String STATUS_PROGRESS = "PROGRESS";
    private static final String STATUS_COMPLETED = "COMPLETED";
    private static final String STATUS_FAILED = "FAILED";
    private static final String EVENT_READY = "ready";
    private static final String EVENT_MESSAGE = "message";
    private static final String EVENT_HEARTBEAT = "heartbeat";
    private static final String EVENT_ERROR = "error";

    private final InboundMessageMapper inboundMessageMapper;
    private final OutboundMessageMapper outboundMessageMapper;
    private final OnlineWorkerMapper onlineWorkerMapper;
    private final WebConsoleService webConsoleService;
    private final ScheduledExecutorService heartbeatExecutorService;
    private final ConcurrentMap<String, StreamSession> streamSessions = new ConcurrentHashMap<String, StreamSession>();

    @Value("${gateway.stream-heartbeat-interval-ms:10000}")
    private long streamHeartbeatIntervalMillis;
    @Value("${gateway.stream-emitter-timeout-ms:1500000}")
    private long streamEmitterTimeoutMillis;

    private String buildWorkerId(String userNo, String workerNo){
        return userNo + '-' + workerNo;
    }

    public SimpleOkResponse online(WorkerRequest request) {
        String userNo = TokenContextHolder.getUserIdOfCurrentUser();
        upsertWorkerStatus(buildWorkerId(userNo, request.getWorkerNo()), STATUS_ONLINE);
        logger.info("Worker marked online. workerNo={}", request.getWorkerNo());
        return new SimpleOkResponse(true);
    }

    public SimpleOkResponse offline(WorkerRequest request) {
        String userNo = TokenContextHolder.getUserIdOfCurrentUser();
        String openId = TokenContextHolder.getOpenIdOfCurrentUser();
        upsertWorkerStatus(buildWorkerId(userNo, request.getWorkerNo()), STATUS_OFFLINE);
        String sessionKey = buildSessionKey(openId, request.getWorkerNo());
        StreamSession streamSession = streamSessions.get(sessionKey);
        if (removeStreamSession(sessionKey, streamSession)) {
            // Offline is an application-level close signal. Complete the emitter here instead
            // of relying on the next heartbeat to discover a dead client socket.
            streamSession.complete();
        }
        logger.info("Worker marked offline. workerId={}", request.getWorkerNo());
        return new SimpleOkResponse(true);
    }

    public SimpleOkResponse acceptMockMessage(MessageRequest request) {
        String sessionKey = TokenContextHolder.getOpenIdOfCurrentUser();
        InboundMessageEntity inboundMessageEntity = new InboundMessageEntity();
        inboundMessageEntity.setSessionKey(sessionKey);
        inboundMessageEntity.setSource(SOURCE_IM);
        inboundMessageEntity.setContent(request.getContent());
        inboundMessageEntity.setStatus(STATUS_RECEIVED);
        inboundMessageEntity.setCreatedAt(LocalDateTime.now());
        inboundMessageMapper.insert(inboundMessageEntity);
        logger.info(
            "Accepted inbound mock message. sessionKey={}, messageId={}, contentLength={}",
                sessionKey,
            inboundMessageEntity.getId(),
            request.getContent() == null ? 0 : request.getContent().length()
        );
        dispatchPendingMessages(null, TokenContextHolder.getOpenIdOfCurrentUser(), false);
        return new SimpleOkResponse(true);
    }

    public SseEmitter stream(String workerNo) {
        String userNo = TokenContextHolder.getUserIdOfCurrentUser();
        String openId = TokenContextHolder.getOpenIdOfCurrentUser();
        String sessionKey = buildSessionKey(openId, workerNo);
        upsertWorkerStatus(buildWorkerId(userNo, workerNo), STATUS_ONLINE);
        SseEmitter emitter = createEmitter(streamEmitterTimeoutMillis);
        StreamSession streamSession = new StreamSession(userNo, workerNo, sessionKey, emitter);
        logger.info(
            "Opening SSE stream. sessionKey={}, sessionId={}, timeoutMillis={}",
            sessionKey,
            streamSession.getSessionId(),
            streamEmitterTimeoutMillis
        );
        StreamSession previousSession = streamSessions.put(sessionKey, streamSession);
        if (previousSession != null) {
            logger.warn(
                "Replacing existing SSE stream with newer connection. sessionKey={}, previousSessionId={}, currentSessionId={}",
                sessionKey,
                previousSession.getSessionId(),
                streamSession.getSessionId()
            );
            previousSession.complete();
        }

        registerEmitterCallbacks(sessionKey, streamSession);
        sendEventSafely(
            streamSession,
            EVENT_READY,
            Collections.<String, Object>singletonMap("sessionKey", sessionKey)
        );
        dispatchPendingMessages(sessionKey, openId, false);

        ScheduledFuture<?> heartbeatFuture = heartbeatExecutorService.scheduleAtFixedRate(
            new Runnable() {
                @Override
                public void run() {
                    sendHeartbeat(workerNo, openId, sessionKey, streamSession);
                }
            },
            streamHeartbeatIntervalMillis,
            streamHeartbeatIntervalMillis,
            TimeUnit.MILLISECONDS
        );
        streamSession.setHeartbeatFuture(heartbeatFuture);
        logger.info(
            "SSE stream opened. sessionKey={}, sessionId={}, heartbeatIntervalMillis={}",
            sessionKey,
            streamSession.getSessionId(),
            streamHeartbeatIntervalMillis
        );
        return emitter;
    }

    public SimpleOkResponse complete(CompleteRequest request) {
        if ("web".equalsIgnoreCase(request.getSource())) {
            String sessionKey = TokenContextHolder.getOpenIdOfCurrentUser();
            OutboundMessageEntity outboundMessageEntity = new OutboundMessageEntity();
            outboundMessageEntity.setSessionKey(sessionKey);
            outboundMessageEntity.setSource("web");
            if ("failed".equalsIgnoreCase(request.getFinalStatus())) {
                outboundMessageEntity.setContent(
                    request.getErrorMessage() == null ? request.getFinalContent() : request.getErrorMessage()
                );
                outboundMessageEntity.setStatus(STATUS_FAILED);
            } else {
                outboundMessageEntity.setContent(request.getFinalContent());
                outboundMessageEntity.setStatus(STATUS_COMPLETED);
            }
            outboundMessageEntity.setCreatedAt(LocalDateTime.now());
            outboundMessageMapper.insert(outboundMessageEntity);
            webConsoleService.dispatchPendingWebEvents(sessionKey, false);
            return new SimpleOkResponse(true);
        }

        OutboundMessageEntity outboundMessageEntity = new OutboundMessageEntity();
        outboundMessageEntity.setSessionKey(TokenContextHolder.getOpenIdOfCurrentUser());
        outboundMessageEntity.setSource("im");
        outboundMessageEntity.setContent(request.getFinalContent());
        outboundMessageEntity.setStatus("SENT");
        outboundMessageEntity.setCreatedAt(LocalDateTime.now());
        outboundMessageMapper.insert(outboundMessageEntity);
        logger.info(
            "Accepted outbound completion. workerId={}, outboundMessageId={}, finalStatus={}, "
                + "errorCode={}, finalContentLength={}",
            request.getWorkerNo(),
            outboundMessageEntity.getId(),
            request.getFinalStatus(),
            request.getErrorCode(),
            request.getFinalContent() == null ? 0 : request.getFinalContent().length()
        );
        return new SimpleOkResponse(true);
    }

    public SimpleOkResponse progress(ProgressRequest request) {
        if ("web".equalsIgnoreCase(request.getSource())) {
            String sessionKey = TokenContextHolder.getOpenIdOfCurrentUser();
            OutboundMessageEntity outboundMessageEntity = new OutboundMessageEntity();
            outboundMessageEntity.setSessionKey(sessionKey);
            outboundMessageEntity.setSource("web");
            outboundMessageEntity.setContent(request.getContent());
            outboundMessageEntity.setStatus(STATUS_PROGRESS);
            outboundMessageEntity.setCreatedAt(LocalDateTime.now());
            outboundMessageMapper.insert(outboundMessageEntity);
            webConsoleService.dispatchPendingWebEvents(sessionKey, false);
            return new SimpleOkResponse(true);
        }

        OutboundMessageEntity outboundMessageEntity = new OutboundMessageEntity();
        outboundMessageEntity.setSessionKey(TokenContextHolder.getOpenIdOfCurrentUser());
        outboundMessageEntity.setSource("im");
        outboundMessageEntity.setContent(request.getContent());
        outboundMessageEntity.setStatus("SENT");
        outboundMessageEntity.setCreatedAt(LocalDateTime.now());
        outboundMessageMapper.insert(outboundMessageEntity);
        logger.info(
            "Accepted outbound progress. workerId={}, outboundMessageId={}, finalContentLength={}",
            request.getWorkerNo(),
            outboundMessageEntity.getId(),
            request.getContent() == null ? 0 : request.getContent().length()
        );
        return new SimpleOkResponse(true);
    }

    public SimpleOkResponse sendText(SendTextRequest request) {
        return new SimpleOkResponse(true);
    }

    public SimpleOkResponse uploadAttachment(MultipartFile file) throws IOException {
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

    private void registerEmitterCallbacks(String sessionKey, StreamSession streamSession) {
        streamSession.getEmitter().onCompletion(new Runnable() {
            @Override
            public void run() {
                logger.info(
                    "SSE stream completed. sessionKey={}, sessionId={}",
                    sessionKey,
                    streamSession.getSessionId()
                );
                removeStreamSession(sessionKey, streamSession);
            }
        });
        streamSession.getEmitter().onTimeout(new Runnable() {
            @Override
            public void run() {
                logger.warn(
                    "SSE stream timed out. sessionKey={}, sessionId={}",
                    sessionKey,
                    streamSession.getSessionId()
                );
                disconnectStreamSession(sessionKey, streamSession);
                streamSession.getEmitter().complete();
            }
        });
        streamSession.getEmitter().onError(new java.util.function.Consumer<Throwable>() {
            @Override
            public void accept(Throwable throwable) {
                if (isClientDisconnect(throwable)) {
                    logger.info(
                        "SSE stream closed by client. sessionKey={}, sessionId={}, reason={}",
                        sessionKey,
                        streamSession.getSessionId(),
                        throwable == null ? "unknown" : throwable.getMessage()
                    );
                } else {
                    logger.warn(
                        "SSE stream closed with error. sessionKey={}, sessionId={}, reason={}",
                        sessionKey,
                        streamSession.getSessionId(),
                        throwable == null ? "unknown" : throwable.getMessage()
                    );
                }
                disconnectStreamSession(sessionKey, streamSession);
            }
        });
    }

    private void sendHeartbeat(String workerNo, String openId, String sessionKey, StreamSession streamSession) {
        String userNo = TokenContextHolder.getUserIdOfCurrentUser();
        StreamSession currentSession = streamSessions.get(sessionKey);
        if (currentSession != streamSession) {
            return;
        }
        if (!isWorkerOnline(buildWorkerId(userNo, workerNo))) {
            logger.info("Closing local SSE stream because worker is offline in database. workerNo={}", workerNo);
            removeStreamSession(sessionKey, streamSession);
            streamSession.complete();
            return;
        }
        dispatchPendingMessages(sessionKey, openId,true);
    }

    private void dispatchPendingMessages(String sessionKey, String openId, boolean emitHeartbeatWhenEmpty) {
        String targetSessionKey = getTargetSessionKey(sessionKey, openId, emitHeartbeatWhenEmpty);
        if (targetSessionKey == null) return;

        StreamSession streamSession = streamSessions.get(targetSessionKey);
        if (streamSession == null) {
            return;
        }

        boolean delivered = false;
        while (true) {
            InboundMessageEntity inboundMessageEntity = inboundMessageMapper.findFirstBySessionKeyAndStatus(
                openId,
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
                    "Failed to update inbound message status before SSE dispatch. sessionKey={}, messageId={}",
                    sessionKey,
                    inboundMessageEntity.getId()
                );
                break;
            }

            delivered = true;
            logger.info(
                "Dispatching inbound message to SSE stream. sessionKey={}, sessionId={}, messageId={}",
                targetSessionKey,
                streamSession.getSessionId(),
                inboundMessageEntity.getId()
            );
            sendEventSafely(
                streamSession,
                EVENT_MESSAGE,
                buildWorkerMessagePayload(inboundMessageEntity)
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

    private String getTargetSessionKey(String sessionKey, String openId, boolean emitHeartbeatWhenEmpty) {
        OnlineWorkerEntity onlineWorkerEntity = onlineWorkerMapper.findByWorkerIdPrefix(TokenContextHolder.getUserIdOfCurrentUser());
        if (onlineWorkerEntity == null) {
            return null;
        }
        String workerId = onlineWorkerEntity.getWorkerId();
        int separatorIndex = workerId.lastIndexOf('-');
        if (separatorIndex < 0 || separatorIndex == workerId.length() - 1) {
            logger.warn("Invalid workerId format when dispatching inbound message. workerId={}", workerId);
            return null;
        }
        String workerNo = workerId.substring(separatorIndex + 1);
        String targetSessionKey = buildSessionKey(openId, workerNo);

        if (sessionKey != null && !targetSessionKey.equals(sessionKey)) {
            StreamSession currentStreamSession = streamSessions.get(sessionKey);
            if (currentStreamSession != null && emitHeartbeatWhenEmpty) {
                sendEventSafely(
                    currentStreamSession,
                    EVENT_HEARTBEAT,
                    Collections.singletonMap("ts", System.currentTimeMillis())
                );
            }
            return null;
        }
        return targetSessionKey;
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
                    streamSession.getSessionKey(),
                    streamSession.getSessionId(),
                    eventName,
                    exception.getMessage()
                );
            } else {
                logger.warn(
                    "Failed to send SSE event. workerId={}, sessionId={}, event={}",
                    streamSession.getSessionKey(),
                    streamSession.getSessionId(),
                    eventName,
                    exception
                );
            }
            disconnectStreamSession(streamSession.getSessionKey(), streamSession);
            streamSession.getEmitter().completeWithError(exception);
        }
    }

    private Map<String, Object> buildWorkerMessagePayload(InboundMessageEntity inboundMessageEntity) {
        Map<String, Object> payload = new ConcurrentHashMap<String, Object>();
        payload.put("content", inboundMessageEntity.getContent());
        payload.put("source", inboundMessageEntity.getSource());
        return payload;
    }

    private boolean removeStreamSession(String sessionKey, StreamSession expectedSession) {
        if (expectedSession == null) {
            return false;
        }
        // Only the current session owner should cancel its heartbeat or close its emitter.
        // A stale callback from a replaced connection must not affect the newer stream.
        boolean removed = streamSessions.remove(sessionKey, expectedSession);
        if (!removed) {
            return false;
        }
        expectedSession.cancelHeartbeat();
        logger.info("Removed local SSE stream session. sessionKey={}, sessionId={}", sessionKey, expectedSession.getSessionId());
        return true;
    }

    private void disconnectStreamSession(String sessionKey, StreamSession expectedSession) {
        boolean removed = removeStreamSession(sessionKey, expectedSession);
        if (!removed) {
            return;
        }
        upsertWorkerStatus(buildWorkerId(expectedSession.getUserNo(), expectedSession.getWorkerNo()), STATUS_OFFLINE);
        logger.info(
                "Marked worker offline after SSE disconnect. sessionKey={}, sessionId={}, workerNo={}",
                sessionKey,
                expectedSession.getSessionId(),
                expectedSession.getWorkerNo()
        );
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

    private String buildSessionKey(String openId, String workerNo) {
        return openId + "-" + workerNo;
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

        private final String userNo;
        private final String workerNo;
        private final String sessionId;
        private final String sessionKey;
        private final SseEmitter emitter;
        private volatile ScheduledFuture<?> heartbeatFuture;

        private StreamSession(String userNo, String workerNo, String sessionKey, SseEmitter emitter) {
            this.userNo = userNo;
            this.workerNo = workerNo;
            this.sessionId = UUID.randomUUID().toString();
            this.sessionKey = sessionKey;
            this.emitter = emitter;
        }

        private String getUserNo() {
            return userNo;
        }

        private String getWorkerNo() {
            return workerNo;
        }

        private String getSessionId() {
            return sessionId;
        }

        private String getSessionKey() {
            return sessionKey;
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
                currentHeartbeatFuture.cancel(false);
            }
        }

        private void complete() {
            cancelHeartbeat();
            emitter.complete();
        }
    }
}
