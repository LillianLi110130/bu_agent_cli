package com.cmb.tg.tgai.service.worker;

import com.cmb.tg.tgai.infrastructure.common.holder.TokenContextHolder;
import com.cmb.tg.tgai.service.message.dto.SubmitWebMessageRequest;
import com.cmb.tg.tgai.service.message.dto.SubmitWebMessageResponse;
import com.cmb.tg.tgai.service.message.dto.WebWorkerSummaryResponse;
import com.cmb.tg.tgai.infrastructure.message.entity.InboundMessageEntity;
import com.cmb.tg.tgai.infrastructure.message.entity.OnlineWorker;
import com.cmb.tg.tgai.infrastructure.message.entity.OutboundMessageEntity;
import com.cmb.tg.tgai.infrastructure.message.mapper.InboundMessageMapper;
import com.cmb.tg.tgai.infrastructure.message.mapper.OnlineWorkerMapper;
import com.cmb.tg.tgai.infrastructure.message.mapper.OutboundMessageMapper;

import java.io.IOException;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.DisposableBean;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.web.server.ResponseStatusException;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

@Service
@Slf4j
@RequiredArgsConstructor
public class WebConsoleService implements DisposableBean {

    private static final Logger logger = LoggerFactory.getLogger(WebConsoleService.class);
    private static final String STATUS_COMPLETED = "COMPLETED";
    private static final String STATUS_PROGRESS = "PROGRESS";
    private static final String STATUS_FAILED = "FAILED";
    private static final String STATUS_SENT = "SENT";

    private final InboundMessageMapper inboundMessageMapper;
    private final OutboundMessageMapper outboundMessageMapper;
    private final OnlineWorkerMapper onlineWorkerMapper;
    private final ScheduledExecutorService heartbeatExecutorService;
    private final ConcurrentMap<String, StreamSession> streamSessions = new ConcurrentHashMap<>();

    @Value("${gateway.stream-heartbeat-interval-ms:10000}")
    private long streamHeartbeatIntervalMillis;

    @Value("${gateway.stream-emitter-timeout-ms:0}")
    private long streamEmitterTimeoutMillis;

    private boolean isWorkerOnline(String workerIdPrefix) {
        OnlineWorker onlineWorker = onlineWorkerMapper.findByWorkerIdPrefix(workerIdPrefix);
        return onlineWorker != null && onlineWorker.getStatus().equals("online");
    }

    public WebWorkerSummaryResponse getWorkerSummary(String workerId) {
        String workerIdPrefix = TokenContextHolder.getUserIdOfCurrentUser();

        return new WebWorkerSummaryResponse(
                workerIdPrefix,
                isWorkerOnline(workerIdPrefix)
        );
    }

    public SubmitWebMessageResponse submitMessage(SubmitWebMessageRequest request) {
        String workerIdPrefix = TokenContextHolder.getUserIdOfCurrentUser();
        if (!isWorkerOnline(workerIdPrefix)) {
            logger.warn("Rejecting web request because worker is offline. workerIdPrefix={}", workerIdPrefix);
            throw new ResponseStatusException(HttpStatus.SERVICE_UNAVAILABLE, "no_online_worker:" + workerIdPrefix);
        }

        String sessionKey = TokenContextHolder.getOpenIdOfCurrentUser();
        InboundMessageEntity inboundMessagePO = new InboundMessageEntity();
        inboundMessagePO.setSessionKey(sessionKey);
        inboundMessagePO.setSource("web");
        inboundMessagePO.setContent(request.getContent());
        inboundMessagePO.setStatus("RECEIVED");
        inboundMessageMapper.insert(inboundMessagePO);

        return new SubmitWebMessageResponse(true);
    }

    public SseEmitter stream(String workerId) {
        String workerIdPrefix = TokenContextHolder.getUserIdOfCurrentUser();
        String sessionKey = TokenContextHolder.getOpenIdOfCurrentUser();
        String ystId = TokenContextHolder.getYstIdOfCurrentUser();

        SseEmitter emitter = createEmitter(streamEmitterTimeoutMillis);
        StreamSession streamSession = new StreamSession(workerIdPrefix, sessionKey, emitter);
        logger.info(
                "Opening web SSE stream. workerIdPrefix={}, sessionId={}, ystId={}, timeoutMillis={}",
                workerIdPrefix,
                streamSession.getSessionId(),
                ystId,
                streamEmitterTimeoutMillis
        );

        StreamSession previousSession = streamSessions.put(sessionKey, streamSession);
        if (previousSession != null) {
            logger.warn(
                    "Replacing existing web SSE stream with newer connection. sessionKey={}, previousSessionId={}, currentSessionId={}",
                    sessionKey,
                    previousSession.getSessionId(),
                    streamSession.getSessionId()
            );
            previousSession.complete();
        }

        registerEmitterCallbacks(sessionKey, streamSession);
        dispatchPendingWebEvents(sessionKey, false);

        ScheduledFuture<?> heartbeatFuture = heartbeatExecutorService.scheduleAtFixedRate(
                () -> sendHeartbeat(sessionKey, streamSession),
                streamHeartbeatIntervalMillis,
                streamHeartbeatIntervalMillis,
                TimeUnit.MILLISECONDS
        );
        streamSession.setHeartbeatFuture(heartbeatFuture);
        logger.info(
                "SSE stream opened. workerIdPrefix={}, sessionId={}, ystId={}, heartbeatIntervalMillis={}",
                workerIdPrefix,
                streamSession.getSessionId(),
                ystId,
                streamHeartbeatIntervalMillis
        );
        return emitter;
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
        streamSession.getEmitter().onCompletion(() -> {
            logger.info(
                    "Web SSE stream completed. sessionKey={}, sessionId={}",
                    sessionKey,
                    streamSession.getSessionId()
            );
            removeStreamSession(sessionKey, streamSession);
        });
        streamSession.getEmitter().onTimeout(() -> {
            logger.warn(
                    "Web SSE stream timed out. sessionKey={}, sessionId={}",
                    sessionKey,
                    streamSession.getSessionId()
            );
            removeStreamSession(sessionKey, streamSession);
            streamSession.getEmitter().complete();
        });
        streamSession.getEmitter().onError(throwable -> {
            if (isClientDisconnect(throwable)) {
                logger.info(
                        "Web SSE stream closed by client. sessionKey={}, sessionId={}, reason={}",
                        sessionKey,
                        streamSession.getSessionId(),
                        throwable == null ? "unknown" : throwable.getMessage()
                );
            } else {
                logger.warn(
                        "Web SSE stream closed with error. sessionKey={}, sessionId={}, reason={}",
                        sessionKey,
                        streamSession.getSessionId(),
                        throwable == null ? "unknown" : throwable.getMessage()
                );
            }
            removeStreamSession(sessionKey, streamSession);
        });
    }

    private void sendHeartbeat(String sessionKey, StreamSession streamSession) {
        StreamSession currentSession = streamSessions.get(sessionKey);
        if (currentSession != streamSession) {
            return;
        }
        dispatchPendingWebEvents(sessionKey, true);
    }

    public void dispatchPendingWebEvents(String sessionKey, boolean emitHeartbeatWhenEmpty) {
        StreamSession streamSession = streamSessions.get(sessionKey);
        if (streamSession == null) {
            return;
        }

        boolean delivered = false;

        InboundMessageEntity inboundMessagePO = inboundMessageMapper.fetchInboundMessage(sessionKey);
        if (inboundMessagePO != null) {
            delivered = true;
            String workerIdPrefix = streamSession.getUserNo();
            if ("RECEIVED".equalsIgnoreCase(inboundMessagePO.getStatus())) {
                sendEventSafely(
                        streamSession,
                        "submitted",
                        buildEventPayload("submitted", workerIdPrefix, inboundMessagePO.getCreatedAt(), null)
                );
            } else if (
                    "CONSUMED".equalsIgnoreCase(inboundMessagePO.getStatus())
            ) {
                sendEventSafely(
                        streamSession,
                        "processing",
                        buildEventPayload("processing", workerIdPrefix, inboundMessagePO.getCreatedAt(), null)
                );
            }
        }

        while (dispatchNextOutboundEvent(streamSession, sessionKey)) {
            delivered = true;
        }

        if (!delivered && emitHeartbeatWhenEmpty) {
            sendHeartbeatSafely(streamSession);
        }
    }

    private Map<String, Object> buildEventPayload(String type, String workerId, LocalDateTime localDateTime, String content) {
        Map<String, Object> payload = new LinkedHashMap<String, Object>();
        payload.put("type", type);
        payload.put("workerId", workerId);
        if (localDateTime != null) {
            if ("completed".equals(type) || "failed".equals(type)) {
                payload.put("finishedAt", toIso(localDateTime));
            } else {
                payload.put("ts", toIso(localDateTime));
            }
        }
        if (content != null) {
            if ("progress".equals(type)) {
                payload.put("content", content);
            } else if ("completed".equals(type)) {
                payload.put("finalContent", content);
            } else if ("failed".equals(type)) {
                payload.put("errorMessage", content);
            }
        }
        return payload;
    }

    private boolean dispatchNextOutboundEvent(StreamSession streamSession, String sessionKey) {
        OutboundMessageEntity outboundMessageEntity = outboundMessageMapper.fetchOutboundMessage(sessionKey);
        if (outboundMessageEntity == null) {
            return false;
        }

        String currentStatus = outboundMessageEntity.getStatus();
        String eventType = toWebEventType(currentStatus);
        if (eventType == null) {
            return false;
        }

        int updated = outboundMessageMapper.updateCurrentStatus(
                outboundMessageEntity.getId(),
                currentStatus,
                STATUS_SENT
        );
        if (updated != 1) {
            logger.warn(
                    "Failed to update outbound message status before web SSE dispatch. sessionKey={}, messageId={}, status={}",
                    sessionKey,
                    outboundMessageEntity.getId(),
                    currentStatus
            );
            return false;
        }

        sendEventSafely(
                streamSession,
                eventType,
                buildEventPayload(
                        eventType,
                        streamSession.getUserNo(),
                        outboundMessageEntity.getCreatedAt(),
                        outboundMessageEntity.getContent()
                )
        );
        if ("completed".equals(eventType) || "failed".equals(eventType)) {
            sendDoneSafely(streamSession);
            removeStreamSession(streamSession.getSessionKey(), streamSession);
            streamSession.complete();
        }
        return true;
    }

    private String toWebEventType(String outboundStatus) {
        if (STATUS_PROGRESS.equalsIgnoreCase(outboundStatus)) {
            return "progress";
        }
        if (STATUS_COMPLETED.equalsIgnoreCase(outboundStatus)) {
            return "completed";
        }
        if (STATUS_FAILED.equalsIgnoreCase(outboundStatus)) {
            return "failed";
        }
        return null;
    }

    private void sendEventSafely(StreamSession streamSession, String eventName, Map<String, Object> payload) {
        try {
            streamSession.getEmitter().send(
                    SseEmitter.event().name(eventName).data(payload)
            );
        } catch (IOException exception) {
            if (isClientDisconnect(exception)) {
                logger.info(
                        "Web SSE client disconnected during event send. sessionKey={}, sessionId={}, event={}, reason={}",
                        streamSession.getSessionKey(),
                        streamSession.getSessionId(),
                        eventName,
                        exception.getMessage()
                );
            } else {
                logger.warn(
                        "Failed to send web SSE event. sessionKey={}, sessionId={}, event={}",
                        streamSession.getSessionKey(),
                        streamSession.getSessionId(),
                        eventName,
                        exception
                );
            }
            removeStreamSession(streamSession.getSessionKey(), streamSession);
            streamSession.getEmitter().completeWithError(exception);
        }
    }

    private void sendHeartbeatSafely(StreamSession streamSession) {
        try {
            streamSession.getEmitter().send(SseEmitter.event().comment("heartbeat"));
        } catch (IOException exception) {
            if (isClientDisconnect(exception)) {
                logger.info(
                        "Web SSE client disconnected during heartbeat. sessionKey={}, sessionId={}, reason={}",
                        streamSession.getSessionKey(),
                        streamSession.getSessionId(),
                        exception.getMessage()
                );
            } else {
                logger.warn(
                        "Failed to send web SSE heartbeat. sessionKey={}, sessionId={}",
                        streamSession.getSessionKey(),
                        streamSession.getSessionId(),
                        exception
                );
            }
            removeStreamSession(streamSession.getSessionKey(), streamSession);
            streamSession.getEmitter().completeWithError(exception);
        }
    }

    private void sendDoneSafely(StreamSession streamSession) {
        try {
            streamSession.getEmitter().send(SseEmitter.event().comment("done"));
        } catch (IOException exception) {
            if (isClientDisconnect(exception)) {
                logger.info(
                        "Web SSE client disconnected during done marker. sessionKey={}, sessionId={}, reason={}",
                        streamSession.getSessionKey(),
                        streamSession.getSessionId(),
                        exception.getMessage()
                );
            } else {
                logger.warn(
                        "Failed to send web SSE done marker. sessionKey={}, sessionId={}",
                        streamSession.getSessionKey(),
                        streamSession.getSessionId(),
                        exception
                );
            }
        }
    }

    private void removeStreamSession(String sessionKey, StreamSession expectedSession) {
        if (expectedSession == null) {
            return;
        }
        boolean removed = streamSessions.remove(sessionKey, expectedSession);
        if (!removed) {
            return;
        }
        expectedSession.cancelHeartbeat();
        logger.info(
                "Removed web SSE stream session. sessionKey={}, sessionId={}",
                sessionKey,
                expectedSession.getSessionId()
        );
    }

    private String toIso(LocalDateTime localDateTime) {
        if (localDateTime == null) {
            return null;
        }
        return localDateTime
                .atZone(ZoneId.systemDefault())
                .toInstant()
                .toString();
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
        private final String sessionId;
        private final String sessionKey;
        private final SseEmitter emitter;
        private volatile ScheduledFuture<?> heartbeatFuture;

        private StreamSession(String userNo, String sessionKey, SseEmitter emitter) {
            this.userNo = userNo;
            this.sessionId = UUID.randomUUID().toString();
            this.sessionKey = sessionKey;
            this.emitter = emitter;
        }

        private String getUserNo() {
            return userNo;
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
