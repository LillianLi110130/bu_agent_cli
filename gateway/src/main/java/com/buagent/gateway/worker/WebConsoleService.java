package com.buagent.gateway.worker;

import com.buagent.gateway.app.dto.SubmitWebMessageRequest;
import com.buagent.gateway.app.dto.SubmitWebMessageResponse;
import com.buagent.gateway.app.dto.WebWorkerSummaryResponse;
import com.buagent.gateway.store.entity.InboundMessageEntity;
import com.buagent.gateway.store.entity.OnlineWorkerEntity;
import com.buagent.gateway.store.entity.OutboundMessageEntity;
import com.buagent.gateway.store.mapper.InboundMessageMapper;
import com.buagent.gateway.store.mapper.OnlineWorkerMapper;
import com.buagent.gateway.store.mapper.OutboundMessageMapper;
import java.io.IOException;
import java.time.Instant;
import java.time.LocalDateTime;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.Executors;
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

    private String buildWorkerId(String channel, String ystId) {
        return channel + "_" + ystId;
    }

    private String buildSessionKey(String channel, String id) {
        return channel + "_" + id;
    }

    private boolean isWorkerOnline(String workerId) {
        OnlineWorker onlineWorker = onlineWorkerMapper.findByWorkerId(workerId);
        return onlineWorker != null && onlineWorker.getStatus().equals(WorkerStatusEnum.ONLINE.getCode());
    }

    public WebWorkerSummaryResponse getWorkerSummary() {
        String workerId = buildWorkerId(TgAiServerConstants.WEBSITE_WEB, TokenContextHolder.getYstIdOfCurrentUser());

        return new WebWorkerSummaryResponse(
                workerId,
                isWorkerOnline(workerId)
        );
    }

    public SubmitWebMessageResponse submitMessage(SubmitWebMessageRequest request) {
        if (!isWorkerOnline(buildWorkerId(TgAiServerConstants.WEBSITE_WEB, TokenContextHolder.getYstIdOfCurrentUser()))) {
            logger.warn("Rejecting web request because worker is offline. workerId={}", TokenContextHolder.getYstIdOfCurrentUser());
            throw new ResponseStatusException(HttpStatus.SERVICE_UNAVAILABLE, "no_online_worker:" + TokenContextHolder.getYstIdOfCurrentUser());
        }

        String sessionKey = buildSessionKey(TgAiServerConstants.WEBSITE_WEB, TokenContextHolder.getOpenIdOfCurrentUser());
        InboundMessagePO inboundMessagePO = new InboundMessagePO();
        inboundMessagePO.setSessionKey(sessionKey);
        inboundMessagePO.setContent(request.getContent());
        inboundMessagePO.setStatus(InboundMessageStatusEnum.RECEIVED.getCode());
        inboundMessagePO.setMsgType("text");
        inboundMessageMapper.insert(inboundMessagePO);

        return new SubmitWebMessageResponse(true);
    }

    public SseEmitter stream() {
        String workerId = buildWorkerId(TgAiServerConstants.WEBSITE_WEB, TokenContextHolder.getYstIdOfCurrentUser());
        String sessionKey = buildSessionKey(TgAiServerConstants.WEBSITE_WEB, TokenContextHolder.getOpenIdOfCurrentUser());

        SseEmitter emitter = createEmitter(streamEmitterTimeoutMillis);
        StreamSession streamSession = new StreamSession(TokenContextHolder.getYstIdOfCurrentUser(), sessionKey, emitter);
        logger.info(
                "Opening web SSE stream. sessionKey={}, sessionId={}, ystId={}, timeoutMillis={}",
                sessionKey,
                streamSession.getSessionId(),
                TokenContextHolder.getYstIdOfCurrentUser(),
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
                "SSE stream opened. workerId={}, sessionId={}, ystId={}, heartbeatIntervalMillis={}",
                workerId,
                streamSession.getSessionId(),
                TokenContextHolder.getYstIdOfCurrentUser(),
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
                    "Web SSE stream completed. workerId={}, sessionId={}",
                    sessionKey,
                    streamSession.getSessionId()
            );
            removeStreamSession(sessionKey, streamSession);
        });
        streamSession.getEmitter().onTimeout(() -> {
            logger.warn(
                    "Web SSE stream timed out. workerId={}, sessionId={}",
                    sessionKey,
                    streamSession.getSessionId()
            );
            removeStreamSession(sessionKey, streamSession);
            streamSession.getEmitter().complete();
        });
        streamSession.getEmitter().onError(throwable -> {
            if (isClientDisconnect(throwable)) {
                logger.info(
                        "Web SSE stream closed by client. workerId={}, sessionId={}, reason={}",
                        sessionKey,
                        streamSession.getSessionId(),
                        throwable == null ? "unknown" : throwable.getMessage()
                );
            } else {
                logger.warn(
                        "Web SSE stream closed with error. workerId={}, sessionId={}",
                        sessionKey,
                        streamSession.getSessionId(),
                        throwable
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
        String workerId = buildWorkerId(TgAiServerConstants.WEBSITE_WEB, currentSession.getYstId());
        if (!isWorkerOnline(workerId)) {
            logger.info("Closing web SSE stream because worker is offline in database. workerId={}", workerId);
            removeStreamSession(sessionKey, streamSession);
            streamSession.complete();
            return;
        }
        dispatchPendingWebEvents(sessionKey, true);
    }

    private void dispatchPendingWebEvents(String sessionKey, boolean emitHeartbeatWhenEmpty) {
        StreamSession streamSession = streamSessions.get(sessionKey);
        if (streamSession == null) {
            return;
        }

        boolean delivered = false;

        InboundMessagePO inboundMessagePO = inboundMessageMapper.fetchInboundMessage(sessionKey);
        if (inboundMessagePO != null) {
            delivered = true;
            String workerId = buildWorkerId(TgAiServerConstants.WEBSITE_WEB, streamSession.getYstId());
            if (InboundMessageStatusEnum.RECEIVED.getCode().equalsIgnoreCase(inboundMessagePO.getStatus())) {
                sendEventSafely(
                        streamSession,
                        SseEventEnum.SUBMITTED.getCode(),
                        buildEventPayload("submitted", workerId, inboundMessagePO.getCreateTime(), null)
                );
            } else if (
                    InboundMessageStatusEnum.CONSUMED.getCode().equalsIgnoreCase(inboundMessagePO.getStatus())
                            && !hasTerminalOutboundEvent(sessionKey)
            ) {
                sendEventSafely(
                        streamSession,
                        SseEventEnum.PROCESSING.getCode(),
                        buildEventPayload("processing", workerId, inboundMessagePO.getCreateTime(), null)
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

    private boolean hasTerminalOutboundEvent(String sessionKey) {
        OutboundMessagePO latestOutboundMessage = outboundMessageMapper.findLatestMessage(sessionKey);
        if (latestOutboundMessage == null || latestOutboundMessage.getStatus() == null) {
            return false;
        }
        String status = latestOutboundMessage.getStatus();
        return STATUS_COMPLETED.equalsIgnoreCase(status)
                || STATUS_FAILED.equalsIgnoreCase(status)
                || STATUS_SENT.equalsIgnoreCase(status);
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
        OutboundMessagePO outboundMessageEntity = outboundMessageMapper.fetchOutboundMessage(sessionKey);
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
                        buildWorkerId(TgAiServerConstants.WEBSITE_WEB, streamSession.getYstId()),
                        outboundMessageEntity.getCreateTime(),
                        outboundMessageEntity.getContent()
                )
        );
        return true;
    }

    private String toWebEventType(String outboundStatus) {
        if (STATUS_PROGRESS.equalsIgnoreCase(outboundStatus)) {
            return SseEventEnum.PROGRESS.getCode();
        }
        if (STATUS_COMPLETED.equalsIgnoreCase(outboundStatus)) {
            return SseEventEnum.COMPLETED.getCode();
        }
        if (STATUS_FAILED.equalsIgnoreCase(outboundStatus)) {
            return SseEventEnum.FAILED.getCode();
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

        private final String ystId;
        private final String sessionId;
        private final String sessionKey;
        private final SseEmitter emitter;
        private volatile ScheduledFuture<?> heartbeatFuture;

        private StreamSession(String ystId, String sessionKey, SseEmitter emitter) {
            this.ystId = ystId;
            this.sessionId = UUID.randomUUID().toString();
            this.sessionKey = sessionKey;
            this.emitter = emitter;
        }

        private String getYstId() {
            return ystId;
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
                currentHeartbeatFuture.cancel(true);
            }
        }

        private void complete() {
            cancelHeartbeat();
            emitter.complete();
        }
    }
}
