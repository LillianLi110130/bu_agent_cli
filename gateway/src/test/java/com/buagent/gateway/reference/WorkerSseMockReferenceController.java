package com.buagent.gateway.reference;

import com.fasterxml.jackson.annotation.JsonProperty;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.HashMap;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;

/**
 * Reference-only Java port of {@code cli/worker/mock_server.py}.
 *
 * <p>This class is intentionally placed under {@code src/test/java} so it does not get wired into
 * the production Spring Boot app by default. It is meant for backend teammates to copy or adapt
 * into the real Java gateway.
 *
 * <p>It mirrors the simplified SSE phase-1 semantics currently used by the Python worker:
 *
 * <ul>
 *   <li>single active SSE stream per {@code worker_id}</li>
 *   <li>server only pushes messages and accepts {@code complete}</li>
 *   <li>server does not track delivery state / inflight state</li>
 *   <li>{@code complete} is accepted by {@code worker_id} only</li>
 * </ul>
 */
@RestController
public class WorkerSseMockReferenceController {

    private final MockGatewayState state = new MockGatewayState();
    private final ExecutorService streamExecutor = Executors.newCachedThreadPool();

    @PostMapping("/api/worker/poll")
    public Map<String, Object> poll(@RequestBody WorkerRequest request) {
        state.markSeen(request.workerId);
        MockWorkerMessage message = state.dequeueMessage(request.workerId);

        Map<String, Object> response = new LinkedHashMap<String, Object>();
        List<Map<String, Object>> messages = new ArrayList<Map<String, Object>>();
        if (message != null) {
            messages.add(message.toMap());
        }
        response.put("messages", messages);
        return response;
    }

    @GetMapping(path = "/api/worker/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public SseEmitter stream(@RequestParam("worker_id") String workerId) {
        final int streamVersion = state.registerStream(workerId);
        final SseEmitter emitter = new SseEmitter(0L);

        streamExecutor.submit(new Runnable() {
            @Override
            public void run() {
                runStreamLoop(workerId, streamVersion, emitter);
            }
        });
        return emitter;
    }

    @PostMapping("/api/worker/online")
    public Map<String, Object> online(@RequestBody WorkerRequest request) {
        state.markOnline(request.workerId);
        return okResponse();
    }

    @PostMapping("/api/worker/offline")
    public Map<String, Object> offline(@RequestBody WorkerRequest request) {
        state.markOffline(request.workerId);
        return okResponse();
    }

    @PostMapping("/api/worker/complete")
    public Map<String, Object> complete(@RequestBody CompleteRequest request) {
        state.addCompletion(request.workerId, request.finalContent);
        return okResponse();
    }

    @PostMapping("/mock/messages")
    public ResponseEntity<Map<String, Object>> enqueue(@RequestBody EnqueueRequest request) {
        if (!state.isOnline(request.workerId)) {
            Map<String, Object> payload = new LinkedHashMap<String, Object>();
            payload.put("ok", false);
            payload.put("error", "no_online_worker");
            payload.put("worker_id", request.workerId);
            return ResponseEntity.status(HttpStatus.CONFLICT).body(payload);
        }

        MockWorkerMessage message = state.enqueueMessage(request.workerId, request.content);
        Map<String, Object> payload = okResponse();
        payload.put("worker_id", message.workerId);
        return ResponseEntity.ok(payload);
    }

    @GetMapping("/mock/messages")
    public Map<String, Object> listMessages() {
        Map<String, Object> payload = new LinkedHashMap<String, Object>();
        payload.put("messages", state.listQueuedMessages());
        return payload;
    }

    @GetMapping("/mock/completions")
    public Map<String, Object> listCompletions() {
        Map<String, Object> payload = new LinkedHashMap<String, Object>();
        payload.put("completions", state.listCompletions());
        return payload;
    }

    @GetMapping("/mock/online")
    public Map<String, Object> listOnline() {
        Map<String, Object> payload = new LinkedHashMap<String, Object>();
        payload.put("online_workers", state.listOnlineWorkers());
        return payload;
    }

    private void runStreamLoop(String workerId, int streamVersion, SseEmitter emitter) {
        try {
            sendEvent(emitter, "ready", singleEntryMap("worker_id", workerId));

            while (state.isOnline(workerId) && state.isCurrentStream(workerId, streamVersion)) {
                MockWorkerMessage message = state.dequeueMessage(workerId);
                if (message != null) {
                    state.markSeen(workerId);
                    sendEvent(emitter, "message", singleEntryMap("content", message.content));
                    continue;
                }

                boolean hasActivity = state.waitForStreamActivity(
                    workerId,
                    state.getStreamHeartbeatIntervalMillis()
                );

                if (!state.isCurrentStream(workerId, streamVersion)) {
                    Map<String, Object> errorPayload = new LinkedHashMap<String, Object>();
                    errorPayload.put("code", "replaced");
                    errorPayload.put("message", "connection replaced by newer stream");
                    sendEvent(emitter, "error", errorPayload);
                    break;
                }

                if (!state.isOnline(workerId)) {
                    break;
                }

                state.markSeen(workerId);
                if (!hasActivity) {
                    sendEvent(emitter, "heartbeat", singleEntryMap("ts", System.currentTimeMillis()));
                }
            }
            emitter.complete();
        } catch (Exception exception) {
            emitter.completeWithError(exception);
        }
    }

    private static void sendEvent(SseEmitter emitter, String eventName, Map<String, Object> payload)
        throws IOException {
        emitter.send(SseEmitter.event().name(eventName).data(payload));
    }

    private static Map<String, Object> okResponse() {
        Map<String, Object> payload = new LinkedHashMap<String, Object>();
        payload.put("ok", true);
        return payload;
    }

    private static Map<String, Object> singleEntryMap(String key, Object value) {
        Map<String, Object> payload = new LinkedHashMap<String, Object>();
        payload.put(key, value);
        return payload;
    }

    public static final class WorkerRequest {
        @JsonProperty("worker_id")
        public String workerId;
    }

    public static final class CompleteRequest {
        @JsonProperty("worker_id")
        public String workerId;

        @JsonProperty("final_content")
        public String finalContent;
    }

    public static final class EnqueueRequest {
        @JsonProperty("worker_id")
        public String workerId;

        @JsonProperty("content")
        public String content;
    }

    private static final class MockGatewayState {
        private final List<MockWorkerMessage> queuedMessages = new ArrayList<MockWorkerMessage>();
        private final List<Map<String, Object>> completions = new ArrayList<Map<String, Object>>();
        private final Map<String, OnlineWorkerRecord> onlineWorkers =
            new HashMap<String, OnlineWorkerRecord>();
        private final Map<String, Integer> streamVersions = new HashMap<String, Integer>();
        private final Map<String, Semaphore> streamSignals = new ConcurrentHashMap<String, Semaphore>();
        private final long workerTtlMillis = 30_000L;
        private final long streamHeartbeatIntervalMillis = 1_000L;

        public synchronized MockWorkerMessage enqueueMessage(String workerId, String content) {
            if (!isOnline(workerId)) {
                throw new IllegalStateException("no_online_worker:" + workerId);
            }
            MockWorkerMessage message = new MockWorkerMessage(workerId, content);
            queuedMessages.add(message);
            notifyStream(workerId);
            return message;
        }

        public synchronized void markOnline(String workerId) {
            onlineWorkers.put(workerId, new OnlineWorkerRecord(true, System.currentTimeMillis()));
            notifyStream(workerId);
        }

        public synchronized void markSeen(String workerId) {
            OnlineWorkerRecord record = onlineWorkers.get(workerId);
            if (record == null) {
                return;
            }
            record.lastSeenAt = System.currentTimeMillis();
        }

        public synchronized void markOffline(String workerId) {
            OnlineWorkerRecord record = onlineWorkers.get(workerId);
            if (record == null) {
                return;
            }
            record.online = false;
            record.lastSeenAt = System.currentTimeMillis();
            notifyStream(workerId);
        }

        public synchronized boolean isOnline(String workerId) {
            OnlineWorkerRecord record = onlineWorkers.get(workerId);
            if (record == null || !record.online) {
                return false;
            }
            if (System.currentTimeMillis() - record.lastSeenAt > workerTtlMillis) {
                record.online = false;
                return false;
            }
            return true;
        }

        public synchronized int registerStream(String workerId) {
            int nextVersion = 1;
            Integer currentVersion = streamVersions.get(workerId);
            if (currentVersion != null) {
                nextVersion = currentVersion + 1;
            }
            streamVersions.put(workerId, nextVersion);
            markSeen(workerId);
            notifyStream(workerId);
            return nextVersion;
        }

        public synchronized boolean isCurrentStream(String workerId, int version) {
            Integer currentVersion = streamVersions.get(workerId);
            return currentVersion != null && currentVersion == version;
        }

        public synchronized MockWorkerMessage dequeueMessage(String workerId) {
            for (int index = 0; index < queuedMessages.size(); index++) {
                MockWorkerMessage message = queuedMessages.get(index);
                if (!workerId.equals(message.workerId)) {
                    continue;
                }
                queuedMessages.remove(index);
                return message;
            }
            return null;
        }

        public void notifyStream(String workerId) {
            getStreamSignal(workerId).release();
        }

        public boolean waitForStreamActivity(String workerId, long timeoutMillis) {
            Semaphore signal = getStreamSignal(workerId);
            signal.drainPermits();
            try {
                return signal.tryAcquire(timeoutMillis, TimeUnit.MILLISECONDS);
            } catch (InterruptedException interruptedException) {
                Thread.currentThread().interrupt();
                return false;
            }
        }

        public synchronized void addCompletion(String workerId, String finalContent) {
            Map<String, Object> payload = new LinkedHashMap<String, Object>();
            payload.put("worker_id", workerId);
            payload.put("final_content", finalContent);
            payload.put("completed_at", System.currentTimeMillis());
            completions.add(payload);
        }

        public synchronized List<Map<String, Object>> listQueuedMessages() {
            List<Map<String, Object>> snapshots = new ArrayList<Map<String, Object>>();
            for (MockWorkerMessage queuedMessage : queuedMessages) {
                snapshots.add(queuedMessage.toMap());
            }
            return snapshots;
        }

        public synchronized List<Map<String, Object>> listCompletions() {
            return new ArrayList<Map<String, Object>>(completions);
        }

        public synchronized Map<String, Map<String, Object>> listOnlineWorkers() {
            Map<String, Map<String, Object>> snapshots = new LinkedHashMap<String, Map<String, Object>>();
            for (Map.Entry<String, OnlineWorkerRecord> entry : onlineWorkers.entrySet()) {
                if (!isOnline(entry.getKey())) {
                    continue;
                }
                Map<String, Object> recordPayload = new LinkedHashMap<String, Object>();
                recordPayload.put("online", entry.getValue().online);
                recordPayload.put("last_seen_at", entry.getValue().lastSeenAt);
                snapshots.put(entry.getKey(), recordPayload);
            }
            return snapshots;
        }

        public long getStreamHeartbeatIntervalMillis() {
            return streamHeartbeatIntervalMillis;
        }

        private Semaphore getStreamSignal(String workerId) {
            Semaphore existingSignal = streamSignals.get(workerId);
            if (existingSignal != null) {
                return existingSignal;
            }

            Semaphore newSignal = new Semaphore(0);
            Semaphore previousSignal = streamSignals.putIfAbsent(workerId, newSignal);
            return previousSignal != null ? previousSignal : newSignal;
        }
    }

    private static final class MockWorkerMessage {
        private final String workerId;
        private final String content;

        private MockWorkerMessage(String workerId, String content) {
            this.workerId = workerId;
            this.content = content;
        }

        private Map<String, Object> toMap() {
            Map<String, Object> payload = new LinkedHashMap<String, Object>();
            payload.put("worker_id", workerId);
            payload.put("content", content);
            return payload;
        }
    }

    private static final class OnlineWorkerRecord {
        private boolean online;
        private long lastSeenAt;

        private OnlineWorkerRecord(boolean online, long lastSeenAt) {
            this.online = online;
            this.lastSeenAt = lastSeenAt;
        }
    }
}
