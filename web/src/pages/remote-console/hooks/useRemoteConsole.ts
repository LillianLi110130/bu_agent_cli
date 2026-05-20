import { message as antdMessage } from 'antd';
import { startTransition, useCallback, useEffect, useRef, useState } from 'react';

import { DEFAULT_WORKER_ID, EMPTY_STATE_SUGGESTIONS } from '../constants';
import {
  fetchWorkerSummary,
  streamWorkerEvents,
  submitMessage
} from '../services/remoteConsoleService';
import type {
  ConversationMessage,
  RemoteConsoleViewState,
  RequestEvent,
  SubmitStatus,
  WorkerSummary
} from '../types';

const STREAM_ROTATE_INTERVAL_MS = 10 * 1000;
const STREAM_RECONNECT_DELAYS_MS = [1_000, 2_000, 5_000, 10_000];
const MAX_STREAM_RECONNECT_ATTEMPTS = 8;

function isAbortError(error: unknown) {
  return error instanceof DOMException
    ? error.name === 'AbortError'
    : error instanceof Error && error.name === 'AbortError';
}

function createClientId(prefix: string) {
  const randomUuid = globalThis.crypto?.randomUUID?.();
  if (randomUuid) {
    return `${prefix}-${randomUuid}`;
  }

  const fallbackRandom = Math.random().toString(36).slice(2, 10);
  return `${prefix}-${Date.now().toString(36)}-${fallbackRandom}`;
}

function buildSessionTitle(index: number) {
  return index === 0 ? '新对话' : `新对话 ${index + 1}`;
}

function buildSystemMessage(
  messageKey: string,
  status: SubmitStatus,
  content: string,
  createdAt: string
): ConversationMessage {
  return {
    id: `${status}-${messageKey}`,
    role: 'system',
    content,
    createdAt,
    status
  };
}

function buildAssistantMessage(
  messageKey: string,
  content: string,
  createdAt: string,
  status: SubmitStatus = 'completed'
): ConversationMessage {
  return {
    id: `assistant-${messageKey}`,
    role: 'assistant',
    content,
    createdAt,
    status
  };
}

function buildErrorMessage(
  messageKey: string,
  content: string,
  createdAt: string
): ConversationMessage {
  return {
    id: `error-${messageKey}`,
    role: 'error',
    content,
    createdAt,
    status: 'failed'
  };
}

function appendMessageIfMissing(
  currentMessages: ConversationMessage[],
  nextMessage: ConversationMessage
) {
  if (currentMessages.some((item) => item.id === nextMessage.id)) {
    return currentMessages;
  }
  return [...currentMessages, nextMessage];
}

function buildRequestEventSignature(requestEvent: RequestEvent) {
  return JSON.stringify({
    type: requestEvent.type,
    workerId: requestEvent.workerId,
    requestId: requestEvent.requestId ?? "",
    ts: requestEvent.ts ?? "",
    content: requestEvent.content ?? "",
    finalContent: requestEvent.finalContent ?? "",
    errorMessage: requestEvent.errorMessage ?? "",
    finishedAt: requestEvent.finishedAt ?? ""
  });
}

export function useRemoteConsole() {
  const [messages, setMessages] = useState<ConversationMessage[]>([]);
  const [draft, setDraft] = useState('');
  const [sessionIndex, setSessionIndex] = useState(0);
  const [sessionId, setSessionId] = useState(() => createClientId('web-session'));
  const [workerSummary, setWorkerSummary] = useState<WorkerSummary>({
    workerId: DEFAULT_WORKER_ID,
    isOnline: undefined
  });
  const [viewState, setViewState] = useState<RemoteConsoleViewState>({
    workerId: DEFAULT_WORKER_ID,
    activeSessionId: sessionId,
    submitStatus: 'idle',
    useMock: import.meta.env.VITE_REMOTE_CONSOLE_USE_MOCK === 'true'
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isAwaitingLocalLaunch, setIsAwaitingLocalLaunch] = useState(false);

  const streamAbortRef = useRef<AbortController | null>(null);
  const streamRotateTimerRef = useRef<number | null>(null);
  const reconnectTimerRef = useRef<number | null>(null);
  const activeRunKeyRef = useRef<string | null>(null);
  const activeStreamTokenRef = useRef<string | null>(null);
  const reconnectAttemptRef = useRef(0);
  const streamTerminalRef = useRef(false);
  const intentionalAbortRef = useRef(false);
  const stopRequestedRef = useRef(false);
  const seenEventSignaturesRef = useRef<Set<string>>(new Set());

  const clearScheduledStreamWork = useCallback(() => {
    if (streamRotateTimerRef.current !== null) {
      window.clearTimeout(streamRotateTimerRef.current);
      streamRotateTimerRef.current = null;
    }

    if (reconnectTimerRef.current !== null) {
      window.clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
  }, []);

  const abortActiveStream = useCallback(
    (markIntentionalAbort: boolean) => {
      if (markIntentionalAbort) {
        intentionalAbortRef.current = true;
      }

      if (streamAbortRef.current) {
        streamAbortRef.current.abort();
        streamAbortRef.current = null;
      }
    },
    []
  );

  const cleanupActiveStream = useCallback(
    (markIntentionalAbort: boolean) => {
      clearScheduledStreamWork();
      abortActiveStream(markIntentionalAbort);
      activeStreamTokenRef.current = null;
    },
    [abortActiveStream, clearScheduledStreamWork]
  );

  const refreshWorkerSummary = useCallback(async () => {
    try {
      const summary = await fetchWorkerSummary(viewState.workerId);
      startTransition(() => {
        setWorkerSummary(summary);
      });
      setViewState((current) => ({
        ...current,
        lastError:
          current.lastError === '当前无法连接 Python 服务，请先确认服务端已经启动。'
            ? undefined
            : current.lastError
      }));
    } catch {
      startTransition(() => {
        setWorkerSummary({
          workerId: viewState.workerId,
          isOnline: false
        });
      });
      setViewState((current) => ({
        ...current,
        lastError: current.lastError ?? '当前无法连接 Python 服务，请先确认服务端已经启动。'
      }));
    }
  }, [viewState.workerId]);

  useEffect(() => {
    void refreshWorkerSummary();
  }, [refreshWorkerSummary]);

  useEffect(() => {
    if (workerSummary.isOnline === true && isAwaitingLocalLaunch) {
      setIsAwaitingLocalLaunch(false);
      void antdMessage.success('已检测到本地 Crab 在线。');
    }
  }, [isAwaitingLocalLaunch, workerSummary.isOnline]);

  useEffect(() => {
    const pollingIntervalMs =
      workerSummary.isOnline === true && !isAwaitingLocalLaunch ? 20000 : 1500;

    const timer = window.setInterval(() => {
      void refreshWorkerSummary();
    }, pollingIntervalMs);

    return () => {
      window.clearInterval(timer);
    };
  }, [isAwaitingLocalLaunch, refreshWorkerSummary, workerSummary.isOnline]);

  useEffect(() => {
    return () => {
      stopRequestedRef.current = true;
      cleanupActiveStream(true);
    };
  }, [cleanupActiveStream]);

  const createSession = useCallback(() => {
    const nextSessionId = createClientId('web-session');
    stopRequestedRef.current = true;
    streamTerminalRef.current = false;
    reconnectAttemptRef.current = 0;
    setSessionIndex((current) => current + 1);
    setSessionId(nextSessionId);
    setMessages([]);
    setDraft('');
    activeRunKeyRef.current = null;
    cleanupActiveStream(true);
    setViewState((current) => ({
      ...current,
      activeSessionId: nextSessionId,
      submitStatus: 'idle',
      lastError: undefined
    }));
    seenEventSignaturesRef.current = new Set();
  }, [cleanupActiveStream]);

  const stopCurrentStream = useCallback(() => {
    stopRequestedRef.current = true;
    streamTerminalRef.current = true;
    reconnectAttemptRef.current = 0;
    activeRunKeyRef.current = null;
    cleanupActiveStream(true);

    setViewState((current) => ({
      ...current,
      submitStatus: 'idle',
      lastError: undefined
    }));
    setMessages((current) => [
      ...current,
      buildSystemMessage(
        createClientId('stream-stop'),
        'idle',
        '已停止当前页面接收。本地任务可能仍在继续执行。',
        new Date().toISOString()
      )
    ]);
    seenEventSignaturesRef.current = new Set();
  }, [cleanupActiveStream]);

  const launchLocalCrab = useCallback(() => {
    setIsAwaitingLocalLaunch(true);
    void refreshWorkerSummary();
    void antdMessage.info('如果浏览器弹出确认，请允许打开本地应用。');
    window.location.href = 'crab://open';
  }, [refreshWorkerSummary]);

  const handleStreamEvent = useCallback(
    (requestEvent: RequestEvent) => {
      const eventSignature = buildRequestEventSignature(requestEvent);
      if (seenEventSignaturesRef.current.has(eventSignature)) {
        return;
      }
      seenEventSignaturesRef.current.add(eventSignature);

      const eventTime = requestEvent.finishedAt ?? requestEvent.ts ?? new Date().toISOString();
      const messageKey = activeRunKeyRef.current ?? `${requestEvent.workerId}-current`;

      if (requestEvent.type === 'submitted') {
        setViewState((current) => ({
          ...current,
          submitStatus: 'submitted',
          lastError: undefined
        }));
        setMessages((current) =>
          appendMessageIfMissing(
            current,
            buildSystemMessage(messageKey, 'submitted', '消息已提交，等待本地终端接收。', eventTime)
          )
        );
        return;
      }

      if (requestEvent.type === 'processing') {
        setViewState((current) => ({
          ...current,
          submitStatus: 'processing',
          lastError: undefined
        }));
        setMessages((current) =>
          appendMessageIfMissing(
            current,
            buildSystemMessage(messageKey, 'processing', '本地终端已开始处理。', eventTime)
          )
        );
        return;
      }

      if (requestEvent.type === 'progress') {
        const nextChunk = requestEvent.content?.trim() ?? '';
        if (!nextChunk) {
          return;
        }

        setViewState((current) => ({
          ...current,
          submitStatus: 'processing',
          lastError: undefined
        }));
        setMessages((current) => [
          ...current,
          buildAssistantMessage(
            `${messageKey}-progress-${createClientId('event')}`,
            nextChunk,
            eventTime,
            'processing'
          )
        ]);
        return;
      }

      if (requestEvent.type === 'completed') {
        const finalContent = requestEvent.finalContent ?? '';
        streamTerminalRef.current = true;
        reconnectAttemptRef.current = 0;
        activeRunKeyRef.current = null;
        activeStreamTokenRef.current = null;
        clearScheduledStreamWork();
        abortActiveStream(true);
        setViewState((current) => ({
          ...current,
          submitStatus: 'completed',
          lastError: undefined
        }));
        setWorkerSummary((current) => ({
          ...current,
          lastCompletedAt: requestEvent.finishedAt ?? eventTime
        }));
        setMessages((current) => [
          ...current,
          buildAssistantMessage(
            `${messageKey}-completed-${createClientId('event')}`,
            finalContent,
            eventTime
          )
        ]);
        return;
      }

      const errorMessage = requestEvent.errorMessage ?? '处理失败，请稍后重试。';
      streamTerminalRef.current = true;
      reconnectAttemptRef.current = 0;
      activeRunKeyRef.current = null;
      activeStreamTokenRef.current = null;
      clearScheduledStreamWork();
      abortActiveStream(true);
      setViewState((current) => ({
        ...current,
        submitStatus: 'failed',
        lastError: errorMessage
      }));
      setMessages((current) =>
        appendMessageIfMissing(current, buildErrorMessage(messageKey, errorMessage, eventTime))
      );
      void antdMessage.error(errorMessage);
    },
    [abortActiveStream, clearScheduledStreamWork]
  );

  const startEventStreamSession = useRef<
    ((mode?: 'initial' | 'reconnect') => Promise<void>) | null
  >(null);

  const scheduleReconnect = useCallback(
    (streamToken: string, delayMs: number) => {
      if (stopRequestedRef.current || streamTerminalRef.current || !activeRunKeyRef.current) {
        return;
      }

      clearScheduledStreamWork();
      reconnectTimerRef.current = window.setTimeout(() => {
        if (
          activeStreamTokenRef.current !== streamToken ||
          stopRequestedRef.current ||
          streamTerminalRef.current ||
          !activeRunKeyRef.current
        ) {
          return;
        }

        void startEventStreamSession.current?.('reconnect');
      }, delayMs);
    },
    [clearScheduledStreamWork]
  );

  const handleStreamDisconnect = useCallback(
    (streamToken: string, error?: unknown) => {
      if (
        activeStreamTokenRef.current !== streamToken ||
        stopRequestedRef.current ||
        streamTerminalRef.current ||
        !activeRunKeyRef.current
      ) {
        return;
      }

      const nextAttempt = reconnectAttemptRef.current + 1;
      reconnectAttemptRef.current = nextAttempt;

      if (nextAttempt > MAX_STREAM_RECONNECT_ATTEMPTS) {
        const errorMessage =
          error instanceof Error ? error.message : '事件流重连失败，请稍后重试。';
        activeRunKeyRef.current = null;
        streamTerminalRef.current = true;
        clearScheduledStreamWork();
        setViewState((current) => ({
          ...current,
          submitStatus: 'failed',
          lastError: errorMessage
        }));
        setMessages((current) =>
          appendMessageIfMissing(
            current,
            buildErrorMessage(
              createClientId('stream-reconnect-failed'),
              errorMessage,
              new Date().toISOString()
            )
          )
        );
        void antdMessage.error(errorMessage);
        return;
      }

      setViewState((current) => ({
        ...current,
        submitStatus: 'reconnecting',
        lastError: undefined
      }));

      const reconnectDelayMs =
        intentionalAbortRef.current
          ? 0
          : STREAM_RECONNECT_DELAYS_MS[
              Math.min(nextAttempt - 1, STREAM_RECONNECT_DELAYS_MS.length - 1)
            ];

      intentionalAbortRef.current = false;
      scheduleReconnect(streamToken, reconnectDelayMs);
    },
    [clearScheduledStreamWork, scheduleReconnect]
  );

  const scheduleStreamRotation = useCallback(
    (streamToken: string) => {
      if (stopRequestedRef.current || streamTerminalRef.current || !activeRunKeyRef.current) {
        return;
      }

      if (streamRotateTimerRef.current !== null) {
        window.clearTimeout(streamRotateTimerRef.current);
      }

      streamRotateTimerRef.current = window.setTimeout(() => {
        if (
          activeStreamTokenRef.current !== streamToken ||
          stopRequestedRef.current ||
          streamTerminalRef.current ||
          !activeRunKeyRef.current
        ) {
          return;
        }

        setViewState((current) => ({
          ...current,
          submitStatus: 'reconnecting',
          lastError: undefined
        }));
        abortActiveStream(true);
      }, STREAM_ROTATE_INTERVAL_MS);
    },
    [abortActiveStream]
  );

  const startEventStreamSessionImpl = useCallback(
    async (mode: 'initial' | 'reconnect' = 'initial') => {
      if (stopRequestedRef.current || streamTerminalRef.current || !activeRunKeyRef.current) {
        return;
      }

      clearScheduledStreamWork();
      abortActiveStream(false);

      const streamToken = createClientId('stream-session');
      const abortController = new AbortController();
      activeStreamTokenRef.current = streamToken;
      streamAbortRef.current = abortController;
      intentionalAbortRef.current = false;

      if (mode === 'reconnect') {
        setViewState((current) => ({
          ...current,
          submitStatus: 'reconnecting',
          lastError: undefined
        }));
      }

      try {
        await streamWorkerEvents(viewState.workerId, {
          signal: abortController.signal,
          onOpen: () => {
            if (activeStreamTokenRef.current !== streamToken || streamTerminalRef.current) {
              return;
            }

            reconnectAttemptRef.current = 0;
            setViewState((current) => ({
              ...current,
              submitStatus:
                current.submitStatus === 'reconnecting' ? 'processing' : current.submitStatus,
              lastError: undefined
            }));
            scheduleStreamRotation(streamToken);
          },
          onEvent: (event) => {
            if (activeStreamTokenRef.current !== streamToken || streamTerminalRef.current) {
              return;
            }
            handleStreamEvent(event);
          }
        });

        if (streamAbortRef.current === abortController) {
          streamAbortRef.current = null;
        }

        if (streamTerminalRef.current || stopRequestedRef.current) {
          return;
        }

        handleStreamDisconnect(streamToken);
      } catch (error) {
        if (streamAbortRef.current === abortController) {
          streamAbortRef.current = null;
        }

        if (streamTerminalRef.current || stopRequestedRef.current) {
          return;
        }

        if (isAbortError(error)) {
          handleStreamDisconnect(streamToken);
          return;
        }

        handleStreamDisconnect(streamToken, error);
      }
    },
    [
      abortActiveStream,
      clearScheduledStreamWork,
      handleStreamDisconnect,
      handleStreamEvent,
      scheduleStreamRotation,
      viewState.workerId
    ]
  );

  startEventStreamSession.current = startEventStreamSessionImpl;

  const submitCurrentDraft = useCallback(
    async (overrideContent?: string) => {
      const rawContent = overrideContent ?? draft;
      const trimmedContent = rawContent.trim();
      if (
        !trimmedContent ||
        isSubmitting ||
        viewState.submitStatus === 'processing' ||
        viewState.submitStatus === 'submitted' ||
        viewState.submitStatus === 'reconnecting'
      ) {
        return;
      }

      const createdAt = new Date().toISOString();
      const optimisticUserMessage: ConversationMessage = {
        id: createClientId('user'),
        role: 'user',
        content: trimmedContent,
        createdAt
      };
      const runKey = createClientId('run');

      stopRequestedRef.current = false;
      streamTerminalRef.current = false;
      reconnectAttemptRef.current = 0;
      clearScheduledStreamWork();
      abortActiveStream(false);
      activeStreamTokenRef.current = null;
      seenEventSignaturesRef.current = new Set();
      setIsSubmitting(true);
      setDraft('');
      setMessages((current) => [...current, optimisticUserMessage]);
      setViewState((current) => ({
        ...current,
        submitStatus: 'submitting',
        lastError: undefined
      }));

      activeRunKeyRef.current = runKey;

      try {
        await submitMessage({
          workerId: viewState.workerId,
          sessionId,
          content: trimmedContent
        });

        setViewState((current) => ({
          ...current,
          submitStatus: 'submitted'
        }));

        void startEventStreamSession.current?.('initial');
      } catch (error) {
        if (isAbortError(error)) {
          return;
        }

        const errorMessage = error instanceof Error ? error.message : '提交失败，请稍后重试。';
        activeRunKeyRef.current = null;
        setViewState((current) => ({
          ...current,
          submitStatus: 'failed',
          lastError: errorMessage
        }));
        setMessages((current) => [
          ...current,
          buildErrorMessage(runKey, errorMessage, new Date().toISOString())
        ]);
        void antdMessage.error(errorMessage);
      } finally {
        setIsSubmitting(false);
      }
    },
    [
      abortActiveStream,
      clearScheduledStreamWork,
      draft,
      isSubmitting,
      sessionId,
      viewState.submitStatus,
      viewState.workerId
    ]
  );

  return {
    canLaunchLocalCrab: workerSummary.isOnline === false,
    createSession,
    draft,
    emptyStateSuggestions: EMPTY_STATE_SUGGESTIONS,
    canStopStream:
      viewState.submitStatus === 'submitted' ||
      viewState.submitStatus === 'processing' ||
      viewState.submitStatus === 'reconnecting',
    isComposerDisabled:
      isSubmitting ||
      viewState.submitStatus === 'reconnecting' ||
      viewState.submitStatus === 'submitted' ||
      viewState.submitStatus === 'processing',
    isSubmitting,
    launchLocalCrab,
    messages,
    sessionTitle: buildSessionTitle(sessionIndex),
    setDraft,
    stopCurrentStream,
    submitCurrentDraft,
    viewState,
    workerSummary
  };
}
