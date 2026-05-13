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

function upsertMessage(
  currentMessages: ConversationMessage[],
  nextMessage: ConversationMessage
) {
  const index = currentMessages.findIndex((item) => item.id === nextMessage.id);
  if (index < 0) {
    return [...currentMessages, nextMessage];
  }
  const nextMessages = [...currentMessages];
  nextMessages[index] = nextMessage;
  return nextMessages;
}

function appendProgressContent(existingContent: string, nextChunk: string) {
  const trimmedChunk = nextChunk.trim();
  if (!trimmedChunk) {
    return existingContent;
  }
  if (!existingContent.trim()) {
    return trimmedChunk;
  }
  if (existingContent.includes(trimmedChunk)) {
    return existingContent;
  }
  return `${existingContent}\n\n${trimmedChunk}`;
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
  const activeRunKeyRef = useRef<string | null>(null);

  const refreshWorkerSummary = useCallback(async () => {
    try {
      const summary = await fetchWorkerSummary(viewState.workerId);
      startTransition(() => {
        setWorkerSummary(summary);
      });
      setViewState((current) => ({
        ...current,
        lastError:
          current.lastError === '当前无法连接 Python server，请先确认服务端已经启动。'
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
        lastError: current.lastError ?? '当前无法连接 Python server，请先确认服务端已经启动。'
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
      streamAbortRef.current?.abort();
    };
  }, []);

  const createSession = useCallback(() => {
    const nextSessionId = createClientId('web-session');
    setSessionIndex((current) => current + 1);
    setSessionId(nextSessionId);
    setMessages([]);
    setDraft('');
    streamAbortRef.current?.abort();
    streamAbortRef.current = null;
    activeRunKeyRef.current = null;
    setViewState((current) => ({
      ...current,
      activeSessionId: nextSessionId,
      submitStatus: 'idle',
      lastError: undefined
    }));
  }, []);

  const stopCurrentStream = useCallback(() => {
    const activeController = streamAbortRef.current;
    if (activeController) {
      activeController.abort();
      streamAbortRef.current = null;
    }
    activeRunKeyRef.current = null;

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
  }, []);

  const launchLocalCrab = useCallback(() => {
    setIsAwaitingLocalLaunch(true);
    void refreshWorkerSummary();
    void antdMessage.info('如果浏览器弹出确认，请允许打开本地应用。');
    window.location.href = 'crab://open';
  }, [refreshWorkerSummary]);

  const handleStreamEvent = useCallback((requestEvent: RequestEvent) => {
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
      setMessages((current) => {
        const existingMessage = current.find(
          (item) => item.id === `assistant-${messageKey}` && item.role === 'assistant'
        );
        const nextContent = appendProgressContent(existingMessage?.content ?? '', nextChunk);
        return upsertMessage(
          current,
          buildAssistantMessage(messageKey, nextContent, eventTime, 'processing')
        );
      });
      return;
    }

    if (requestEvent.type === 'completed') {
      const finalContent = requestEvent.finalContent ?? '';
      activeRunKeyRef.current = null;
      setViewState((current) => ({
        ...current,
        submitStatus: 'completed',
        lastError: undefined
      }));
      setWorkerSummary((current) => ({
        ...current,
        lastCompletedAt: requestEvent.finishedAt ?? eventTime
      }));
      setMessages((current) =>
        upsertMessage(current, buildAssistantMessage(messageKey, finalContent, eventTime))
      );
      return;
    }

    const errorMessage = requestEvent.errorMessage ?? '处理失败，请稍后重试。';
    activeRunKeyRef.current = null;
    setViewState((current) => ({
      ...current,
      submitStatus: 'failed',
      lastError: errorMessage
    }));
    setMessages((current) =>
      appendMessageIfMissing(current, buildErrorMessage(messageKey, errorMessage, eventTime))
    );
    void antdMessage.error(errorMessage);
  }, []);

  const submitCurrentDraft = useCallback(
    async (overrideContent?: string) => {
      const rawContent = overrideContent ?? draft;
      const trimmedContent = rawContent.trim();
      if (!trimmedContent || isSubmitting || viewState.submitStatus === 'processing') {
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

      setIsSubmitting(true);
      setDraft('');
      setMessages((current) => [...current, optimisticUserMessage]);
      setViewState((current) => ({
        ...current,
        submitStatus: 'submitting',
        lastError: undefined
      }));

      let abortController: AbortController | null = null;
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

        streamAbortRef.current?.abort();
        abortController = new AbortController();
        streamAbortRef.current = abortController;

        await streamWorkerEvents(viewState.workerId, {
          signal: abortController.signal,
          onEvent: handleStreamEvent
        });
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
        if (abortController !== null && streamAbortRef.current === abortController) {
          streamAbortRef.current = null;
        }
        setIsSubmitting(false);
      }
    },
    [draft, handleStreamEvent, isSubmitting, sessionId, viewState.submitStatus, viewState.workerId]
  );

  return {
    canLaunchLocalCrab: workerSummary.isOnline === false,
    createSession,
    draft,
    emptyStateSuggestions: EMPTY_STATE_SUGGESTIONS,
    canStopStream:
      viewState.submitStatus === 'submitted' || viewState.submitStatus === 'processing',
    isComposerDisabled:
      isSubmitting ||
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
