import axios, { AxiosHeaders } from 'axios';

import type {
  RequestEvent,
  SubmitMessagePayload,
  SubmitMessageResponse,
  WorkerSummary
} from '../types';

const USE_MOCK = import.meta.env.VITE_REMOTE_CONSOLE_USE_MOCK === 'true';
const API_BASE_URL = import.meta.env.VITE_REMOTE_CONSOLE_API_BASE_URL ?? '';
const AUTHORIZATION =
  import.meta.env.VITE_REMOTE_CONSOLE_AUTHORIZATION?.trim() ||
  (import.meta.env.VITE_REMOTE_CONSOLE_AUTH_TOKEN?.trim()
    ? `Bearer ${import.meta.env.VITE_REMOTE_CONSOLE_AUTH_TOKEN.trim()}`
    : '');

const apiClient = axios.create({
  baseURL: API_BASE_URL || undefined,
  timeout: 60_000
});

apiClient.interceptors.request.use((config) => {
  const headers = AxiosHeaders.from(config.headers ?? {});
  if (AUTHORIZATION && !headers.has('Authorization')) {
    headers.set('Authorization', AUTHORIZATION);
  }
  config.headers = headers;
  return config;
});

type MockRequestRecord = {
  requestId: string;
  workerId: string;
  content: string;
};

const mockRequestStore = new Map<string, MockRequestRecord>();

function buildUrl(path: string) {
  if (!API_BASE_URL) {
    return path;
  }
  return `${API_BASE_URL}${path}`;
}

function buildMockAnswer(content: string) {
  return [
    '我已经按当前 Web 远程对话链路帮你整理了一版结果。',
    '',
    `原始问题：${content}`
  ].join('\n');
}

function sleep(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function buildSseHeaders() {
  const headers: Record<string, string> = {
    Accept: 'text/event-stream'
  };
  if (AUTHORIZATION) {
    headers.Authorization = AUTHORIZATION;
  }
  return headers;
}

function parseSseChunk(rawChunk: string): RequestEvent | null {
  const trimmedChunk = rawChunk.replace(/\r/g, '').trim();
  if (!trimmedChunk || trimmedChunk.startsWith(':')) {
    return null;
  }

  const dataLines = trimmedChunk
    .split('\n')
    .filter((line) => line.startsWith('data:'))
    .map((line) => line.slice(5).trim());
  if (dataLines.length === 0) {
    return null;
  }

  return JSON.parse(dataLines.join('\n')) as RequestEvent;
}

export async function fetchWorkerSummary(workerId: string): Promise<WorkerSummary> {
  if (USE_MOCK) {
    return {
      workerId,
      isOnline: true,
      lastCompletedAt: new Date(Date.now() - 180_000).toISOString()
    };
  }

  const response = await apiClient.get<WorkerSummary>(`/web-console/workers/${workerId}`);
  return response.data;
}

export async function submitMessage(
  payload: SubmitMessagePayload
): Promise<SubmitMessageResponse> {
  if (USE_MOCK) {
    const acceptedAt = new Date().toISOString();
    const requestId = `mock-request-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
    mockRequestStore.set(payload.workerId, {
      requestId,
      workerId: payload.workerId,
      content: payload.content
    });
    return {
      ok: true,
      acceptedAt,
      requestId
    };
  }

  const response = await apiClient.post<SubmitMessageResponse>('/web-console/messages', payload);
  return response.data;
}

export async function streamWorkerEvents(
  workerId: string,
  {
    onOpen,
    onEvent,
    signal
  }: {
    onOpen?: () => void;
    onEvent: (event: RequestEvent) => void;
    signal?: AbortSignal;
  }
) {
  if (USE_MOCK) {
    onOpen?.();
    let lastDeliveredRequestId = '';

    while (!signal?.aborted) {
      const record = mockRequestStore.get(workerId);
      if (!record || record.requestId === lastDeliveredRequestId) {
        await sleep(300);
        continue;
      }

      lastDeliveredRequestId = record.requestId;

      onEvent({
        type: 'submitted',
        workerId,
        requestId: record.requestId,
        ts: new Date().toISOString()
      });

      await sleep(220);
      if (signal?.aborted) {
        return;
      }

      onEvent({
        type: 'processing',
        workerId,
        requestId: record.requestId,
        ts: new Date().toISOString()
      });

      await sleep(500);
      if (signal?.aborted) {
        return;
      }

      onEvent({
        type: 'progress',
        workerId,
        requestId: record.requestId,
        content: '正在整理相关上下文，请稍候。',
        ts: new Date().toISOString()
      });

      await sleep(450);
      if (signal?.aborted) {
        return;
      }

      onEvent({
        type: 'completed',
        workerId,
        requestId: record.requestId,
        finalContent: buildMockAnswer(record.content),
        finishedAt: new Date().toISOString()
      });
    }
    return;
  }

  const response = await fetch(buildUrl(`/web-console/workers/${workerId}/events`), {
    method: 'GET',
    headers: buildSseHeaders(),
    signal
  });

  if (!response.ok || !response.body) {
    throw new Error(`连接消息事件流失败，状态码：${response.status}`);
  }

  onOpen?.();

  const reader = response.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });
    let boundary = buffer.indexOf('\n\n');

    while (boundary >= 0) {
      const rawChunk = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);

      const event = parseSseChunk(rawChunk);
      if (event) {
        onEvent(event);
      }

      boundary = buffer.indexOf('\n\n');
    }
  }
}
