// SSE 流式请求封装
import type { StreamEvent, QueryRequest } from '@/types';

// API 基础路径：开发环境通过 Vite 代理转发，生产环境通过 nginx 转发
const BASE_URL = '/api';

/**
 * 流式对话生成器
 * 使用 Fetch API + ReadableStream 处理 SSE 流式数据
 *
 * @param message 用户消息
 * @param sessionId 会话 ID（可选）
 * @yields SSE 事件
 */
export async function* streamChat(
  message: string,
  sessionId?: string
): AsyncGenerator<StreamEvent, void, unknown> {
  const requestBody: QueryRequest = {
    message,
    session_id: sessionId,
  };

  const response = await fetch(`${BASE_URL}/agent/query-stream-delta`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(requestBody),
  });

  // 检查响应状态
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  // 检查响应体是否可读
  if (!response.body) {
    throw new Error('Response body is null');
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();

      if (done) {
        break;
      }

      // 解码当前块并添加到缓冲区
      buffer += decoder.decode(value, { stream: true });

      // 按行分割处理
      const lines = buffer.split('\n');
      // 保留最后一行（可能不完整）
      buffer = lines.pop() || '';

      for (const line of lines) {
        const trimmedLine = line.trim();

        // 跳过空行和注释行
        if (!trimmedLine || trimmedLine.startsWith(':')) {
          // 检查是否是结束标记
          if (trimmedLine === ':done') {
            return;
          }
          continue;
        }

        // 解析 data: 前缀的行
        if (trimmedLine.startsWith('data: ')) {
          try {
            const eventData = JSON.parse(trimmedLine.slice(6));
            yield eventData as StreamEvent;
          } catch (e) {
            console.warn('[Stream] Failed to parse event:', trimmedLine, e);
          }
        }
      }
    }
  } finally {
    // 释放 reader
    reader.releaseLock();
  }
}

/**
 * 流式对话（带回调函数版本）
 * 适用于不使用生成器的场景
 *
 * @param message 用户消息
 * @param sessionId 会话 ID（可选）
 * @param callbacks 回调函数集合
 */
export interface StreamCallbacks {
  onData?: (event: StreamEvent) => void;
  onError?: (error: Error) => void;
  onComplete?: () => void;
}

export async function streamChatWithCallbacks(
  message: string,
  sessionId: string | undefined,
  callbacks: StreamCallbacks
): Promise<void> {
  const { onData, onError, onComplete } = callbacks;

  try {
    const generator = streamChat(message, sessionId);

    for await (const event of generator) {
      if (onData) {
        onData(event);
      }
    }

    if (onComplete) {
      onComplete();
    }
  } catch (error) {
    if (onError) {
      onError(error as Error);
    } else {
      console.error('[Stream] Error:', error);
    }
  }
}
