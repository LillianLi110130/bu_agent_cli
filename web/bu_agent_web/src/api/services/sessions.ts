/**
 * 会话管理 API
 * 使用 swagger-typescript-api 生成的类型定义
 * 使用统一的 request.ts 封装的请求函数
 */
import request from '../request';
import type {
  SessionInfoResponse,
  SessionCreateResponse,
  ClearHistoryResponse,
} from '../types/data-contracts';

/**
 * 创建新会话
 */
export async function createSession(): Promise<SessionCreateResponse> {
  return request<SessionCreateResponse>({
    method: 'POST',
    url: '/sessions',
  });
}

/**
 * 获取会话信息
 */
export async function getSession(sessionId: string): Promise<SessionInfoResponse> {
  return request<SessionInfoResponse>({
    method: 'GET',
    url: `/sessions/${sessionId}`,
  });
}

/**
 * 获取所有会话列表
 * 后端返回 { sessions: SessionInfoResponse[] }
 */
export async function listSessions(): Promise<SessionInfoResponse[]> {
  const res = await request<{ sessions: SessionInfoResponse[] }>({
    method: 'GET',
    url: '/sessions',
  });
  return res.sessions || [];
}

/**
 * 删除会话
 */
export async function deleteSession(sessionId: string): Promise<void> {
  return request<void>({
    method: 'DELETE',
    url: `/sessions/${sessionId}`,
  });
}

/**
 * 清空会话历史
 */
export async function clearSessionHistory(sessionId: string): Promise<ClearHistoryResponse> {
  return request<ClearHistoryResponse>({
    method: 'POST',
    url: `/sessions/${sessionId}/clear`,
  });
}
