// API 相关类型定义

/**
 * 会话信息
 */
export interface Session {
  session_id: string;
  created_at: string;
  updated_at: string;
  title?: string;
}

/**
 * 创建会话请求
 */
export interface CreateSessionRequest {
  title?: string;
}

/**
 * 健康检查响应
 */
export interface HealthResponse {
  status: string;
  version?: string;
}

/**
 * 清空会话历史响应
 */
export interface ClearHistoryResponse {
  success: boolean;
  message?: string;
}
