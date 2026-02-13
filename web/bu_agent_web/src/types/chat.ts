// 对话相关类型定义

/**
 * 消息角色
 */
export type MessageRole = 'user' | 'assistant';

/**
 * 消息状态
 */
export type MessageStatus = 'streaming' | 'done' | 'error';

/**
 * 工具调用记录
 */
export interface ToolCall {
  tool: string;
  args: Record<string, any>;
  result?: string;
  isError?: boolean;
  timestamp?: number;
}

/**
 * 消息
 */
export interface Message {
  id: string;
  role: MessageRole;
  content: string;
  timestamp: number;
  thinking?: string[];        // 思维链内容
  toolCalls?: ToolCall[];     // 工具调用记录
  status?: MessageStatus;
  sessionId: string;
}

/**
 * 会话项（用于会话列表）
 */
export interface Conversation {
  key: string;           // session_id
  label: string;         // 会话名称（首条消息截断）
  timestamp: number;     // 创建时间
  group?: string;        // 分组（今日/昨日/更早）
}

/**
 * Token 使用统计
 */
export interface UsageInfo {
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
}

/**
 * SSE 事件类型
 */
export type SSEEventType =
  | 'text_delta'
  | 'text'
  | 'thinking'
  | 'tool_call'
  | 'tool_result'
  | 'step_start'
  | 'step_complete'
  | 'final'
  | 'usage'
  | 'error';

/**
 * SSE 事件基类
 */
export interface SSEEvent {
  type: SSEEventType;
  timestamp?: string;
}

/**
 * 文本增量事件
 */
export interface TextDeltaEvent extends SSEEvent {
  type: 'text_delta';
  delta: string;
}

/**
 * 完整文本事件
 */
export interface TextEvent extends SSEEvent {
  type: 'text';
  content: string;
}

/**
 * 思维链事件
 */
export interface ThinkingEvent extends SSEEvent {
  type: 'thinking';
  content: string;
}

/**
 * 工具调用事件
 */
export interface ToolCallEvent extends SSEEvent {
  type: 'tool_call';
  tool: string;
  args: Record<string, any>;
  id?: string;
}

/**
 * 工具结果事件
 */
export interface ToolResultEvent extends SSEEvent {
  type: 'tool_result';
  tool: string;
  result: string;
  id?: string;
}

/**
 * 步骤开始事件
 */
export interface StepStartEvent extends SSEEvent {
  type: 'step_start';
  step: string;
}

/**
 * 步骤完成事件
 */
export interface StepCompleteEvent extends SSEEvent {
  type: 'step_complete';
  step: string;
}

/**
 * 最终响应事件
 */
export interface FinalEvent extends SSEEvent {
  type: 'final';
  content?: string;
}

/**
 * 使用统计事件
 */
export interface UsageEvent extends SSEEvent {
  type: 'usage';
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

/**
 * 错误事件
 */
export interface ErrorEvent extends SSEEvent {
  type: 'error';
  error: string;
  message?: string;
}

/**
 * 聊合 SSE 事件类型
 */
export type StreamEvent =
  | TextDeltaEvent
  | TextEvent
  | ThinkingEvent
  | ToolCallEvent
  | ToolResultEvent
  | StepStartEvent
  | StepCompleteEvent
  | FinalEvent
  | UsageEvent
  | ErrorEvent;

/**
 * 查询请求
 */
export interface QueryRequest {
  message: string;
  session_id?: string;
}
