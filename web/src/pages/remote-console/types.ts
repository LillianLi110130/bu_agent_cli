export type SubmitStatus =
  | 'idle'
  | 'submitting'
  | 'submitted'
  | 'processing'
  | 'completed'
  | 'failed';

export type MessageRole = 'user' | 'assistant' | 'system' | 'error';

export interface WebSessionItem {
  id: string;
  title: string;
  updatedAt: string;
  lastMessage?: string;
}

export interface ConversationMessage {
  id: string;
  role: MessageRole;
  content: string;
  createdAt: string;
  status?: SubmitStatus;
}

export interface WorkerSummary {
  workerId: string;
  isOnline?: boolean;
  lastCompletedAt?: string;
}

export interface RemoteConsoleViewState {
  workerId: string;
  activeSessionId: string;
  submitStatus: SubmitStatus;
  lastError?: string;
  useMock: boolean;
}

export interface SubmitMessagePayload {
  workerId: string;
  sessionId: string;
  content: string;
}

export interface SubmitMessageResponse {
  ok: boolean;
  acceptedAt: string;
  requestId?: string;
}

export type RequestEventType =
  | 'submitted'
  | 'processing'
  | 'progress'
  | 'completed'
  | 'failed';

export interface RequestEvent {
  type: RequestEventType;
  workerId: string;
  requestId?: string;
  ts?: string;
  content?: string;
  finalContent?: string;
  errorMessage?: string;
  finishedAt?: string;
}
