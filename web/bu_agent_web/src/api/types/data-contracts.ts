/* eslint-disable */
/* tslint:disable */
// @ts-nocheck
/*
 * ---------------------------------------------------------------
 * ## THIS FILE WAS GENERATED VIA SWAGGER-TYPESCRIPT-API        ##
 * ##                                                           ##
 * ## AUTHOR: acacode                                           ##
 * ## SOURCE: https://github.com/acacode/swagger-typescript-api ##
 * ---------------------------------------------------------------
 */

/**
 * ClearHistoryResponse
 * Response model for clearing history.
 */
export interface ClearHistoryResponse {
  /**
   * Session Id
   * Session ID that was cleared
   */
  session_id: string;
  /**
   * Cleared
   * Whether the history was cleared
   */
  cleared: boolean;
}

/** HTTPValidationError */
export interface HTTPValidationError {
  /** Detail */
  detail?: ValidationError[];
}

/**
 * HealthResponse
 * Health check response.
 */
export interface HealthResponse {
  /**
   * Status
   * Server health status
   */
  status: "healthy" | "unhealthy";
  /**
   * Version
   * API version
   * @default "1.0.0"
   */
  version?: string;
  /**
   * Active Sessions
   * Number of active sessions
   * @default 0
   */
  active_sessions?: number;
}

/**
 * QueryRequest
 * Request model for agent query endpoint.
 */
export interface QueryRequest {
  /**
   * Message
   * The user message to send to the agent
   */
  message: string;
  /**
   * Session Id
   * Optional session ID for maintaining conversation state. If not provided, a new session will be created.
   */
  session_id?: string | null;
  /**
   * Stream
   * Whether to stream the response as Server-Sent Events
   * @default false
   */
  stream?: boolean;
}

/**
 * QueryResponse
 * Response model for agent query endpoint.
 */
export interface QueryResponse {
  /**
   * Session Id
   * Session ID for this conversation
   */
  session_id: string;
  /**
   * Response
   * The agent's final response
   */
  response: string;
  /** Token usage information */
  usage: UsageInfo;
}

/**
 * SessionCreateRequest
 * Request model for creating a new session.
 */
export interface SessionCreateRequest {
  /**
   * System Prompt
   * Optional system prompt for this session
   */
  system_prompt?: string | null;
}

/**
 * SessionCreateResponse
 * Response model for session creation.
 */
export interface SessionCreateResponse {
  /**
   * Session Id
   * The newly created session ID
   */
  session_id: string;
  /**
   * Created At
   * @format date-time
   */
  created_at?: string;
}

/**
 * SessionInfoResponse
 * Response model for session info.
 */
export interface SessionInfoResponse {
  /**
   * Session Id
   * Session ID
   */
  session_id: string;
  /**
   * Created At
   * Session creation timestamp
   * @format date-time
   */
  created_at: string;
  /**
   * Message Count
   * Number of messages in this session
   */
  message_count: number;
  /** Token usage information */
  usage: UsageInfo;
}

/**
 * UsageInfo
 * Token usage information.
 */
export interface UsageInfo {
  /**
   * Total Tokens
   * Total tokens used
   */
  total_tokens: number;
  /**
   * Total Prompt Tokens
   * Total prompt tokens
   */
  total_prompt_tokens: number;
  /**
   * Total Completion Tokens
   * Total completion tokens
   */
  total_completion_tokens: number;
  /**
   * By Model
   * Usage statistics broken down by model
   */
  by_model?: Record<string, any>;
}

/** ValidationError */
export interface ValidationError {
  /** Location */
  loc: (string | number)[];
  /** Message */
  msg: string;
  /** Error Type */
  type: string;
  /** Input */
  input?: any;
  /** Context */
  ctx?: object;
}
