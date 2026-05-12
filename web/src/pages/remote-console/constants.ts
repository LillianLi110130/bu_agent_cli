import type { SubmitStatus } from './types';

export const DEFAULT_WORKER_ID = 'mock-yst-123';

export const SUBMIT_STATUS_LABEL: Record<SubmitStatus, string> = {
  idle: '就绪',
  submitting: '发送中',
  submitted: '已提交',
  processing: '处理中',
  completed: '已完成',
  failed: '失败'
};

export const EMPTY_STATE_SUGGESTIONS = [
  '请帮我概览一下当前项目的目录结构和主要模块职责',
  '请帮我检查 README 和 Windows 相关脚本是否一致',
  '请帮我总结一下最近这部分改动的影响范围'
];
