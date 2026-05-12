"""Canonical communication protocol for filesystem-backed teams."""

from __future__ import annotations

CANONICAL_MESSAGE_TYPES = {
    "message",
    "clarification_response",
    "task_assignment",
    "task_update",
    "shutdown_request",
    "message_ack",
    "status_check",
    "status_response",
    "clarification_request",
    "task_done_notification",
    "task_blocked_notification",
    "worker_failed",
    "idle_notification",
    "stopped",
    "started",
}

MODEL_CONTEXT_MESSAGE_TYPES = {"message", "clarification_response"}

RUNTIME_CONTROL_MESSAGE_TYPES = {
    "task_assignment",
    "task_update",
    "shutdown_request",
    "message_ack",
    "status_check",
    "status_response",
}

LEAD_AUTO_TRIGGER_MESSAGE_TYPES = {
    "clarification_request",
    "task_done_notification",
    "task_blocked_notification",
    "worker_failed",
    "idle_notification",
}


def normalize_message_type(value: str | None) -> str:
    return (value or "message").strip() or "message"
