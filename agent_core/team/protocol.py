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

MESSAGE_TYPE_ALIASES = {
    "note": "message",
    "guidance": "message",
    "handoff": "message",
    "progress_update": "message",
    "messages_processed": "message",
    "messages_blocked": "message",
    "note_ack": "message_ack",
    "task_assigned": "task_assignment",
    "task_updated": "task_update",
    "task_completed": "task_done_notification",
    "task_blocked": "task_blocked_notification",
    "shutdown": "shutdown_request",
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
    raw = (value or "message").strip() or "message"
    return MESSAGE_TYPE_ALIASES.get(raw, raw)
