"""Structured member agent event logging for filesystem-backed teams."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from agent_core.agent.hooks import BaseAgentHook
from agent_core.agent.runtime_events import (
    ContextMaintenanceRequested,
    FinishRequested,
    IterationStarted,
    LLMCallRequested,
    LLMResponseReceived,
    RunFailed,
    RunFinished,
    RunStarted,
    ToolCallRequested,
    ToolResultReceived,
)
from agent_core.team.models import utc_now_iso

PREVIEW_LIMIT = 500


class TeamAgentEventLogHook(BaseAgentHook):
    """Append compact runtime-event records to a member session JSONL file."""

    priority: int = 900

    def __init__(
        self,
        *,
        team_id: str,
        member_id: str,
        event_log_path: Path,
    ) -> None:
        self.team_id = team_id
        self.member_id = member_id
        self.event_log_path = event_log_path

    async def before_event(self, event, ctx):
        del ctx
        payload = self._payload_for_event(event)
        if payload is None:
            return None
        self._append(type(event).__name__, payload)
        return None

    def _append(self, event_name: str, payload: dict[str, Any]) -> None:
        record = {
            "id": f"evt_{uuid.uuid4().hex[:12]}",
            "type": self._event_type(event_name),
            "runtime_event": event_name,
            "team_id": self.team_id,
            "member_id": self.member_id,
            "created_at": utc_now_iso(),
            "payload": payload,
        }
        self.event_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.event_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _payload_for_event(self, event) -> dict[str, Any] | None:
        if isinstance(event, RunStarted):
            return {
                "query_mode": event.query_mode,
                "message_preview": self._preview(event.message),
            }
        if isinstance(event, IterationStarted):
            return {"iteration": event.iteration}
        if isinstance(event, LLMCallRequested):
            return {
                "iteration": event.iteration,
                "message_count": len(event.messages),
                "tool_count": len(event.tools or []),
                "tool_choice": self._preview(event.tool_choice),
            }
        if isinstance(event, LLMResponseReceived):
            response = event.response
            return {
                "iteration": event.iteration,
                "has_tool_calls": response.has_tool_calls,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "name": tool_call.function.name,
                    }
                    for tool_call in response.tool_calls
                ],
                "content_preview": self._preview(response.content),
                "stop_reason": response.stop_reason,
                "usage": response.usage.model_dump() if response.usage else None,
            }
        if isinstance(event, ToolCallRequested):
            return {
                "iteration": event.iteration,
                "tool_call_id": event.tool_call.id,
                "tool": event.tool_call.function.name,
                "args_preview": self._preview(event.tool_call.function.arguments),
            }
        if isinstance(event, ToolResultReceived):
            return {
                "iteration": event.iteration,
                "tool_call_id": event.tool_call.id,
                "tool": event.tool_call.function.name,
                "is_error": event.tool_result.is_error,
                "result_preview": self._preview(event.tool_result.text),
            }
        if isinstance(event, ContextMaintenanceRequested):
            return {"iteration": event.iteration}
        if isinstance(event, FinishRequested):
            return {
                "iteration": event.iteration,
                "final_preview": self._preview(event.final_response),
            }
        if isinstance(event, RunFinished):
            return {
                "iterations": event.iterations,
                "final_preview": self._preview(event.final_response),
            }
        if isinstance(event, RunFailed):
            return {
                "iteration": event.iteration,
                "error_type": type(event.error).__name__,
                "error": self._preview(str(event.error)),
            }
        return None

    @staticmethod
    def _event_type(event_name: str) -> str:
        chars: list[str] = []
        for index, char in enumerate(event_name):
            previous = event_name[index - 1] if index > 0 else ""
            next_char = event_name[index + 1] if index + 1 < len(event_name) else ""
            if (
                char.isupper()
                and index > 0
                and (previous.islower() or previous.isdigit() or next_char.islower())
            ):
                chars.append("_")
            chars.append(char.lower())
        return "".join(chars)

    @staticmethod
    def _preview(value: Any, *, limit: int = PREVIEW_LIMIT) -> str:
        if value is None:
            return ""
        text = value if isinstance(value, str) else str(value)
        if len(text) <= limit:
            return text
        return text[: limit - 3] + "..."
