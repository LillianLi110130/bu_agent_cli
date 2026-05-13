"""Agent-loop heartbeat refresh hook for team members."""

from __future__ import annotations

from typing import Any, Callable

from agent_core.agent.hooks import BaseAgentHook
from agent_core.agent.runtime_events import (
    LLMCallRequested,
    ToolCallRequested,
    ToolResultReceived,
)

HeartbeatWriter = Callable[..., None]
HeartbeatStateGetter = Callable[[], tuple[str | None, dict[str, Any]]]


class TeamHeartbeatHook(BaseAgentHook):
    """Refresh teammate heartbeat at meaningful agent-loop boundaries."""

    priority: int = 20

    def __init__(
        self,
        *,
        write_heartbeat: HeartbeatWriter,
        get_state: HeartbeatStateGetter,
    ) -> None:
        self._write_heartbeat = write_heartbeat
        self._get_state = get_state

    async def before_event(self, event, ctx):
        del ctx
        if isinstance(event, LLMCallRequested):
            self._refresh(
                heartbeat_reason="llm_call_requested",
                iteration=event.iteration,
            )
        elif isinstance(event, ToolCallRequested):
            self._refresh(
                heartbeat_reason="tool_call_requested",
                iteration=event.iteration,
                current_tool=event.tool_call.function.name,
                current_tool_call_id=event.tool_call.id,
            )
        return None

    async def after_event(self, event, ctx, emitted_events):
        del ctx, emitted_events
        if isinstance(event, ToolResultReceived):
            self._refresh(
                heartbeat_reason="tool_result_received",
                iteration=event.iteration,
                last_tool=event.tool_call.function.name,
                last_tool_call_id=event.tool_call.id,
                last_tool_error=event.tool_result.is_error,
            )
        return None

    def _refresh(self, **extra: Any) -> None:
        status, base_extra = self._get_state()
        if status is None:
            return
        self._write_heartbeat(status, **base_extra, **extra)
