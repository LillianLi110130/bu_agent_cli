"""State container for a single runtime execution of the agent loop."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone

from bu_agent_sdk.agent.runtime_events import QueryMode
from bu_agent_sdk.llm.views import ChatInvokeCompletion, ChatInvokeUsage


@dataclass
class AgentRunState:
    query_mode: QueryMode
    max_iterations: int
    iterations: int = 0
    done: bool = False
    final_response: str = ""
    last_response: ChatInvokeCompletion | None = None
    last_usage: ChatInvokeUsage | None = None
    error: Exception | None = None
    overflow_recovery_attempted: bool = False
    incomplete_todos_prompted: bool = False
    current_event_name: str | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None
    cancel_event: asyncio.Event | None = None
