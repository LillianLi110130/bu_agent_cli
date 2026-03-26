"""Runtime control events for the agent loop.

These events drive the internal execution loop and are intentionally separate
from `agent_core.agent.events`, which is reserved for UI/stream output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

from agent_core.llm.base import ToolChoice, ToolDefinition
from agent_core.llm.messages import (
    BaseMessage,
    ContentPartImageParam,
    ContentPartTextParam,
    ToolCall,
    ToolMessage,
)
from agent_core.llm.views import ChatInvokeCompletion

QueryMode = Literal["query", "stream", "stream_delta"]


@dataclass
class RunStarted:
    message: str | list[ContentPartTextParam | ContentPartImageParam]
    query_mode: QueryMode
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class IterationStarted:
    iteration: int


@dataclass
class EphemeralPruneRequested:
    iteration: int


@dataclass
class LLMCallRequested:
    messages: list[BaseMessage]
    tools: list[ToolDefinition] | None
    tool_choice: ToolChoice | None
    iteration: int


@dataclass
class LLMResponseReceived:
    response: ChatInvokeCompletion
    iteration: int


@dataclass
class ToolCallRequested:
    tool_call: ToolCall
    iteration: int


@dataclass
class ToolResultReceived:
    tool_call: ToolCall
    tool_result: ToolMessage
    iteration: int


@dataclass
class ContextMaintenanceRequested:
    response: ChatInvokeCompletion
    iteration: int


@dataclass
class FinishRequested:
    final_response: str
    iteration: int


@dataclass
class RunFinished:
    final_response: str
    iterations: int


@dataclass
class RunFailed:
    error: Exception
    iteration: int


RuntimeEvent = (
    RunStarted
    | IterationStarted
    | EphemeralPruneRequested
    | LLMCallRequested
    | LLMResponseReceived
    | ToolCallRequested
    | ToolResultReceived
    | ContextMaintenanceRequested
    | FinishRequested
    | RunFinished
    | RunFailed
)
