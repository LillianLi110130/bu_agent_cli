"""
Core BU Agent implementation.
"""

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from bu_agent_sdk import Agent
from bu_agent_sdk.agent.events import (
    FinalResponseEvent,
    StepCompleteEvent,
    StepStartEvent,
    TextEvent,
    ThinkingEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from bu_agent_sdk.llm.messages import ContentPartTextParam


@dataclass
class ContextSnapshot:
    """Snapshot of the agent context."""

    context_usage: float = 0.0
    """The usage of the context, in percentage."""
    thinking: bool = False
    """Whether thinking mode is currently enabled."""


class BUAgent:
    """
    A simplified Agent wrapper that provides a clean interface
    for running agents with tool calling.
    """

    def __init__(self, agent: Agent, name: str = "BU Agent"):
        self._agent = agent
        self._name = name
        self._thinking = False
        self._context_usage = 0.0

    @property
    def name(self) -> str:
        return self._name

    @property
    def model_name(self) -> str:
        if hasattr(self._agent.llm, "model"):
            return self._agent.llm.model
        return "unknown-model"

    @property
    def model_capabilities(self) -> set[str] | None:
        return {
            "streaming",
            "function_calling",
        }

    @property
    def thinking(self) -> bool:
        return self._thinking

    @property
    def status(self) -> ContextSnapshot:
        return ContextSnapshot(
            context_usage=self._context_usage,
            thinking=self._thinking,
        )

    @property
    def available_slash_commands(self) -> list[Any]:
        return []

    async def run(self, user_input: str | list, on_event: callable):
        """
        Run the agent with the given user input.

        Args:
            user_input: The user input (string or list of content parts)
            on_event: Callback function for events
        """
        # Convert list to string if needed
        if isinstance(user_input, list):
            text_input = ""
            for part in user_input:
                if hasattr(part, "text"):
                    text_input += part.text
                elif isinstance(part, str):
                    text_input += part
            message = text_input
        else:
            message = user_input

        step_count = 0

        try:
            step_count += 1

            async for event in self._agent.query_stream(message):
                if isinstance(event, TextEvent):
                    on_event("text", event.content)

                elif isinstance(event, ThinkingEvent):
                    self._thinking = True
                    on_event("thinking", event.content)
                    self._thinking = False

                elif isinstance(event, ToolCallEvent):
                    on_event("tool_call", {
                        "id": event.tool_call_id,
                        "name": event.tool,
                        "arguments": json.dumps(event.args)
                    })

                elif isinstance(event, ToolResultEvent):
                    on_event("tool_result", {
                        "tool_call_id": event.tool_call_id,
                        "result": event.result,
                        "is_error": event.is_error
                    })

                elif isinstance(event, StepStartEvent):
                    on_event("step_start", {
                        "step_id": event.step_id,
                        "title": event.title,
                        "step_number": event.step_number
                    })

                elif isinstance(event, StepCompleteEvent):
                    on_event("step_complete", {
                        "step_id": event.step_id,
                        "status": event.status,
                        "duration_ms": event.duration_ms
                    })

                elif isinstance(event, FinalResponseEvent):
                    if event.content:
                        on_event("final", event.content)

        except asyncio.CancelledError:
            on_event("error", "Cancelled by user")
            raise
        except Exception as e:
            on_event("error", str(e))
            raise

    def get_usage_summary(self):
        """Get usage summary."""
        return asyncio.create_task(self._agent.get_usage())
