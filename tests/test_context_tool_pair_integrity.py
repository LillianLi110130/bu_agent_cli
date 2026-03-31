from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_core.agent.context import ContextManager
from agent_core.llm.messages import AssistantMessage, Function, ToolCall, ToolMessage, UserMessage


class FakeCompactionService:
    def __init__(self) -> None:
        self.compact_llms = []

    async def compact(self, messages, llm=None):
        self.compact_llms.append(llm)
        return SimpleNamespace(summary="summary")


@pytest.mark.asyncio
async def test_sliding_window_keeps_assistant_tool_pair_together():
    context = ContextManager(
        messages=[
            UserMessage(content="first user"),
            AssistantMessage(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        function=Function(name="search", arguments='{"q":"hello"}'),
                    )
                ],
            ),
            ToolMessage(
                tool_call_id="call-1",
                tool_name="search",
                content="tool result",
            ),
            UserMessage(content="latest user"),
        ],
        sliding_window_messages=2,
    )
    compaction_service = FakeCompactionService()
    context._compaction_service = compaction_service
    current_llm = SimpleNamespace(model="current-model")

    changed = await context.apply_sliding_window_by_messages(
        keep_count=2,
        llm=current_llm,
        buffer=0,
    )

    assert changed is True
    roles = [message.role for message in context.get_messages()]
    assert roles == ["user", "assistant", "tool", "user"]
    kept_messages = context.get_messages()
    assert kept_messages[1].tool_calls is not None
    assert kept_messages[2].tool_call_id == "call-1"
    assert compaction_service.compact_llms == [current_llm]
