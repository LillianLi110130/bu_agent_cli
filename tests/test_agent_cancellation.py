from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from agent_core import Agent
from agent_core.llm.messages import BaseMessage, Function, ToolCall, ToolMessage, UserMessage
from agent_core.llm.openai.chat import ChatOpenAI
from agent_core.llm.views import ChatInvokeCompletion, ChatInvokeCompletionChunk
from agent_core.tools.decorator import tool
from cli.worker.runtime_factory import EchoLLM


def _create_agent() -> Agent:
    return Agent(
        llm=EchoLLM(prefix="echo:"),
        tools=[],
        system_prompt="test",
    )


class ScriptedLLM:
    def __init__(self, responses: list[ChatInvokeCompletion]) -> None:
        self.responses = list(responses)
        self.invocations: list[list[BaseMessage]] = []
        self.model = "scripted-model"

    @property
    def provider(self) -> str:
        return "fake"

    @property
    def name(self) -> str:
        return self.model

    async def ainvoke(
        self,
        messages: list[BaseMessage],
        tools=None,
        tool_choice=None,
        **kwargs,
    ) -> ChatInvokeCompletion:
        del tools, tool_choice, kwargs
        self.invocations.append(list(messages))
        if not self.responses:
            raise AssertionError("No scripted response left for ScriptedLLM")
        return self.responses.pop(0)

    async def astream(
        self,
        messages: list[BaseMessage],
        tools=None,
        tool_choice=None,
        **kwargs,
    ) -> AsyncIterator[ChatInvokeCompletionChunk]:
        del messages, tools, tool_choice, kwargs
        if False:
            yield ChatInvokeCompletionChunk()


@pytest.mark.asyncio
async def test_run_cancellable_returns_result_without_cancellation() -> None:
    agent = _create_agent()

    result = await agent._run_cancellable(asyncio.sleep(0, result="done"))

    assert result == "done"


@pytest.mark.asyncio
async def test_run_cancellable_exits_quickly_when_cancelled() -> None:
    agent = _create_agent()
    agent._cancel_event = asyncio.Event()

    task = asyncio.create_task(agent._run_cancellable(asyncio.sleep(10)))
    await asyncio.sleep(0.05)
    agent._cancel_event.set()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=0.5)


@pytest.mark.asyncio
async def test_query_stream_cancellation_appends_synthetic_tool_results() -> None:
    slow_started = asyncio.Event()

    @tool("Fast tool")
    async def fast_tool() -> str:
        return "fast result"

    @tool("Slow tool")
    async def slow_tool() -> str:
        slow_started.set()
        await asyncio.Event().wait()
        return "slow result"

    llm = ScriptedLLM(
        [
            ChatInvokeCompletion(
                tool_calls=[
                    ToolCall(
                        id="call-fast",
                        function=Function(name="fast_tool", arguments="{}"),
                    ),
                    ToolCall(
                        id="call-slow",
                        function=Function(name="slow_tool", arguments="{}"),
                    ),
                ]
            )
        ]
    )
    agent = Agent(
        llm=llm,
        tools=[fast_tool, slow_tool],
        system_prompt="test",
    )
    cancel_event = asyncio.Event()
    events = []

    async def consume_events() -> None:
        async for event in agent.query_stream("run tools", cancel_event=cancel_event):
            events.append(event)

    task = asyncio.create_task(consume_events())
    await asyncio.wait_for(slow_started.wait(), timeout=0.5)
    cancel_event.set()
    await asyncio.wait_for(task, timeout=0.5)

    assert [getattr(event, "content", None) for event in events if hasattr(event, "content")] == [
        "[Cancelled by user]"
    ]

    tool_messages = [message for message in agent.messages if isinstance(message, ToolMessage)]
    assert [message.tool_call_id for message in tool_messages] == ["call-fast", "call-slow"]
    assert tool_messages[0].content == "fast result"
    assert tool_messages[0].is_error is False
    assert tool_messages[1].content == "Tool execution cancelled by user."
    assert tool_messages[1].is_error is True

    sanitized = ChatOpenAI(model="gpt-4o-mini")._sanitize_messages_for_openai(
        [*agent.messages, UserMessage(content="follow up")]
    )

    assert [message.role for message in sanitized] == [
        "system",
        "user",
        "assistant",
        "tool",
        "tool",
        "user",
    ]


@pytest.mark.asyncio
async def test_sleep_with_cancel_returns_after_timeout_when_not_cancelled() -> None:
    agent = _create_agent()
    agent._cancel_event = asyncio.Event()

    await asyncio.wait_for(agent._sleep_with_cancel(0.05), timeout=0.3)


@pytest.mark.asyncio
async def test_sleep_with_cancel_exits_quickly_when_cancelled() -> None:
    agent = _create_agent()
    agent._cancel_event = asyncio.Event()

    task = asyncio.create_task(agent._sleep_with_cancel(10))
    await asyncio.sleep(0.05)
    agent._cancel_event.set()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=0.5)
