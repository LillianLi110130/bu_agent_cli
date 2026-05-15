from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from agent_core import Agent
from agent_core.agent.events import (
    FinalResponseEvent,
    TextDeltaEvent,
    TextEvent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingEvent,
    ThinkingStartEvent,
)
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
        del tools, tool_choice, kwargs
        self.invocations.append(list(messages))
        if not self.responses:
            raise AssertionError("No scripted response left for ScriptedLLM")
        response = self.responses.pop(0)
        yield ChatInvokeCompletionChunk(
            delta=response.content or "",
            tool_calls=response.tool_calls,
            thinking=response.thinking,
            usage=response.usage,
            stop_reason=response.stop_reason,
        )


class ScriptedStreamingLLM:
    def __init__(self, chunks: list[ChatInvokeCompletionChunk]) -> None:
        self.chunks = list(chunks)
        self.stream_invocations: list[list[BaseMessage]] = []
        self.model = "scripted-streaming-model"

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
        del messages, tools, tool_choice, kwargs
        raise AssertionError("query_stream() should consume astream() in stream mode")

    async def astream(
        self,
        messages: list[BaseMessage],
        tools=None,
        tool_choice=None,
        **kwargs,
    ) -> AsyncIterator[ChatInvokeCompletionChunk]:
        del tools, tool_choice, kwargs
        self.stream_invocations.append(list(messages))
        for chunk in self.chunks:
            yield chunk


class HangingStreamingLLM:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.closed = asyncio.Event()
        self.model = "hanging-streaming-model"

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
        del messages, tools, tool_choice, kwargs
        raise AssertionError("query_stream() should consume astream() in stream mode")

    async def astream(
        self,
        messages: list[BaseMessage],
        tools=None,
        tool_choice=None,
        **kwargs,
    ) -> AsyncIterator[ChatInvokeCompletionChunk]:
        del messages, tools, tool_choice, kwargs
        self.started.set()
        try:
            await asyncio.Event().wait()
            yield ChatInvokeCompletionChunk(delta="unreachable")
        finally:
            self.closed.set()


class SlowCancelStreamingLLM:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.model = "slow-cancel-streaming-model"

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
        del messages, tools, tool_choice, kwargs
        raise AssertionError("query_stream() should consume astream() in stream mode")

    async def astream(
        self,
        messages: list[BaseMessage],
        tools=None,
        tool_choice=None,
        **kwargs,
    ) -> AsyncIterator[ChatInvokeCompletionChunk]:
        del messages, tools, tool_choice, kwargs
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await asyncio.sleep(10)
            raise
        yield ChatInvokeCompletionChunk(delta="unreachable")


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
async def test_query_stream_cancels_while_waiting_for_stream_chunk() -> None:
    llm = HangingStreamingLLM()
    agent = Agent(llm=llm, tools=[], system_prompt="test")
    cancel_event = asyncio.Event()
    events = []

    async def consume_events() -> None:
        async for event in agent.query_stream("stream forever", cancel_event=cancel_event):
            events.append(event)

    task = asyncio.create_task(consume_events())
    await asyncio.wait_for(llm.started.wait(), timeout=0.5)

    cancel_event.set()
    await asyncio.wait_for(task, timeout=0.5)

    assert llm.closed.is_set()
    assert any(
        isinstance(event, FinalResponseEvent) and event.content == "[Cancelled by user]"
        for event in events
    )


@pytest.mark.asyncio
async def test_query_stream_cancellation_does_not_wait_for_slow_stream_close() -> None:
    llm = SlowCancelStreamingLLM()
    agent = Agent(llm=llm, tools=[], system_prompt="test")
    cancel_event = asyncio.Event()
    events = []

    async def consume_events() -> None:
        async for event in agent.query_stream("stream forever", cancel_event=cancel_event):
            events.append(event)

    task = asyncio.create_task(consume_events())
    await asyncio.wait_for(llm.started.wait(), timeout=0.5)

    cancel_event.set()
    await asyncio.wait_for(task, timeout=0.7)

    assert any(
        isinstance(event, FinalResponseEvent) and event.content == "[Cancelled by user]"
        for event in events
    )


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
async def test_query_stream_emits_text_delta_events_before_final_response() -> None:
    llm = ScriptedStreamingLLM(
        [
            ChatInvokeCompletionChunk(delta="hel"),
            ChatInvokeCompletionChunk(delta="lo", stop_reason="stop"),
        ]
    )
    agent = Agent(
        llm=llm,
        tools=[],
        system_prompt="test",
    )

    events = [event async for event in agent.query_stream("hello")]

    text_deltas = [event.delta for event in events if isinstance(event, TextDeltaEvent)]
    final_events = [event for event in events if isinstance(event, FinalResponseEvent)]
    text_events = [event for event in events if isinstance(event, TextEvent)]

    assert llm.stream_invocations
    assert text_deltas == ["hel", "lo"]
    assert len(final_events) == 1
    assert final_events[0].content == "hello"
    assert text_events == []


@pytest.mark.asyncio
async def test_query_stream_emits_thinking_delta_events_before_final_response() -> None:
    llm = ScriptedStreamingLLM(
        [
            ChatInvokeCompletionChunk(thinking="先"),
            ChatInvokeCompletionChunk(thinking="想", delta="答", stop_reason="stop"),
        ]
    )
    agent = Agent(
        llm=llm,
        tools=[],
        system_prompt="test",
    )

    events = [event async for event in agent.query_stream("hello")]

    thinking_starts = [event for event in events if isinstance(event, ThinkingStartEvent)]
    thinking_deltas = [event.delta for event in events if isinstance(event, ThinkingDeltaEvent)]
    thinking_ends = [event for event in events if isinstance(event, ThinkingEndEvent)]
    thinking_events = [event for event in events if isinstance(event, ThinkingEvent)]
    final_events = [event for event in events if isinstance(event, FinalResponseEvent)]

    assert len(thinking_starts) == 1
    assert thinking_deltas == ["先", "想"]
    assert len(thinking_ends) == 1
    assert thinking_starts[0].think_id == thinking_ends[0].think_id
    assert thinking_events == []
    assert len(final_events) == 1
    assert final_events[0].content == "答"


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
