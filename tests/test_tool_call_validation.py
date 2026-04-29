from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Literal

import pytest

from agent_core.agent import Agent
from agent_core.agent.events import FinalResponseEvent, HiddenUserMessageEvent
from agent_core.agent.tool_call_validation import WRITE_RECOVERY_CHUNK_MAX_CHARS
from agent_core.llm.messages import AssistantMessage, BaseMessage, Function, ToolCall, ToolMessage
from agent_core.llm.views import ChatInvokeCompletion, ChatInvokeCompletionChunk
from agent_core.tools.decorator import tool
from tools.files import write


class FakeLLM:
    def __init__(self, responses: list[ChatInvokeCompletion]):
        self.responses = list(responses)
        self.invocations: list[list[BaseMessage]] = []
        self.model = "fake-model"

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
        self.invocations.append(list(messages))
        if not self.responses:
            raise AssertionError("No scripted response left for FakeLLM")
        return self.responses.pop(0)

    async def ainvoke_streaming(
        self,
        messages: list[BaseMessage],
        tools=None,
        tool_choice=None,
        **kwargs,
    ) -> ChatInvokeCompletion:
        return await self.ainvoke(messages, tools=tools, tool_choice=tool_choice, **kwargs)

    async def astream(
        self,
        messages: list[BaseMessage],
        tools=None,
        tool_choice=None,
        **kwargs,
    ) -> AsyncIterator[ChatInvokeCompletionChunk]:
        if False:
            yield ChatInvokeCompletionChunk()


class FakeStreamingLLM(FakeLLM):
    def __init__(self, streams: list[list[ChatInvokeCompletionChunk]]):
        super().__init__(responses=[])
        self.streams = list(streams)

    async def astream(
        self,
        messages: list[BaseMessage],
        tools=None,
        tool_choice=None,
        **kwargs,
    ) -> AsyncIterator[ChatInvokeCompletionChunk]:
        self.invocations.append(list(messages))
        if not self.streams:
            raise AssertionError("No scripted stream left for FakeStreamingLLM")
        for chunk in self.streams.pop(0):
            yield chunk


def _write_call(arguments: str) -> ToolCall:
    return ToolCall(
        id="call-write",
        function=Function(name="write", arguments=arguments),
    )


@pytest.mark.asyncio
async def test_query_rejects_truncated_write_tool_call_before_history_pollution():
    truncated_arguments = '{"file_path": "/tmp/design.md"'
    llm = FakeLLM(
        [
            ChatInvokeCompletion(tool_calls=[_write_call(truncated_arguments)]),
            ChatInvokeCompletion(content="recovered"),
        ]
    )
    agent = Agent(llm=llm, tools=[write])

    result = await agent.query("write a document")

    assert result == "recovered"
    assert len(llm.invocations) == 2
    recovery_message = llm.invocations[1][-1]
    assert recovery_message.role == "user"
    assert "invalid tool call arguments" in recovery_message.text
    assert "smaller chunks" in recovery_message.text
    assert "/tmp/design.md" in recovery_message.text
    assert not any(
        isinstance(message, AssistantMessage) and message.tool_calls
        for message in agent.messages
    )
    assert not any(isinstance(message, ToolMessage) for message in agent.messages)


@pytest.mark.asyncio
async def test_query_rejects_write_tool_call_missing_content_argument():
    llm = FakeLLM(
        [
            ChatInvokeCompletion(tool_calls=[_write_call('{"file_path": "/tmp/design.md"}')]),
            ChatInvokeCompletion(content="recovered"),
        ]
    )
    agent = Agent(llm=llm, tools=[write])

    result = await agent.query("write a document")

    assert result == "recovered"
    recovery_message = llm.invocations[1][-1]
    assert "Required argument 'content' is missing" in recovery_message.text
    assert 'mode="overwrite"' in recovery_message.text
    assert not any(
        isinstance(message, AssistantMessage) and message.tool_calls
        for message in agent.messages
    )


@pytest.mark.asyncio
async def test_query_rejects_large_single_write_after_truncated_write_recovery():
    oversized_content = "x" * (WRITE_RECOVERY_CHUNK_MAX_CHARS + 1)
    llm = FakeLLM(
        [
            ChatInvokeCompletion(tool_calls=[_write_call('{"file_path": "/tmp/design.md"')]),
            ChatInvokeCompletion(
                tool_calls=[
                    _write_call(
                        '{"file_path": "/tmp/design.md", '
                        f'"content": "{oversized_content}", '
                        '"mode": "overwrite"}'
                    )
                ]
            ),
            ChatInvokeCompletion(content="recovered"),
        ]
    )
    agent = Agent(llm=llm, tools=[write])

    result = await agent.query("write a document")

    assert result == "recovered"
    assert len(llm.invocations) == 3
    second_recovery_message = llm.invocations[2][-1]
    assert "exceeding the recovery chunk limit of 1000" in second_recovery_message.text
    assert "will be rejected before execution" in second_recovery_message.text
    assert not any(isinstance(message, ToolMessage) for message in agent.messages)


@pytest.mark.asyncio
async def test_query_rejects_first_recovery_chunk_without_overwrite_mode():
    llm = FakeLLM(
        [
            ChatInvokeCompletion(tool_calls=[_write_call('{"file_path": "/tmp/design.md"')]),
            ChatInvokeCompletion(
                tool_calls=[
                    _write_call(
                        '{"file_path": "/tmp/design.md", '
                        '"content": "small chunk", '
                        '"mode": "append"}'
                    )
                ]
            ),
            ChatInvokeCompletion(content="recovered"),
        ]
    )
    agent = Agent(llm=llm, tools=[write])

    result = await agent.query("write a document")

    assert result == "recovered"
    recovery_message = llm.invocations[2][-1]
    assert 'first recovery write chunk must use mode="overwrite"' in recovery_message.text


@pytest.mark.asyncio
async def test_query_allows_small_recovery_chunks_in_order():
    writes: list[tuple[str, str, str]] = []

    @tool("Write content", name="write")
    async def fake_write(
        file_path: str,
        content: str,
        mode: Literal["overwrite", "append", "append_line"] = "overwrite",
    ) -> str:
        writes.append((file_path, content, mode))
        return "wrote chunk"

    llm = FakeLLM(
        [
            ChatInvokeCompletion(tool_calls=[_write_call('{"file_path": "/tmp/design.md"')]),
            ChatInvokeCompletion(
                tool_calls=[
                    _write_call(
                        '{"file_path": "/tmp/design.md", '
                        '"content": "first chunk", '
                        '"mode": "overwrite"}'
                    )
                ]
            ),
            ChatInvokeCompletion(
                tool_calls=[
                    _write_call(
                        '{"file_path": "/tmp/design.md", '
                        '"content": "second chunk", '
                        '"mode": "append"}'
                    )
                ]
            ),
            ChatInvokeCompletion(content="done"),
        ]
    )
    agent = Agent(llm=llm, tools=[fake_write])

    result = await agent.query("write a document")

    assert result == "done"
    assert writes == [
        ("/tmp/design.md", "first chunk", "overwrite"),
        ("/tmp/design.md", "second chunk", "append"),
    ]


@pytest.mark.asyncio
async def test_query_rejects_write_tool_call_with_invalid_mode():
    llm = FakeLLM(
        [
            ChatInvokeCompletion(
                tool_calls=[
                    _write_call(
                        '{"file_path": "/tmp/design.md", "content": "hello", "mode": "replace"}'
                    )
                ]
            ),
            ChatInvokeCompletion(content="recovered"),
        ]
    )
    agent = Agent(llm=llm, tools=[write])

    result = await agent.query("write a document")

    assert result == "recovered"
    recovery_message = llm.invocations[1][-1]
    assert "invalid value 'replace'" in recovery_message.text
    assert "overwrite" in recovery_message.text
    assert "append_line" in recovery_message.text


@pytest.mark.asyncio
async def test_query_stream_delta_rejects_invalid_tool_call_and_continues():
    llm = FakeStreamingLLM(
        [
            [ChatInvokeCompletionChunk(tool_calls=[_write_call('{"file_path": "/tmp/design.md"')])],
            [ChatInvokeCompletionChunk(delta="recovered")],
        ]
    )
    agent = Agent(llm=llm, tools=[write])

    events = [event async for event in agent.query_stream_delta("write a document")]

    assert any(isinstance(event, HiddenUserMessageEvent) for event in events)
    assert isinstance(events[-1], FinalResponseEvent)
    assert events[-1].content == "recovered"
    assert len(llm.invocations) == 2
    assert "smaller chunks" in llm.invocations[1][-1].text
    assert not any(
        isinstance(message, AssistantMessage) and message.tool_calls
        for message in agent.messages
    )
