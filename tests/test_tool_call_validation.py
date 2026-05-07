from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from agent_core.agent import Agent
from agent_core.agent.events import FinalResponseEvent, HiddenUserMessageEvent
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
    assert "valid JSON arguments" in recovery_message.text
    assert "Raw arguments preview" in recovery_message.text
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
    assert not any(
        isinstance(message, AssistantMessage) and message.tool_calls
        for message in agent.messages
    )


@pytest.mark.asyncio
async def test_query_allows_large_single_write_arguments():
    oversized_content = "x" * 12000
    writes: list[tuple[str, str]] = []

    @tool("Write content", name="write")
    async def fake_write(file_path: str, content: str) -> str:
        writes.append((file_path, content))
        return "wrote"

    llm = FakeLLM(
        [
            ChatInvokeCompletion(
                tool_calls=[
                    _write_call(
                        '{"file_path": "/tmp/design.md", '
                        f'"content": "{oversized_content}"'
                        "}"
                    )
                ]
            ),
            ChatInvokeCompletion(content="done"),
        ]
    )
    agent = Agent(llm=llm, tools=[fake_write])

    result = await agent.query("write a document")

    assert result == "done"
    assert len(llm.invocations) == 2
    assert writes == [("/tmp/design.md", oversized_content)]


@pytest.mark.asyncio
async def test_query_rejects_write_tool_call_with_unknown_mode_argument():
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
    assert "Argument 'mode' is not accepted by tool 'write'" in recovery_message.text


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
    assert "valid JSON arguments" in llm.invocations[1][-1].text
    assert not any(
        isinstance(message, AssistantMessage) and message.tool_calls
        for message in agent.messages
    )
