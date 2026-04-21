from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_core.llm.base import ToolDefinition
from agent_core.llm.messages import AssistantMessage, Function, ToolCall, ToolMessage, UserMessage
from agent_core.llm.openai.chat import ChatOpenAI


def test_sanitize_messages_drops_orphan_tool_messages():
    llm = ChatOpenAI(model="gpt-4o-mini")
    messages = [
        UserMessage(content="summary"),
        ToolMessage(tool_call_id="call-1", tool_name="search", content="orphan result"),
        UserMessage(content="follow up"),
    ]

    sanitized = llm._sanitize_messages_for_openai(messages)

    assert [message.role for message in sanitized] == ["user", "user"]


def test_sanitize_messages_strips_incomplete_assistant_tool_calls():
    llm = ChatOpenAI(model="gpt-4o-mini")
    messages = [
        UserMessage(content="question"),
        AssistantMessage(
            content="I will inspect that.",
            tool_calls=[
                ToolCall(
                    id="call-1",
                    function=Function(name="search", arguments='{"q":"hello"}'),
                )
            ],
        ),
        UserMessage(content="next turn"),
    ]

    sanitized = llm._sanitize_messages_for_openai(messages)

    assert [message.role for message in sanitized] == ["user", "assistant", "user"]
    assert sanitized[1].tool_calls is None
    assert sanitized[1].content == "I will inspect that."


def test_sanitize_messages_keeps_complete_tool_transactions():
    llm = ChatOpenAI(model="gpt-4o-mini")
    messages = [
        UserMessage(content="question"),
        AssistantMessage(
            content=None,
            tool_calls=[
                ToolCall(
                    id="call-1",
                    function=Function(name="search", arguments='{"q":"hello"}'),
                )
            ],
        ),
        ToolMessage(tool_call_id="call-1", tool_name="search", content="result"),
        UserMessage(content="next turn"),
    ]

    sanitized = llm._sanitize_messages_for_openai(messages)

    assert [message.role for message in sanitized] == ["user", "assistant", "tool", "user"]
    assert sanitized[1].tool_calls is not None


@pytest.mark.asyncio
async def test_chat_openai_reuses_and_closes_owned_client():
    llm = ChatOpenAI(model="gpt-4o-mini", api_key="test-key", base_url="http://example.invalid/v1")

    client = llm.get_client()
    assert llm.get_client() is client
    assert not client.is_closed()

    await llm.close()

    assert client.is_closed()
    reopened = llm.get_client()
    assert reopened is not client

    await llm.close()


class _FakeStream:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __aiter__(self):
        self._index = 0
        return self

    async def __anext__(self):
        if self._index >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._index]
        self._index += 1
        return chunk


@pytest.mark.asyncio
async def test_ainvoke_streaming_preserves_usage_from_usage_only_chunk_with_tool_calls():
    tool_delta = SimpleNamespace(
        id="call_1",
        index=0,
        function=SimpleNamespace(name="search", arguments='{"q":"hel'),
    )
    tool_delta_continued = SimpleNamespace(
        id=None,
        index=0,
        function=SimpleNamespace(name=None, arguments='lo"}'),
    )
    usage_details = SimpleNamespace(cached_tokens=3)
    completion_details = SimpleNamespace(reasoning_tokens=2)
    stream_chunks = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content=None, tool_calls=[tool_delta]),
                    finish_reason=None,
                )
            ],
            usage=None,
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content=None, tool_calls=[tool_delta_continued]),
                    finish_reason="tool_calls",
                )
            ],
            usage=None,
        ),
        SimpleNamespace(
            choices=[],
            usage=SimpleNamespace(
                prompt_tokens=21,
                prompt_tokens_details=usage_details,
                completion_tokens=5,
                completion_tokens_details=completion_details,
                total_tokens=26,
            ),
        ),
    ]

    class _FakeCompletions:
        def __init__(self, chunks):
            self._chunks = chunks
            self.calls = []

        async def create(self, **kwargs):
            self.calls.append(kwargs)
            return _FakeStream(self._chunks)

    completions = _FakeCompletions(stream_chunks)
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions),
        is_closed=lambda: False,
    )

    llm = ChatOpenAI(model="gpt-4o-mini")
    llm._client = fake_client
    llm._owns_client = False

    response = await llm.ainvoke_streaming(
        messages=[UserMessage(content="find hello")],
        tools=[
            ToolDefinition(
                name="search",
                description="Search for content",
                parameters={
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                    "required": ["q"],
                },
            )
        ],
    )

    assert completions.calls[0]["stream_options"] == {"include_usage": True}
    assert response.stop_reason == "tool_calls"
    assert response.content is None
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].id == "call_1"
    assert response.tool_calls[0].function.name == "search"
    assert response.tool_calls[0].function.arguments == '{"q":"hello"}'
    assert response.usage is not None
    assert response.usage.prompt_tokens == 21
    assert response.usage.prompt_cached_tokens == 3
    assert response.usage.completion_tokens == 7
    assert response.usage.total_tokens == 26
