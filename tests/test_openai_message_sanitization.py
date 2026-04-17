from __future__ import annotations

import pytest

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
