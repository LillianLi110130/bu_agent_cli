from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from bu_agent_sdk.agent import Agent, AuditHook, ToolPolicyHook
from bu_agent_sdk.agent.events import FinalResponseEvent, HiddenUserMessageEvent
from bu_agent_sdk.llm.messages import BaseMessage, Function, ToolCall
from bu_agent_sdk.llm.views import ChatInvokeCompletion, ChatInvokeCompletionChunk
from bu_agent_sdk.tools.decorator import tool


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

    async def astream(
        self,
        messages: list[BaseMessage],
        tools=None,
        tool_choice=None,
        **kwargs,
    ) -> AsyncIterator[ChatInvokeCompletionChunk]:
        if False:
            yield ChatInvokeCompletionChunk()


class TodoAgent(Agent):
    async def _get_incomplete_todos_prompt(self) -> str | None:
        if not getattr(self, "_todo_prompted_once", False):
            self._todo_prompted_once = True
            return "There are unfinished todos. Continue working."
        return None


@pytest.mark.anyio
async def test_query_uses_runtime_loop_for_basic_completion():
    llm = FakeLLM([ChatInvokeCompletion(content="done")])
    agent = Agent(llm=llm, tools=[], system_prompt="system prompt")

    result = await agent.query("hello")

    assert result == "done"
    assert len(llm.invocations) == 1
    assert [message.role for message in agent.messages] == ["system", "user", "assistant"]


@pytest.mark.anyio
async def test_finish_guard_hook_continues_when_todos_incomplete():
    llm = FakeLLM(
        [
            ChatInvokeCompletion(content="premature"),
            ChatInvokeCompletion(content="final"),
        ]
    )
    agent = TodoAgent(llm=llm, tools=[])

    result = await agent.query("hello")

    assert result == "final"
    assert len(llm.invocations) == 2
    assert [message.role for message in agent.messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]


@pytest.mark.anyio
async def test_tool_policy_hook_blocks_disallowed_tool():
    called = {"value": False}

    @tool("Dangerous tool")
    async def dangerous() -> str:
        called["value"] = True
        return "ran"

    llm = FakeLLM(
        [
            ChatInvokeCompletion(
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        function=Function(name="dangerous", arguments="{}"),
                    )
                ]
            ),
            ChatInvokeCompletion(content="blocked"),
        ]
    )
    agent = Agent(
        llm=llm,
        tools=[dangerous],
        hooks=[ToolPolicyHook(deny_tool_names={"dangerous"})],
    )

    result = await agent.query("run dangerous")

    assert result == "blocked"
    assert called["value"] is False
    tool_messages = [message for message in agent.messages if message.role == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0].is_error is True
    assert "blocked by ToolPolicyHook" in tool_messages[0].text


@pytest.mark.anyio
async def test_query_stream_emits_hidden_message_from_finish_guard():
    llm = FakeLLM(
        [
            ChatInvokeCompletion(content="premature"),
            ChatInvokeCompletion(content="final"),
        ]
    )
    agent = TodoAgent(llm=llm, tools=[])

    events = [event async for event in agent.query_stream("hello")]

    assert any(isinstance(event, HiddenUserMessageEvent) for event in events)
    assert isinstance(events[-1], FinalResponseEvent)
    assert events[-1].content == "final"


@pytest.mark.anyio
async def test_audit_hook_records_runtime_events():
    audit_hook = AuditHook()
    llm = FakeLLM([ChatInvokeCompletion(content="done")])
    agent = Agent(llm=llm, tools=[], hooks=[audit_hook])

    result = await agent.query("hello")

    assert result == "done"
    recorded_events = [record["event"] for record in audit_hook.records]
    assert "RunStarted" in recorded_events
    assert "RunFinished" in recorded_events
