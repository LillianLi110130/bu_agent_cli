from __future__ import annotations

from collections.abc import AsyncIterator
import json

import pytest

from agent_core.agent import (
    Agent,
    AuditHook,
    ExcelReadGuardHook,
    HumanApprovalDecision,
    HumanApprovalHook,
    HumanApprovalRequest,
    ToolPolicyHook,
    build_default_approval_policy,
)
from agent_core.agent.events import FinalResponseEvent, HiddenUserMessageEvent
from agent_core.llm.messages import BaseMessage, Function, ToolCall
from agent_core.llm.views import ChatInvokeCompletion, ChatInvokeCompletionChunk
from agent_core.tools.decorator import tool


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


class FakeApprovalHandler:
    def __init__(self, decision: HumanApprovalDecision):
        self.decision = decision
        self.requests: list[HumanApprovalRequest] = []

    async def request_approval(self, request: HumanApprovalRequest) -> HumanApprovalDecision:
        self.requests.append(request)
        return self.decision


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


@pytest.mark.anyio
async def test_human_approval_hook_skips_when_disabled():
    called = {"value": False}

    @tool("Execute shell command")
    async def bash(command: str) -> str:
        called["value"] = True
        return f"ran: {command}"

    handler = FakeApprovalHandler(HumanApprovalDecision(approved=True))
    llm = FakeLLM(
        [
            ChatInvokeCompletion(
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        function=Function(name="bash", arguments='{"command":"echo hi"}'),
                    )
                ]
            ),
            ChatInvokeCompletion(content="done"),
        ]
    )
    agent = Agent(
        llm=llm,
        tools=[bash],
        hooks=[HumanApprovalHook(policy=build_default_approval_policy("primary"))],
    )
    agent.human_in_loop_handler = handler
    agent.human_in_loop_config.enabled = False

    result = await agent.query("run shell")

    assert result == "done"
    assert called["value"] is True
    assert handler.requests == []


@pytest.mark.anyio
async def test_human_approval_hook_allows_approved_tool_call():
    called = {"value": False}

    @tool("Execute shell command")
    async def bash(command: str) -> str:
        called["value"] = True
        return f"ran: {command}"

    handler = FakeApprovalHandler(HumanApprovalDecision(approved=True))
    llm = FakeLLM(
        [
            ChatInvokeCompletion(
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        function=Function(name="bash", arguments='{"command":"echo hi"}'),
                    )
                ]
            ),
            ChatInvokeCompletion(content="approved"),
        ]
    )
    agent = Agent(
        llm=llm,
        tools=[bash],
        hooks=[HumanApprovalHook(policy=build_default_approval_policy("primary"))],
    )
    agent.human_in_loop_handler = handler
    agent.human_in_loop_config.enabled = True

    result = await agent.query("run shell")

    assert result == "approved"
    assert called["value"] is True
    assert len(handler.requests) == 1
    assert handler.requests[0].tool_name == "bash"
    assert handler.requests[0].arguments["command"] == "echo hi"


@pytest.mark.anyio
async def test_human_approval_hook_blocks_rejected_tool_call():
    called = {"value": False}

    @tool("Execute shell command")
    async def bash(command: str) -> str:
        called["value"] = True
        return f"ran: {command}"

    handler = FakeApprovalHandler(HumanApprovalDecision(approved=False, reason="operator denied"))
    llm = FakeLLM(
        [
            ChatInvokeCompletion(
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        function=Function(name="bash", arguments='{"command":"echo hi"}'),
                    )
                ]
            )
        ]
    )
    agent = Agent(
        llm=llm,
        tools=[bash],
        hooks=[HumanApprovalHook(policy=build_default_approval_policy("primary"))],
    )
    agent.human_in_loop_handler = handler
    agent.human_in_loop_config.enabled = True

    result = await agent.query("run shell")

    assert "本轮对话已结束" in result
    assert called["value"] is False
    tool_messages = [message for message in agent.messages if message.role == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0].is_error is True
    assert "已被人工审批拒绝" in tool_messages[0].text
    assert "operator denied" in tool_messages[0].text
    assert len(llm.invocations) == 1


@pytest.mark.anyio
async def test_human_approval_hook_fails_closed_without_handler():
    called = {"value": False}

    @tool("Execute shell command")
    async def bash(command: str) -> str:
        called["value"] = True
        return f"ran: {command}"

    llm = FakeLLM(
        [
            ChatInvokeCompletion(
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        function=Function(name="bash", arguments='{"command":"echo hi"}'),
                    )
                ]
            )
        ]
    )
    agent = Agent(
        llm=llm,
        tools=[bash],
        hooks=[HumanApprovalHook(policy=build_default_approval_policy("primary"))],
    )
    agent.human_in_loop_config.enabled = True

    result = await agent.query("run shell")

    assert "本轮对话已结束" in result
    assert called["value"] is False
    tool_messages = [message for message in agent.messages if message.role == "tool"]
    assert len(tool_messages) == 1
    assert "未配置审批处理器" in tool_messages[0].text
    assert len(llm.invocations) == 1


@pytest.mark.anyio
async def test_excel_read_guard_hook_blocks_excel_related_bash_after_read_excel():
    called = {"value": False}

    @tool("Read excel")
    async def read_excel(file_path: str) -> str:
        return json.dumps(
            {
                "resolved_path": file_path,
                "sheet_names": ["Sheet1"],
                "selected_sheet": None,
                "preview_limits": {"max_rows": 20, "max_cols": 20},
                "sheets": [],
            },
            ensure_ascii=False,
        )

    @tool("Execute shell command")
    async def bash(command: str) -> str:
        called["value"] = True
        return f"ran: {command}"

    llm = FakeLLM(
        [
            ChatInvokeCompletion(
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        function=Function(
                            name="read_excel",
                            arguments='{"file_path":"D:/workspace/demo.xlsx"}',
                        ),
                    )
                ]
            ),
            ChatInvokeCompletion(
                tool_calls=[
                    ToolCall(
                        id="call-2",
                        function=Function(
                            name="bash",
                            arguments='{"command":"python -c \\"import openpyxl\\""}',
                        ),
                    )
                ]
            ),
            ChatInvokeCompletion(content="done"),
        ]
    )
    agent = Agent(
        llm=llm,
        tools=[read_excel, bash],
        hooks=[ExcelReadGuardHook()],
    )

    result = await agent.query("analyze workbook")

    assert result == "done"
    assert called["value"] is False
    tool_messages = [message for message in agent.messages if message.role == "tool"]
    assert len(tool_messages) == 2
    assert tool_messages[1].is_error is True
    assert "read_excel" in tool_messages[1].text
    assert "Do not use `bash`" in tool_messages[1].text


@pytest.mark.anyio
async def test_excel_read_guard_hook_allows_non_excel_bash_after_read_excel():
    called = {"value": False}

    @tool("Read excel")
    async def read_excel(file_path: str) -> str:
        return json.dumps(
            {
                "resolved_path": file_path,
                "sheet_names": ["Sheet1"],
                "selected_sheet": None,
                "preview_limits": {"max_rows": 20, "max_cols": 20},
                "sheets": [],
            },
            ensure_ascii=False,
        )

    @tool("Execute shell command")
    async def bash(command: str) -> str:
        called["value"] = True
        return f"ran: {command}"

    llm = FakeLLM(
        [
            ChatInvokeCompletion(
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        function=Function(
                            name="read_excel",
                            arguments='{"file_path":"D:/workspace/demo.xlsx"}',
                        ),
                    )
                ]
            ),
            ChatInvokeCompletion(
                tool_calls=[
                    ToolCall(
                        id="call-2",
                        function=Function(name="bash", arguments='{"command":"echo hi"}'),
                    )
                ]
            ),
            ChatInvokeCompletion(content="done"),
        ]
    )
    agent = Agent(
        llm=llm,
        tools=[read_excel, bash],
        hooks=[ExcelReadGuardHook()],
    )

    result = await agent.query("analyze workbook")

    assert result == "done"
    assert called["value"] is True


@pytest.mark.anyio
async def test_query_stream_with_human_approval_rejection_still_finishes():
    called = {"value": False}

    @tool("Execute shell command")
    async def bash(command: str) -> str:
        called["value"] = True
        return f"ran: {command}"

    handler = FakeApprovalHandler(HumanApprovalDecision(approved=False, reason="operator denied"))
    llm = FakeLLM(
        [
            ChatInvokeCompletion(
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        function=Function(name="bash", arguments='{"command":"echo hi"}'),
                    )
                ]
            )
        ]
    )
    agent = Agent(
        llm=llm,
        tools=[bash],
        hooks=[HumanApprovalHook(policy=build_default_approval_policy("primary"))],
    )
    agent.human_in_loop_handler = handler
    agent.human_in_loop_config.enabled = True

    events = [event async for event in agent.query_stream("run shell")]

    assert called["value"] is False
    assert len(handler.requests) == 1
    assert isinstance(events[-1], FinalResponseEvent)
    assert "本轮对话已结束" in events[-1].content
