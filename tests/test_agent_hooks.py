from __future__ import annotations

import json
from collections.abc import AsyncIterator

import pytest

from agent_core.agent import (
    Agent,
    AgentRunState,
    AuditHook,
    HumanApprovalDecision,
    HumanApprovalRequest,
    PermissionEnforcementHook,
    ToolPolicyHook,
)
from agent_core.agent.events import FinalResponseEvent, HiddenUserMessageEvent
from agent_core.agent.runtime_events import ToolCallRequested
from agent_core.agent.runtime_loop import AgentRuntimeLoop
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

    async def ainvoke_streaming(
        self,
        messages: list[BaseMessage],
        tools=None,
        tool_choice=None,
        **kwargs,
    ) -> ChatInvokeCompletion:
        return await self.ainvoke(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs,
        )

    async def astream(
        self,
        messages: list[BaseMessage],
        tools=None,
        tool_choice=None,
        **kwargs,
    ) -> AsyncIterator[ChatInvokeCompletionChunk]:
        response = await self.ainvoke(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs,
        )
        if response.content:
            yield ChatInvokeCompletionChunk(delta=response.content)
        if response.tool_calls:
            yield ChatInvokeCompletionChunk(tool_calls=response.tool_calls)
        if response.usage is not None or response.stop_reason is not None:
            yield ChatInvokeCompletionChunk(
                usage=response.usage,
                stop_reason=response.stop_reason,
            )


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


@pytest.mark.asyncio
async def test_query_uses_runtime_loop_for_basic_completion():
    llm = FakeLLM([ChatInvokeCompletion(content="done")])
    agent = Agent(llm=llm, tools=[], system_prompt="system prompt")

    result = await agent.query("hello")

    assert result == "done"
    assert len(llm.invocations) == 1
    assert [message.role for message in agent.messages] == ["system", "user", "assistant"]


@pytest.mark.asyncio
async def test_runtime_loop_handle_tool_call_requested_uses_override_result():
    @tool("Echo payload")
    async def echo(payload: str) -> str:
        return payload

    agent = Agent(llm=FakeLLM([]), tools=[echo])
    state = AgentRunState(query_mode="query", max_iterations=5, iterations=1)
    runtime_loop = AgentRuntimeLoop(agent=agent, state=state)
    event = ToolCallRequested(
        tool_call=ToolCall(
            id="call-1",
            function=Function(name="echo", arguments='{"payload":"hello"}'),
        ),
        iteration=1,
    )

    emitted_events, ui_events = await runtime_loop._handle_event(event, "hook override")

    assert len(emitted_events) == 1
    assert emitted_events[0].tool_result.text == "hook override"
    assert emitted_events[0].tool_result.is_error is False
    assert ui_events == []


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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


@pytest.mark.asyncio
async def test_audit_hook_records_runtime_events():
    audit_hook = AuditHook()
    llm = FakeLLM([ChatInvokeCompletion(content="done")])
    agent = Agent(llm=llm, tools=[], hooks=[audit_hook])

    result = await agent.query("hello")

    assert result == "done"
    recorded_events = [record["event"] for record in audit_hook.records]
    assert "RunStarted" in recorded_events
    assert "RunFinished" in recorded_events


@pytest.mark.asyncio
async def test_permission_enforcement_hook_allows_ask_command_when_approval_disabled():
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
                        function=Function(name="bash", arguments='{"command":"rm -rf build"}'),
                    )
                ]
            ),
            ChatInvokeCompletion(content="done"),
        ]
    )
    agent = Agent(
        llm=llm,
        tools=[bash],
        hooks=[PermissionEnforcementHook()],
    )
    handler = FakeApprovalHandler(HumanApprovalDecision(approved=True))
    agent.human_in_loop_handler = handler
    agent.human_in_loop_config.enabled = False

    result = await agent.query("remove build")

    assert result == "done"
    assert called["value"] is True
    assert handler.requests == []


@pytest.mark.asyncio
async def test_permission_enforcement_hook_denies_block_command():
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
                        function=Function(name="bash", arguments='{"command":"git reset --hard"}'),
                    )
                ]
            ),
            ChatInvokeCompletion(content="blocked"),
        ]
    )
    agent = Agent(
        llm=llm,
        tools=[bash],
        hooks=[PermissionEnforcementHook()],
    )

    result = await agent.query("reset hard")

    assert result == "blocked"
    assert called["value"] is False
    tool_messages = [message for message in agent.messages if message.role == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0].is_error is True
    assert "Bash command blocked by permission policy" in tool_messages[0].text
    assert "git_reset_hard" in tool_messages[0].text


@pytest.mark.asyncio
async def test_permission_enforcement_hook_allows_approved_ask_command():
    called = {"value": False}

    @tool("Execute shell command")
    async def bash(command: str) -> str:
        called["value"] = True
        return f"ran: {command}"

    handler = FakeApprovalHandler(HumanApprovalDecision(approved=True, scope="once"))
    llm = FakeLLM(
        [
            ChatInvokeCompletion(
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        function=Function(name="bash", arguments='{"command":"rm -rf build"}'),
                    )
                ]
            ),
            ChatInvokeCompletion(content="approved"),
        ]
    )
    agent = Agent(
        llm=llm,
        tools=[bash],
        hooks=[PermissionEnforcementHook()],
    )
    agent.human_in_loop_handler = handler
    agent.human_in_loop_config.enabled = True

    result = await agent.query("remove build")

    assert result == "approved"
    assert called["value"] is True
    assert len(handler.requests) == 1
    assert handler.requests[0].tool_name == "bash"
    assert handler.requests[0].arguments["command"] == "rm -rf build"
    assert handler.requests[0].approval_kind == "safety"
    assert handler.requests[0].approval_keys == ("safety:rm_recursive",)


@pytest.mark.asyncio
async def test_permission_enforcement_hook_blocks_rejected_ask_command():
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
                        function=Function(name="bash", arguments='{"command":"rm -rf build"}'),
                    )
                ]
            )
        ]
    )
    agent = Agent(
        llm=llm,
        tools=[bash],
        hooks=[PermissionEnforcementHook()],
    )
    agent.human_in_loop_handler = handler
    agent.human_in_loop_config.enabled = True

    result = await agent.query("remove build")

    assert "本轮对话已结束" in result
    assert called["value"] is False
    tool_messages = [message for message in agent.messages if message.role == "tool"]
    assert len(tool_messages) == 1
    assert "已被人工审批拒绝" in tool_messages[0].text
    assert "operator denied" in tool_messages[0].text
    assert len(llm.invocations) == 1


@pytest.mark.asyncio
async def test_permission_enforcement_hook_fails_closed_without_handler():
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
                        function=Function(
                            name="bash",
                            arguments='{"command":"rm -rf build"}',
                        ),
                    )
                ]
            )
        ]
    )
    agent = Agent(
        llm=llm,
        tools=[bash],
        hooks=[PermissionEnforcementHook()],
    )
    agent.human_in_loop_config.enabled = True

    result = await agent.query("remove build")

    assert "本轮对话已结束" in result
    assert called["value"] is False
    tool_messages = [message for message in agent.messages if message.role == "tool"]
    assert len(tool_messages) == 1
    assert "未配置审批处理器" in tool_messages[0].text
    assert len(llm.invocations) == 1


@pytest.mark.asyncio
async def test_permission_enforcement_hook_session_approval_skips_later_same_rule():
    called = {"value": False}

    @tool("Execute shell command")
    async def bash(command: str) -> str:
        called["value"] = True
        return f"ran: {command}"

    hook = PermissionEnforcementHook()
    handler = FakeApprovalHandler(HumanApprovalDecision(approved=True, scope="session"))
    llm = FakeLLM(
        [
            ChatInvokeCompletion(
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        function=Function(name="bash", arguments='{"command":"rm -rf build"}'),
                    )
                ]
            ),
            ChatInvokeCompletion(
                tool_calls=[
                    ToolCall(
                        id="call-2",
                        function=Function(name="bash", arguments='{"command":"rm -rf dist"}'),
                    )
                ]
            ),
            ChatInvokeCompletion(content="done"),
        ]
    )
    agent = Agent(
        llm=llm,
        tools=[bash],
        hooks=[hook],
    )
    agent.human_in_loop_handler = handler
    agent.human_in_loop_config.enabled = True

    result = await agent.query("remove build and dist")

    assert result == "done"
    assert called["value"] is True
    assert len(handler.requests) == 1


async def _run_permission_enforced_bash(command: str):
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
                        function=Function(
                            name="bash",
                            arguments=json.dumps({"command": command}),
                        ),
                    )
                ]
            ),
            ChatInvokeCompletion(content="done"),
        ]
    )
    agent = Agent(
        llm=llm,
        tools=[bash],
        hooks=[PermissionEnforcementHook()],
    )
    result = await agent.query("run bash")
    return result, called, agent


@pytest.mark.asyncio
async def test_permission_enforcement_hook_blocks_directory_listing_commands():
    result, called, agent = await _run_permission_enforced_bash("dir")

    assert result == "done"
    assert called["value"] is False
    tool_messages = [message for message in agent.messages if message.role == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0].is_error is True
    assert "file_discovery" in tool_messages[0].text
    assert "file discovery or directory listing" in tool_messages[0].text
    assert "`glob_search`" in tool_messages[0].text


@pytest.mark.asyncio
async def test_permission_enforcement_hook_blocks_text_search_commands():
    result, called, agent = await _run_permission_enforced_bash("grep TODO src/app.py")

    assert result == "done"
    assert called["value"] is False
    tool_messages = [message for message in agent.messages if message.role == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0].is_error is True
    assert "text_search" in tool_messages[0].text
    assert "text search inside files" in tool_messages[0].text
    assert "`grep`" in tool_messages[0].text


@pytest.mark.asyncio
async def test_permission_enforcement_hook_blocks_file_read_commands():
    result, called, agent = await _run_permission_enforced_bash("type README.md")

    assert result == "done"
    assert called["value"] is False
    tool_messages = [message for message in agent.messages if message.role == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0].is_error is True
    assert "file_read" in tool_messages[0].text
    assert "trying to read file contents" in tool_messages[0].text
    assert "`read`" in tool_messages[0].text


@pytest.mark.asyncio
async def test_permission_enforcement_hook_points_shell_task_logs_to_task_output():
    command = (
        "sleep 3 && cat "
        "/home/user/project/.tg_agent/shell_tasks/22b62dbb/58983d6c.log"
    )
    result, called, agent = await _run_permission_enforced_bash(command)

    assert result == "done"
    assert called["value"] is False
    tool_messages = [message for message in agent.messages if message.role == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0].is_error is True
    assert "shell_task_log_read" in tool_messages[0].text
    assert "background shell task log" in tool_messages[0].text
    assert 'task_output(task_id="58983d6c"' in tool_messages[0].text
    assert "`sleep` or `cat`" in tool_messages[0].text


@pytest.mark.asyncio
async def test_permission_enforcement_hook_allows_non_file_shell_commands_when_approval_off():
    result, called, _agent = await _run_permission_enforced_bash('python -c "print(123)"')

    assert result == "done"
    assert called["value"] is True


@pytest.mark.asyncio
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
                        function=Function(name="bash", arguments='{"command":"rm -rf build"}'),
                    )
                ]
            )
        ]
    )
    agent = Agent(
        llm=llm,
        tools=[bash],
        hooks=[PermissionEnforcementHook()],
    )
    agent.human_in_loop_handler = handler
    agent.human_in_loop_config.enabled = True

    events = [event async for event in agent.query_stream("run shell")]

    assert called["value"] is False
    assert len(handler.requests) == 1
    assert isinstance(events[-1], FinalResponseEvent)
    assert "本轮对话已结束" in events[-1].content
