from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from cli.init_agent import (
    InitOutputGuardHook,
    InitRepeatedToolCallGuardHook,
    build_init_agent,
    build_init_system_prompt,
    build_init_tools,
    build_init_user_prompt,
    validate_init_output,
)
from agent_core.agent.runtime_events import ToolCallRequested, ToolResultReceived
from agent_core.llm.messages import Function, ToolCall, ToolMessage


class DummyLLM:
    def __init__(self, model: str = "dummy-model") -> None:
        self.model = model

    @property
    def provider(self) -> str:
        return "dummy"

    @property
    def name(self) -> str:
        return self.model

    async def ainvoke(self, messages, tools=None, tool_choice=None, **kwargs):
        raise NotImplementedError

    async def astream(self, messages, tools=None, tool_choice=None, **kwargs):
        if False:
            yield None


def test_build_init_tools_uses_restricted_tool_whitelist() -> None:
    tool_names = [tool.name for tool in build_init_tools()]

    assert tool_names == [
        "resolve_path",
        "glob_search",
        "grep",
        "read",
        "write",
        "edit",
        "done",
    ]


def test_build_init_agent_requires_done_and_uses_restricted_tools() -> None:
    agent = build_init_agent(llm=DummyLLM(), workspace_root=Path("."))

    assert agent.require_done_tool is True
    assert agent.tool_choice == "required"
    assert agent.max_iterations == 24
    assert [tool.name for tool in agent.tools] == [
        "resolve_path",
        "glob_search",
        "grep",
        "read",
        "write",
        "edit",
        "done",
    ]
    assert len(agent.hooks) == 2
    assert isinstance(agent.hooks[0], InitOutputGuardHook)
    assert isinstance(agent.hooks[1], InitRepeatedToolCallGuardHook)


def test_init_prompts_contain_expected_constraints(tmp_path) -> None:
    system_prompt = build_init_system_prompt()
    user_prompt = build_init_user_prompt(tmp_path)

    assert "Only modify TGAGENTS.md" in system_prompt
    assert "Do not use shell commands" in system_prompt
    assert "Use `write` or `edit`" in system_prompt
    assert "Prefer high-signal files first" in system_prompt
    assert "Do not repeatedly read the same file slice" in system_prompt
    assert "stop exploring and draft the full TGAGENTS.md immediately" in system_prompt
    assert "Call the done tool" in system_prompt
    assert str(tmp_path) in user_prompt
    assert "TGAGENTS.md" in user_prompt
    assert "1. 项目目标" in user_prompt
    assert "Prefer high-signal files first" in user_prompt
    assert "Do not repeat the same read or search with identical parameters" in user_prompt
    assert "stop exploring and write TGAGENTS.md" in user_prompt
    assert "Do not call done until TGAGENTS.md exists and is non-empty" in user_prompt


def test_validate_init_output_checks_presence_and_non_empty(tmp_path) -> None:
    ok, error = validate_init_output(tmp_path)
    assert ok is False
    assert "TGAGENTS.md" in (error or "")

    agents_md = tmp_path / "TGAGENTS.md"
    agents_md.write_text("", encoding="utf-8")
    ok, error = validate_init_output(tmp_path)
    assert ok is False
    assert "空" in (error or "")

    agents_md.write_text("rules", encoding="utf-8")
    ok, error = validate_init_output(tmp_path)
    assert ok is True
    assert error is None


@pytest.mark.asyncio
async def test_init_output_guard_hook_blocks_done_before_file_exists(tmp_path) -> None:
    hook = InitOutputGuardHook(workspace_root=tmp_path)
    event = ToolCallRequested(
        tool_call=ToolCall(
            id="done-1",
            function=Function(name="done", arguments='{"message":"finished"}'),
        ),
        iteration=1,
    )

    decision = await hook.before_event(event, SimpleNamespace())

    assert decision is not None
    assert decision.action == "override_result"
    assert decision.override_result is not None
    assert "TGAGENTS.md" in decision.override_result.content


@pytest.mark.asyncio
async def test_init_output_guard_hook_allows_done_after_file_exists(tmp_path) -> None:
    hook = InitOutputGuardHook(workspace_root=tmp_path)
    (tmp_path / "TGAGENTS.md").write_text("rules", encoding="utf-8")
    event = ToolCallRequested(
        tool_call=ToolCall(
            id="done-1",
            function=Function(name="done", arguments='{"message":"finished"}'),
        ),
        iteration=1,
    )

    decision = await hook.before_event(event, SimpleNamespace())

    assert decision is None


@pytest.mark.asyncio
async def test_init_repeated_tool_call_guard_blocks_identical_read_call(tmp_path) -> None:
    del tmp_path
    hook = InitRepeatedToolCallGuardHook()
    tool_call = ToolCall(
        id="read-1",
        function=Function(
            name="read",
            arguments='{"file_path":"docs/guide.md","offset_line":100,"n_lines":20}',
        ),
    )

    first_decision = await hook.before_event(ToolCallRequested(tool_call=tool_call, iteration=1), SimpleNamespace())
    assert first_decision is None

    await hook.after_event(
        ToolResultReceived(
            tool_call=tool_call,
            tool_result=ToolMessage(
                tool_call_id="read-1",
                tool_name="read",
                content="[Lines 100-119 of 200]",
                is_error=False,
            ),
            iteration=1,
        ),
        SimpleNamespace(),
        [],
    )

    repeated_call = ToolCall(
        id="read-2",
        function=Function(
            name="read",
            arguments='{"file_path":"docs/guide.md","offset_line":100,"n_lines":20}',
        ),
    )
    repeated_decision = await hook.before_event(
        ToolCallRequested(tool_call=repeated_call, iteration=2),
        SimpleNamespace(),
    )

    assert repeated_decision is not None
    assert repeated_decision.action == "override_result"
    assert "same tool with identical parameters" in repeated_decision.override_result.content


@pytest.mark.asyncio
async def test_init_repeated_tool_call_guard_allows_changed_read_range(tmp_path) -> None:
    del tmp_path
    hook = InitRepeatedToolCallGuardHook()
    first_call = ToolCall(
        id="read-1",
        function=Function(
            name="read",
            arguments='{"file_path":"docs/guide.md","offset_line":100,"n_lines":20}',
        ),
    )

    await hook.after_event(
        ToolResultReceived(
            tool_call=first_call,
            tool_result=ToolMessage(
                tool_call_id="read-1",
                tool_name="read",
                content="[Lines 100-119 of 200]",
                is_error=False,
            ),
            iteration=1,
        ),
        SimpleNamespace(),
        [],
    )

    changed_call = ToolCall(
        id="read-2",
        function=Function(
            name="read",
            arguments='{"file_path":"docs/guide.md","offset_line":120,"n_lines":20}',
        ),
    )

    decision = await hook.before_event(
        ToolCallRequested(tool_call=changed_call, iteration=2),
        SimpleNamespace(),
    )

    assert decision is None
