from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from cli.init_agent import (
    InitOutputGuardHook,
    build_init_agent,
    build_init_system_prompt,
    build_init_tools,
    build_init_user_prompt,
    validate_init_output,
)
from agent_core.agent.runtime_events import ToolCallRequested
from agent_core.llm.messages import Function, ToolCall


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
    assert [tool.name for tool in agent.tools] == [
        "resolve_path",
        "glob_search",
        "grep",
        "read",
        "write",
        "edit",
        "done",
    ]
    assert len(agent.hooks) == 1
    assert isinstance(agent.hooks[0], InitOutputGuardHook)


def test_init_prompts_contain_expected_constraints(tmp_path) -> None:
    system_prompt = build_init_system_prompt()
    user_prompt = build_init_user_prompt(tmp_path)

    assert "Only modify TGAGENTS.md" in system_prompt
    assert "Do not use shell commands" in system_prompt
    assert "Use `write` or `edit`" in system_prompt
    assert "Call the done tool" in system_prompt
    assert str(tmp_path) in user_prompt
    assert "TGAGENTS.md" in user_prompt
    assert "1. 项目目标" in user_prompt
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
