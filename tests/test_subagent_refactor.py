from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from agent_core.agent import Agent, SubagentCompletionHook
from agent_core.agent.config import parse_agent_config
from agent_core.agent.registry import AgentRegistry
from agent_core.runtime import AgentCallRunner
from agent_core.task import SubagentTaskResult
from agent_core.task.subagent import SubagentCallRequest
from agent_core.tools import tool
from agent_core.llm.messages import UserMessage
from tools.agent_tool import delegate
from agent_core.llm.views import ChatInvokeCompletion


class _FakeLLM:
    def __init__(self) -> None:
        self.model = "fake-model"

    @property
    def provider(self) -> str:
        return "fake"

    @property
    def name(self) -> str:
        return self.model

    async def ainvoke(self, *args, **kwargs):
        return ChatInvokeCompletion(content="ok")


def test_parse_agent_config_supports_new_subagent_fields(tmp_path: Path) -> None:
    path = tmp_path / "code-reviewer.md"
    path.write_text(
        """---
name: code-reviewer
description: review code
model: inherit
tools:
  - glob_search
  - grep
  - read
  - bash
disallowedTools:
  - write
  - edit
maxTurns: 50
background: true
skills:
  - simplify
---
Review carefully.
""",
        encoding="utf-8",
    )

    config = parse_agent_config(path, source_scope="workspace", source_priority=1)

    assert config is not None
    assert config.name == "code-reviewer"
    assert config.tools == ["glob_search", "grep", "read", "bash"]
    assert config.disallowed_tools == ["write", "edit"]
    assert config.max_turns == 50
    assert config.background is True
    assert config.skills == ["simplify"]
    assert config.source_scope == "workspace"
    assert config.source_priority == 1


def test_parse_agent_config_rejects_legacy_boolean_tool_mapping(tmp_path: Path) -> None:
    path = tmp_path / "legacy-reviewer.md"
    path.write_text(
        """---
name: legacy-reviewer
description: legacy config
tools:
  Read: true
  Bash: true
disallowedTools:
  Write: true
---
Legacy prompt.
""",
        encoding="utf-8",
    )

    config = parse_agent_config(path, source_scope="workspace", source_priority=1)

    assert config is not None
    assert config.tools is None
    assert config.disallowed_tools == []


def test_agent_registry_prefers_higher_priority_source(tmp_path: Path) -> None:
    builtin_dir = tmp_path / "builtin"
    workspace_dir = tmp_path / "workspace"
    builtin_dir.mkdir()
    workspace_dir.mkdir()

    (builtin_dir / "reviewer.md").write_text(
        """---
description: builtin
---
builtin prompt
""",
        encoding="utf-8",
    )
    (workspace_dir / "reviewer.md").write_text(
        """---
description: workspace
---
workspace prompt
""",
        encoding="utf-8",
    )

    registry = AgentRegistry(
        agent_sources=[
            ("builtin", builtin_dir, 100),
            ("workspace", workspace_dir, 1),
        ]
    )

    config = registry.get_config("reviewer")
    assert config is not None
    assert config.description == "workspace"
    assert config.system_prompt == "workspace prompt"
    assert config.source_scope == "workspace"


def test_subagent_completion_hook_injects_background_result() -> None:
    agent = Agent(llm=_FakeLLM(), tools=[], hooks=[SubagentCompletionHook()])
    result = SubagentTaskResult(
        task_id="task123",
        subagent_name="reviewer",
        prompt="review this",
        final_response="done",
        execution_time_ms=12.0,
        status="completed",
        description="Review code",
        task_kind="named",
        subagent_type="reviewer",
        run_in_background=True,
    )

    ui_events = asyncio.run(agent._hook_manager.dispatch_subagent_result(agent, result))

    assert ui_events == [result]
    injected = agent.messages[-1]
    assert injected.role == "user"
    assert "Background subagent 'Review code' completed." in injected.text
    assert "task_id=task123" in injected.text


def test_fork_execution_builds_child_directive_and_disables_delegate() -> None:
    @tool("Read files")
    async def read() -> str:
        return "ok"

    @tool("Run shell commands")
    async def bash() -> str:
        return "ok"

    parent = Agent(
        llm=_FakeLLM(),
        tools=[delegate, read, bash],
        system_prompt="parent system",
    )
    parent.load_history([UserMessage(content="Parent context")])

    runner = AgentCallRunner(registry=AgentRegistry(agent_sources=[]), all_tools=[])
    child, initial_message = runner.build_execution(
        parent_agent=parent,
        request=SubagentCallRequest(
            prompt="Investigate the failing tests",
            description="Debug test failures",
        ),
    )

    assert child.is_fork_child is True
    assert [tool.name for tool in child.tools] == ["read", "bash"]
    assert "You are a forked child agent" in initial_message
    assert "Do not call `delegate`" in initial_message
    assert "Task summary: Debug test failures" in initial_message
    assert "Investigate the failing tests" in initial_message
    assert child.messages == parent.messages


def test_delegate_rejects_nested_fork_from_fork_child() -> None:
    result = asyncio.run(
        delegate.func(
            ctx=SimpleNamespace(subagent_executor=object()),
            prompt="do work",
            description="fork again",
            current_agent=SimpleNamespace(is_fork_child=True),
            subagent_type=None,
            model=None,
            run_in_background=None,
        )
    )

    assert result == "Error: fork child agents cannot create nested forks."
