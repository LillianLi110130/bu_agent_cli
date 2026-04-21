from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from agent_core.agent import Agent, SubagentCompletionHook
from agent_core.agent.events import FinalResponseEvent
from agent_core.agent.config import parse_agent_config
from agent_core.agent.registry import AgentRegistry
from agent_core.runtime import AgentCallRunner
from agent_core.task import SubagentTaskResult
from agent_core.task.local_agent_task import SubagentTaskManager
from agent_core.task.local_agent_task import SubagentCallRequest
from agent_core.tools import tool
from agent_core.llm.messages import UserMessage
from tools import ALL_TOOLS
from tools.agent_tool import DelegateParallelParams, delegate, delegate_parallel
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


def test_named_subagent_model_inherit_uses_parent_llm(tmp_path: Path) -> None:
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    (agents_dir / "reviewer.md").write_text(
        """---
name: reviewer
description: review code
model: inherit
---
Review carefully.
""",
        encoding="utf-8",
    )

    registry = AgentRegistry(agent_sources=[("workspace", agents_dir, 1)])
    parent = Agent(llm=_FakeLLM(), tools=[], system_prompt="parent system")
    runner = AgentCallRunner(registry=registry, all_tools=[])

    child, initial_message = runner.build_execution(
        parent_agent=parent,
        request=SubagentCallRequest(
            prompt="Review the auth patch",
            description="Review auth patch",
            subagent_type="reviewer",
            model="claude-sonnet-4-20250514",
        ),
    )

    assert child.llm is parent.llm
    assert initial_message == "Review the auth patch"


class _FakeParallelExecutor:
    def __init__(self) -> None:
        self.parent_agent = None
        self.requests: list[SubagentCallRequest] | None = None

    async def run_parallel_foreground(self, *, parent_agent, requests):
        self.parent_agent = parent_agent
        self.requests = list(requests)
        return [
            SubagentTaskResult(
                task_id=f"task-{index}",
                subagent_name=request.subagent_type or "fork",
                prompt=request.prompt,
                final_response=f"done-{index}",
                execution_time_ms=10.0 + index,
                status="completed",
                description=request.description,
                task_kind="named" if request.subagent_type else "fork",
                subagent_type=request.subagent_type,
                model="fake-model",
                run_in_background=False,
            )
            for index, request in enumerate(self.requests, start=1)
        ]


def test_delegate_parallel_runs_multiple_foreground_agents() -> None:
    executor = _FakeParallelExecutor()

    result = asyncio.run(
        delegate_parallel.func(
            params=DelegateParallelParams(
                agents=[
                    {
                        "prompt": "Review auth changes",
                        "description": "Auth review",
                        "subagent_type": "reviewer",
                    },
                    {
                        "prompt": "Review test updates",
                        "description": "Test review",
                        "subagent_type": "reviewer",
                        "model": "inherit",
                    },
                ]
            ),
            ctx=SimpleNamespace(subagent_executor=executor),
            current_agent=SimpleNamespace(is_fork_child=False),
        )
    )

    assert executor.requests is not None
    assert len(executor.requests) == 2
    assert all(request.run_in_background is False for request in executor.requests)
    assert all(request.model is None for request in executor.requests)

    payload = json.loads(result)
    assert [item["final_response"] for item in payload["results"]] == ["done-1", "done-2"]


def test_delegate_parallel_rejects_nested_fork_from_fork_child() -> None:
    result = asyncio.run(
        delegate_parallel.func(
            params=DelegateParallelParams(
                agents=[
                    {
                        "prompt": "Handle task A",
                        "description": "Task A",
                    },
                    {
                        "prompt": "Handle task B",
                        "description": "Task B",
                        "subagent_type": "reviewer",
                    },
                ]
            ),
            ctx=SimpleNamespace(subagent_executor=_FakeParallelExecutor()),
            current_agent=SimpleNamespace(is_fork_child=True),
        )
    )

    assert result == "Error: fork child agents cannot create nested forks."


def test_delegate_parallel_is_registered_in_all_tools() -> None:
    assert delegate in ALL_TOOLS
    assert delegate_parallel in ALL_TOOLS


class _CancellingTaskManager(SubagentTaskManager):
    def __init__(self) -> None:
        self.cancelled_task_ids: list[str] = []
        super().__init__(
            registry=AgentRegistry(agent_sources=[]),
            all_tools=[],
            context=SimpleNamespace(subagent_events=None),
            skill_registry=None,
        )

    async def cancel_run(self, task_id: str) -> str:
        self.cancelled_task_ids.append(task_id)
        return await super().cancel_run(task_id)


def test_cancelled_final_response_is_recorded_as_cancelled() -> None:
    manager = SubagentTaskManager(
        registry=AgentRegistry(agent_sources=[]),
        all_tools=[],
        context=SimpleNamespace(subagent_events=None),
        skill_registry=None,
    )
    parent = Agent(llm=_FakeLLM(), tools=[], system_prompt="parent system")

    class _CancelledChild:
        def __init__(self) -> None:
            self.llm = SimpleNamespace(model="fake-model")

        async def query_stream(self, initial_message, cancel_event=None):
            del initial_message, cancel_event
            yield FinalResponseEvent(content="[Cancelled by user]")

        async def get_usage(self):
            return SimpleNamespace(
                total_prompt_tokens=0,
                total_completion_tokens=0,
                total_tokens=0,
            )

    manager._runner = SimpleNamespace(
        build_execution=lambda parent_agent, request: (_CancelledChild(), request.prompt)
    )

    result = asyncio.run(
        manager.run_foreground(
            parent_agent=parent,
            request=SubagentCallRequest(
                prompt="Review auth patch",
                description="Review auth patch",
            ),
        )
    )

    assert result.status == "cancelled"
    assert result.final_response == ""
    assert result.error == "Task was cancelled"


def test_run_foreground_cancels_child_when_parent_is_cancelled() -> None:
    manager = _CancellingTaskManager()
    parent = Agent(llm=_FakeLLM(), tools=[], system_prompt="parent system")

    async def scenario() -> None:
        task = asyncio.create_task(
            manager.run_foreground(
                parent_agent=parent,
                request=SubagentCallRequest(
                    prompt="Review auth patch",
                    description="Review auth patch",
                ),
                timeout=300.0,
            )
        )
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    import pytest

    asyncio.run(scenario())

    assert len(manager.cancelled_task_ids) == 1


def test_subagent_shutdown_cancels_all_running_tasks() -> None:
    manager = _CancellingTaskManager()
    parent = Agent(llm=_FakeLLM(), tools=[], system_prompt="parent system")

    async def scenario() -> None:
        first = await manager.start_background_run(
            parent_agent=parent,
            request=SubagentCallRequest(
                prompt="Review file A",
                description="Review file A",
            ),
        )
        second = await manager.start_background_run(
            parent_agent=parent,
            request=SubagentCallRequest(
                prompt="Review file B",
                description="Review file B",
            ),
        )
        await asyncio.sleep(0)
        await manager.shutdown(cancel_running=True)
        assert {first, second}.issubset(set(manager.cancelled_task_ids))

    asyncio.run(scenario())
