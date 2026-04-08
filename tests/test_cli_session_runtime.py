from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import cli.app as app_module
from agent_core import Agent
from cli.app import TGAgentCLI
from cli.session_runtime import CLISessionRuntime
from cli.slash_commands import SlashCommandRegistry
from cli.worker.runtime_factory import EchoLLM
from agent_core.llm.messages import Function, ToolCall
from agent_core.tools.decorator import tool
from tools import SandboxContext


def _read_artifact_parts(path: Path) -> tuple[dict[str, str], str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    marker_index = lines.index("--- artifact_body ---")
    header: dict[str, str] = {}
    for line in lines[1:marker_index]:
        if not line.strip():
            continue
        key, _, value = line.partition(":")
        header[key.strip()] = value.strip()
    body = "\n".join(lines[marker_index + 1 :])
    return header, body


class _DummyPrompter:
    def __init__(self, console):
        self.console = console


@pytest.mark.asyncio
async def test_cli_session_runtime_creates_rollout_dir_and_meta(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TG_AGENT_HOME", str(tmp_path / ".tg_agent"))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    ctx = SandboxContext.create(workspace)
    now = datetime(2026, 4, 1, 15, 30, 15, tzinfo=timezone(timedelta(hours=8)))

    runtime = CLISessionRuntime.create_for_context(ctx, now=now)

    expected_rollout = (
        runtime.root_dir
        / "sessions"
        / "2026"
        / "04"
        / "01"
        / f"rollout-20260401-153015-{ctx.session_id}"
    )
    assert runtime.root_dir == (tmp_path / ".tg_agent").resolve()
    assert runtime.rollout_dir == expected_rollout
    assert runtime.rollout_dir.exists()
    assert runtime.checkpoints_dir == expected_rollout / "checkpoints"
    assert runtime.artifacts_dir == expected_rollout / "artifacts"
    assert runtime.checkpoints_dir.exists()
    assert runtime.artifacts_dir.exists()
    assert runtime.working_state_path == expected_rollout / "working_state.json"
    assert runtime.meta_path.exists()
    assert runtime.root_dir.resolve() in {path.resolve() for path in ctx.allowed_dirs}

    meta = json.loads(runtime.meta_path.read_text(encoding="utf-8"))
    assert meta == {
        "session_id": ctx.session_id,
        "rollout_dir_name": expected_rollout.name,
        "started_at": "2026-04-01T15:30:15+08:00",
        "last_active_at": "2026-04-01T15:30:15+08:00",
        "version": 1,
    }


@pytest.mark.asyncio
async def test_helper_top_level_runtime_creates_init_rollout_dir_and_meta(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TG_AGENT_HOME", str(tmp_path / ".tg_agent"))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    ctx = SandboxContext.create(workspace)
    now = datetime(2026, 4, 1, 15, 30, 15, tzinfo=timezone(timedelta(hours=8)))

    runtime = CLISessionRuntime.create_helper_top_level_runtime(
        ctx,
        helper_name="init",
        now=now,
    )

    expected_parent = runtime.root_dir / "sessions" / "2026" / "04" / "01"
    assert runtime.root_dir == (tmp_path / ".tg_agent").resolve()
    assert runtime.rollout_dir.parent == expected_parent
    assert runtime.rollout_dir.name == f"rollout-20260401-153015-init-{runtime.session_id}"
    assert runtime.session_id != ctx.session_id
    assert runtime.checkpoints_dir == runtime.rollout_dir / "checkpoints"
    assert runtime.artifacts_dir == runtime.rollout_dir / "artifacts"
    assert runtime.checkpoints_dir.exists()
    assert runtime.artifacts_dir.exists()
    assert runtime.meta_path.exists()
    assert runtime.root_dir.resolve() in {path.resolve() for path in ctx.allowed_dirs}

    meta = json.loads(runtime.meta_path.read_text(encoding="utf-8"))
    assert meta == {
        "session_id": runtime.session_id,
        "rollout_dir_name": runtime.rollout_dir.name,
        "started_at": "2026-04-01T15:30:15+08:00",
        "last_active_at": "2026-04-01T15:30:15+08:00",
        "version": 1,
    }


@pytest.mark.asyncio
async def test_cli_input_touches_bound_session_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TG_AGENT_HOME", str(tmp_path / ".tg_agent"))
    monkeypatch.setattr(app_module, "InteractivePrompter", _DummyPrompter)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    ctx = SandboxContext.create(workspace)
    runtime = CLISessionRuntime.create_for_context(ctx)
    agent = Agent(llm=EchoLLM(prefix="echo:"), tools=[], system_prompt="test")
    cli = TGAgentCLI(
        agent=agent,
        context=ctx,
        slash_registry=SlashCommandRegistry(),
        session_runtime=runtime,
    )

    touched: list[str] = []

    def fake_touch(self) -> None:
        touched.append(self.session_id)

    monkeypatch.setattr(CLISessionRuntime, "touch", fake_touch)

    outcome = await cli._execute_input_text("/pwd")

    assert touched == [runtime.session_id]
    assert outcome.continue_running is True
    assert str(workspace) in outcome.final_content
    assert runtime.working_state_path.exists()


@pytest.mark.asyncio
async def test_bound_session_runtime_persists_large_tool_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TG_AGENT_HOME", str(tmp_path / ".tg_agent"))

    @tool("Emit a large output payload")
    async def emit_large_output() -> str:
        return "x" * 1600

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sandbox = SandboxContext.create(workspace)
    runtime = CLISessionRuntime.create_for_context(sandbox)
    agent = Agent(llm=EchoLLM(prefix="echo:"), tools=[emit_large_output], system_prompt="test")
    agent.bind_session_runtime(runtime)

    tool_message = await agent._execute_tool_call(
        ToolCall(
            id="call-large",
            function=Function(name="emit_large_output", arguments="{}"),
        )
    )

    artifact_path = runtime.artifacts_dir / "tool" / "call-large.artifact.txt"
    assert artifact_path.exists()
    header, body = _read_artifact_parts(artifact_path)
    assert header["tool_name"] == "emit_large_output"
    assert header["content_format"] == "text"
    assert body == tool_message.content


@pytest.mark.asyncio
async def test_run_init_agent_binds_independent_helper_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TG_AGENT_HOME", str(tmp_path / ".tg_agent"))
    monkeypatch.setattr(app_module, "InteractivePrompter", _DummyPrompter)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    ctx = SandboxContext.create(workspace)
    main_runtime = CLISessionRuntime.create_for_context(ctx)
    main_agent = Agent(llm=EchoLLM(prefix="echo:"), tools=[], system_prompt="test")
    cli = TGAgentCLI(
        agent=main_agent,
        context=ctx,
        slash_registry=SlashCommandRegistry(),
        session_runtime=main_runtime,
    )

    init_agent = Agent(llm=EchoLLM(prefix="echo:"), tools=[], system_prompt="init-test")
    captured: dict[str, object] = {}
    original_bind = init_agent.bind_session_runtime

    def spy_bind(runtime: CLISessionRuntime) -> None:
        captured["runtime"] = runtime
        original_bind(runtime)

    monkeypatch.setattr(init_agent, "bind_session_runtime", spy_bind)
    monkeypatch.setattr(app_module, "build_init_agent", lambda **_: init_agent)
    monkeypatch.setattr(app_module, "validate_init_output", lambda _: (True, None))
    monkeypatch.setattr(cli, "_maybe_inject_agents_md", lambda: None)

    async def fake_run_agent(user_input, has_image=False, agent=None):
        captured["run_agent"] = {
            "user_input": user_input,
            "has_image": has_image,
            "agent": agent,
        }
        return "init-finished"

    monkeypatch.setattr(cli, "_run_agent", fake_run_agent)

    result = await cli._run_init_agent()

    assert result == "init-finished"
    assert captured["run_agent"]["agent"] is init_agent
    helper_runtime = captured["runtime"]
    assert isinstance(helper_runtime, CLISessionRuntime)
    assert helper_runtime is not main_runtime
    assert helper_runtime.rollout_dir != main_runtime.rollout_dir
    assert helper_runtime.rollout_dir.name.endswith(f"init-{helper_runtime.session_id}")
    assert helper_runtime.working_state_path.exists()
