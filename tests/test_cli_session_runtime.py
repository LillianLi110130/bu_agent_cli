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
from tools import SandboxContext


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
