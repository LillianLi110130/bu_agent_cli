from __future__ import annotations

import json
from pathlib import Path

import pytest
from rich.console import Console

from cli.cron_handler import CronSlashHandler
from cli.slash_commands import SlashCommandRegistry
from cron.jobs import CronJobStore
from cron.service import CronService
from tools.cronjob import cronjob
from tools.sandbox import SandboxContext, get_sandbox_context


def test_slash_registry_contains_cron_command() -> None:
    registry = SlashCommandRegistry()

    command = registry.get("cron")

    assert command is not None
    assert command.usage == "/cron [list|get <job_id>|remove <job_id>]"


@pytest.mark.asyncio
async def test_cron_slash_handler_lists_jobs(tmp_path: Path) -> None:
    service = CronService(CronJobStore(base_dir=tmp_path / "cron"))
    service.create_job(
        name="Morning",
        prompt="Summarize today",
        schedule="every 30m",
        workspace_root=tmp_path,
        session_binding_id="local-cli",
    )
    console = Console(record=True, force_terminal=False, width=120)
    handler = CronSlashHandler(console=console, service=service)

    handled = await handler.handle(["list"])

    output = console.export_text()
    assert handled is True
    assert "Morning" in output
    assert "every 30m" in output


@pytest.mark.asyncio
async def test_cronjob_tool_create_and_list(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    ctx = SandboxContext.create(tmp_path)
    overrides = {get_sandbox_context: lambda: ctx}

    created_raw = await cronjob.execute(
        _overrides=overrides,
        action="create",
        name="Morning",
        prompt="Summarize today",
        schedule="every 30m",
    )
    listed_raw = await cronjob.execute(_overrides=overrides, action="list")

    created = json.loads(created_raw)
    listed = json.loads(listed_raw)
    assert created["ok"] is True
    assert created["data"]["name"] == "Morning"
    assert listed["ok"] is True
    assert listed["data"][0]["name"] == "Morning"
