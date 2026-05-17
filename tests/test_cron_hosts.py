from __future__ import annotations

from pathlib import Path

import pytest

from cli.app import TGAgentCLI


class _RecordingCronScheduler:
    def __init__(self) -> None:
        self.contexts = []

    async def tick(self, *, host_context):
        self.contexts.append(host_context)


class _FakeContext:
    def __init__(self, working_dir: Path) -> None:
        self.working_dir = working_dir


@pytest.mark.asyncio
async def test_cli_maybe_tick_cron_builds_local_host_context(tmp_path: Path) -> None:
    cli = object.__new__(TGAgentCLI)
    scheduler = _RecordingCronScheduler()
    cli._ctx = _FakeContext(tmp_path)
    cli._cron_scheduler = scheduler
    cli._cron_next_tick_at = 0

    await cli._maybe_tick_cron()

    assert len(scheduler.contexts) == 1
    context = scheduler.contexts[0]
    assert context.source == "local"
    assert context.workspace_root == tmp_path
    assert context.session_binding_id == "local-cli"
    assert context.default_delivery == "local"
    assert callable(context.fresh_agent_runner)
