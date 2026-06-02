from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from cron.jobs import CronJobStore
from cron.models import CronHostContext
from cron.scheduler import CronScheduler
from cli.im_bridge import SqliteBridgeStore


def _host_context(workspace_root: Path) -> CronHostContext:
    return CronHostContext(
        source="local",
        workspace_root=workspace_root,
        session_binding_id="local-cli",
        default_delivery="local",
    )


@pytest.mark.asyncio
async def test_tick_skips_when_lock_is_held(tmp_path: Path) -> None:
    store = CronJobStore(base_dir=tmp_path / "cron")
    scheduler = CronScheduler(store=store)
    scheduler.lock_path.mkdir(parents=True)

    result = await scheduler.tick(host_context=_host_context(tmp_path))

    assert result.skipped_locked is True
    assert result.claimed == 0


@pytest.mark.asyncio
async def test_due_once_job_is_enqueued_and_not_reexecuted(tmp_path: Path) -> None:
    now = datetime(2026, 5, 17, 12, 0, tzinfo=timezone.utc)
    store = CronJobStore(base_dir=tmp_path / "cron")
    job = store.create_job(
        name="Once",
        prompt="Run once",
        schedule_text=(now - timedelta(seconds=10)).isoformat(),
        workspace_root=tmp_path,
        session_binding_id="local-cli",
        now=now,
    )
    scheduler = CronScheduler(store=store)

    first = await scheduler.tick(host_context=_host_context(tmp_path), now=now)
    second = await scheduler.tick(host_context=_host_context(tmp_path), now=now)

    updated = store.get_job(job.id)
    assert first.claimed == 1
    assert first.executed == 1
    assert second.claimed == 0
    assert updated is not None
    assert updated.state == "completed"
    assert updated.last_status == "enqueued"
    assert updated.repeat.completed == 1
    assert updated.last_run is not None
    assert updated.last_run.archive_path is not None
    assert Path(updated.last_run.archive_path).exists()


@pytest.mark.asyncio
async def test_recurring_job_advances_next_run_on_claim(tmp_path: Path) -> None:
    created_at = datetime(2026, 5, 17, 10, 0, tzinfo=timezone.utc)
    tick_at = datetime(2026, 5, 17, 12, 0, tzinfo=timezone.utc)
    store = CronJobStore(base_dir=tmp_path / "cron")
    job = store.create_job(
        name="Recurring",
        prompt="Run recurring",
        schedule_text="every 30m",
        workspace_root=tmp_path,
        session_binding_id="local-cli",
        now=created_at,
    )
    scheduler = CronScheduler(store=store)

    result = await scheduler.tick(host_context=_host_context(tmp_path), now=tick_at)

    updated = store.get_job(job.id)
    assert result.claimed == 1
    assert updated is not None
    assert updated.state == "scheduled"
    assert updated.next_run_at == tick_at + timedelta(minutes=30)
    assert updated.last_status == "enqueued"


@pytest.mark.asyncio
async def test_oneshot_missed_after_grace_window(tmp_path: Path) -> None:
    now = datetime(2026, 5, 17, 12, 0, tzinfo=timezone.utc)
    store = CronJobStore(base_dir=tmp_path / "cron")
    job = store.create_job(
        name="Missed",
        prompt="Run missed",
        schedule_text=(now - timedelta(minutes=10)).isoformat(),
        workspace_root=tmp_path,
        session_binding_id="local-cli",
        now=now,
    )
    scheduler = CronScheduler(store=store, oneshot_grace_seconds=120)

    result = await scheduler.tick(host_context=_host_context(tmp_path), now=now)

    updated = store.get_job(job.id)
    assert result.missed == 1
    assert result.executed == 0
    assert updated is not None
    assert updated.state == "completed"
    assert updated.last_status == "missed"


@pytest.mark.asyncio
async def test_enqueue_current_session_writes_cron_source_meta(tmp_path: Path) -> None:
    now = datetime(2026, 5, 17, 12, 0, tzinfo=timezone.utc)
    store = CronJobStore(base_dir=tmp_path / "cron")
    job = store.create_job(
        name="Remote Queue",
        prompt="Run remote queue",
        schedule_text=(now - timedelta(seconds=1)).isoformat(),
        workspace_root=tmp_path,
        session_binding_id="worker-1",
        delivery="remote",
        now=now,
    )
    scheduler = CronScheduler(store=store)

    await scheduler.tick(
        host_context=CronHostContext(
            source="remote",
            workspace_root=tmp_path,
            session_binding_id="worker-1",
            default_delivery="remote",
        ),
        now=now,
    )

    bridge_store = SqliteBridgeStore(tmp_path, session_binding_id="worker-1")
    request = bridge_store.claim_next_pending()
    assert request is not None
    assert request.content == "Run remote queue"
    assert request.remote_response_required is True
    assert request.request_id.startswith("req_run_")
    assert request.source_meta["kind"] == "cron"
    assert request.source_meta["job_id"] == job.id


@pytest.mark.asyncio
async def test_fresh_agent_background_starts_without_blocking_tick(tmp_path: Path) -> None:
    now = datetime(2026, 5, 17, 12, 0, tzinfo=timezone.utc)
    store = CronJobStore(base_dir=tmp_path / "cron")
    job = store.create_job(
        name="Background",
        prompt="Run in background",
        schedule_text=(now - timedelta(seconds=1)).isoformat(),
        workspace_root=tmp_path,
        session_binding_id="worker-1",
        execution_mode="fresh_agent_background",
        now=now,
    )
    started = asyncio.Event()
    finish = asyncio.Event()

    async def fresh_runner(_job):
        started.set()
        await finish.wait()
        return "background done"

    scheduler = CronScheduler(store=store)

    result = await scheduler.tick(
        host_context=CronHostContext(
            source="remote",
            workspace_root=tmp_path,
            session_binding_id="worker-1",
            default_delivery="remote",
            fresh_agent_runner=fresh_runner,
        ),
        now=now,
    )

    assert result.claimed == 1
    assert result.executed == 0
    assert result.started_background == 1
    await asyncio.wait_for(started.wait(), timeout=1.0)
    claimed = store.get_job(job.id)
    assert claimed is not None
    assert claimed.last_status == "claimed"

    finish.set()
    await scheduler.wait_background_tasks()

    updated = store.get_job(job.id)
    assert updated is not None
    assert updated.last_status == "success"
    assert updated.repeat.completed == 1
    assert updated.last_run is not None
    assert updated.last_run.archive_path is not None
    assert Path(updated.last_run.archive_path).read_text(encoding="utf-8").find(
        "background done"
    ) >= 0
