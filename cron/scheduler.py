"""Cron scheduler tick implementation."""

from __future__ import annotations

import copy
import logging
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path

from cli.im_bridge import FileBridgeStore
from cron.archive import CronArchiveWriter
from cron.delivery import CronDeliveryPort
from cron.jobs import CronJobStore, compute_next_run, utc_now
from cron.locks import CronTickLock
from cron.models import CronHostContext, CronJob, CronRun, CronTickResult

logger = logging.getLogger("cron.scheduler")


class CronScheduler:
    """Claim and execute due cron jobs."""

    def __init__(
        self,
        *,
        store: CronJobStore | None = None,
        archive_writer: CronArchiveWriter | None = None,
        delivery_port: CronDeliveryPort | None = None,
        oneshot_grace_seconds: int = 120,
    ) -> None:
        self.store = store or CronJobStore()
        self.archive_writer = archive_writer or CronArchiveWriter(base_dir=self.store.base_dir)
        self.delivery_port = delivery_port or CronDeliveryPort()
        self.oneshot_grace_seconds = oneshot_grace_seconds
        self.lock_path = self.store.base_dir / ".tick.lock"

    async def tick(
        self,
        *,
        host_context: CronHostContext,
        now: datetime | None = None,
    ) -> CronTickResult:
        current_time = now or utc_now()
        with CronTickLock(self.lock_path) as tick_lock:
            if not tick_lock.acquired:
                return CronTickResult(skipped_locked=True)
            claimed_jobs, missed_count = self._claim_due_jobs(current_time)

        result = CronTickResult(claimed=len(claimed_jobs), missed=missed_count)
        for job in claimed_jobs:
            try:
                await self._execute_claimed_job(job, host_context=host_context)
                result.executed += 1
            except Exception as exc:
                message = str(exc) or exc.__class__.__name__
                logger.exception(f"Cron job execution failed for job_id={job.id}: {message}")
                result.errors.append(f"{job.id}: {message}")
        return result

    def _claim_due_jobs(self, now: datetime) -> tuple[list[CronJob], int]:
        claimed: list[CronJob] = []
        missed_count = 0

        def claim(jobs: list[CronJob]) -> list[CronJob]:
            nonlocal missed_count
            updated_jobs: list[CronJob] = []
            for job in jobs:
                if not self._is_due(job, now):
                    updated_jobs.append(job)
                    continue

                scheduled_at = job.next_run_at
                if self._is_missed_oneshot(job, now):
                    job.state = "completed"
                    job.last_status = "missed"
                    job.last_error = "One-shot job missed its grace window"
                    job.updated_at = now
                    missed_count += 1
                    updated_jobs.append(job)
                    continue

                run = CronRun(
                    run_id=new_run_id(job.id, scheduled_at),
                    scheduled_at=scheduled_at,
                    claimed_at=now,
                    execution_mode=job.execution.mode,
                    status="claimed",
                )
                job.last_run = run
                job.last_status = "claimed"
                job.last_error = None
                job.last_delivery_error = None
                job.updated_at = now

                if job.schedule.kind == "once":
                    job.state = "completed"
                else:
                    job.next_run_at = compute_next_run(
                        job.schedule,
                        now=now,
                        last_run_at=scheduled_at,
                    )

                claimed.append(copy.deepcopy(job))
                updated_jobs.append(job)
            return updated_jobs

        self.store.modify_jobs(claim)
        return claimed, missed_count

    def _is_due(self, job: CronJob, now: datetime) -> bool:
        return job.enabled and job.state == "scheduled" and job.next_run_at <= now

    def _is_missed_oneshot(self, job: CronJob, now: datetime) -> bool:
        if job.schedule.kind != "once":
            return False
        return now > job.next_run_at + timedelta(seconds=self.oneshot_grace_seconds)

    async def _execute_claimed_job(
        self,
        job: CronJob,
        *,
        host_context: CronHostContext,
    ) -> None:
        run = self._require_last_run(job)
        run.started_at = utc_now()
        output = ""
        error: str | None = None

        try:
            if job.execution.mode == "enqueue_current_session":
                output = self._enqueue_current_session(job, run)
                run.status = "enqueued"
                run.delivery_status = "queued" if job.delivery == "remote" else "local_only"
            elif job.execution.mode == "fresh_agent_background":
                output = await self._run_fresh_agent_background(job, host_context=host_context)
                run.status = "success"
                run.delivery_status = "local_only"
                if job.delivery == "remote":
                    delivery_result = await self.delivery_port.complete(
                        job=job,
                        run_id=run.run_id,
                        final_content=output,
                        context=host_context,
                    )
                    run.delivery_status = delivery_result.status
                    run.delivery_error = delivery_result.error
                    if not delivery_result.ok:
                        job.last_delivery_error = delivery_result.error
            else:
                raise ValueError(f"Unsupported cron execution mode: {job.execution.mode}")
        except Exception as exc:
            error = str(exc) or exc.__class__.__name__
            run.status = "failed"
            run.delivery_status = "not_attempted"
            job.last_error = error
        finally:
            run.finished_at = utc_now()
            archive_path = self.archive_writer.write(
                job=job,
                run=run,
                output=output,
                error=error,
            )
            run.archive_path = str(archive_path)
            self._write_execution_result(job, run)

    def _enqueue_current_session(self, job: CronJob, run: CronRun) -> str:
        store = FileBridgeStore(
            root_dir=Path(job.execution.workspace_root),
            session_binding_id=job.execution.session_binding_id,
        )
        request_id = f"req_{run.run_id}"
        request = store.enqueue_text(
            job.prompt,
            source=job.source,
            source_meta={
                "kind": "cron",
                "job_id": job.id,
                "job_name": job.name,
                "run_id": run.run_id,
                "delivery": job.delivery,
            },
            remote_response_required=job.delivery == "remote",
            request_id=request_id,
            input_kind="text",
        )
        run.bridge_request_id = request.request_id
        return (
            "Cron job was queued into the current session bridge.\n\n"
            f"- Request ID: {request.request_id}\n"
            f"- Sequence: {request.seq}\n"
            "- Note: this archive records enqueue success, not the final Agent response."
        )

    async def _run_fresh_agent_background(
        self,
        job: CronJob,
        *,
        host_context: CronHostContext,
    ) -> str:
        runner = host_context.fresh_agent_runner
        if not callable(runner):
            raise RuntimeError("fresh_agent_background requires a host fresh_agent_runner")
        return await runner(job)

    def _write_execution_result(self, job: CronJob, run: CronRun) -> None:
        with CronTickLock(self.lock_path, timeout_seconds=5.0) as tick_lock:
            if not tick_lock.acquired:
                raise TimeoutError(f"Timed out waiting for cron tick lock: {self.lock_path}")

            def update(stored_job: CronJob) -> CronJob:
                stored_job.last_run = run
                stored_job.last_run_at = run.finished_at
                stored_job.last_status = run.status
                stored_job.last_error = job.last_error
                stored_job.last_delivery_error = job.last_delivery_error or run.delivery_error
                stored_job.updated_at = run.finished_at or utc_now()
                if run.status in {"enqueued", "success"}:
                    stored_job.repeat.completed += 1
                    if (
                        stored_job.repeat.times is not None
                        and stored_job.repeat.completed >= stored_job.repeat.times
                    ):
                        stored_job.state = "completed"
                return stored_job

            self.store.update_job(job.id, update)

    @staticmethod
    def _require_last_run(job: CronJob) -> CronRun:
        if job.last_run is None:
            raise RuntimeError(f"Claimed cron job has no last_run metadata: {job.id}")
        return job.last_run


def new_run_id(job_id: str, scheduled_at: datetime) -> str:
    utc_scheduled = scheduled_at.astimezone(timezone.utc)
    stamp = utc_scheduled.strftime("%Y%m%dT%H%M%SZ")
    suffix = secrets.token_hex(4)
    return f"run_{stamp}_{job_id.removeprefix('cron_')}_{suffix}"
