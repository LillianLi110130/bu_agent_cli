"""Application service for cron tools and slash commands."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from cron.jobs import (
    CronJobStore,
    compute_next_run,
    normalize_repeat_times,
    parse_schedule,
    utc_now,
    validate_prompt_safety,
)
from cron.models import CronDelivery, CronExecutionMode, CronHostContext, CronJob
from cron.scheduler import CronScheduler


@dataclass(slots=True)
class CronService:
    store: CronJobStore

    def create_job(
        self,
        *,
        prompt: str,
        schedule: str,
        workspace_root: Path,
        session_binding_id: str,
        name: str | None = None,
        source: str = "local",
        delivery: str = "local",
        execution_mode: str = "enqueue_current_session",
        repeat_times: int | None = None,
        now: datetime | None = None,
    ) -> CronJob:
        return self.store.create_job(
            name=name,
            prompt=prompt,
            schedule_text=schedule,
            workspace_root=workspace_root,
            session_binding_id=session_binding_id,
            source=_coerce_source(source),
            delivery=_coerce_delivery(delivery),
            execution_mode=_coerce_execution_mode(execution_mode),
            repeat_times=repeat_times,
            now=now,
        )

    def list_jobs(self) -> list[CronJob]:
        return self.store.list_jobs()

    def get_job(self, job_id: str) -> CronJob | None:
        return self.store.get_job(job_id)

    def remove_job(self, job_id: str) -> bool:
        return self.store.remove_job(job_id)

    def pause_job(self, job_id: str) -> CronJob | None:
        return self.store.update_job(job_id, lambda job: _set_enabled(job, enabled=False))

    def resume_job(self, job_id: str) -> CronJob | None:
        return self.store.update_job(job_id, lambda job: _set_enabled(job, enabled=True))

    def update_job(
        self,
        job_id: str,
        *,
        name: str | None = None,
        prompt: str | None = None,
        schedule: str | None = None,
        delivery: str | None = None,
        execution_mode: str | None = None,
        repeat_times: int | None = None,
        now: datetime | None = None,
    ) -> CronJob | None:
        current_time = now or utc_now()

        def apply_update(job: CronJob) -> CronJob:
            if name is not None:
                job.name = name.strip() or job.name
            if prompt is not None:
                validate_prompt_safety(prompt)
                job.prompt = prompt
            if schedule is not None:
                parsed_schedule = parse_schedule(schedule, now=current_time)
                job.schedule = parsed_schedule
                job.schedule_display = parsed_schedule.display
                job.next_run_at = compute_next_run(parsed_schedule, now=current_time)
                job.state = "scheduled"
            if delivery is not None:
                job.delivery = _coerce_delivery(delivery)
            if execution_mode is not None:
                job.execution.mode = _coerce_execution_mode(execution_mode)
            if repeat_times is not None:
                job.repeat.times = normalize_repeat_times(repeat_times)
            job.updated_at = current_time
            return job

        return self.store.update_job(job_id, apply_update)

    async def run_job_now(
        self,
        job_id: str,
        *,
        host_context: CronHostContext,
        now: datetime | None = None,
    ) -> CronJob | None:
        current_time = now or utc_now()
        updated = self.store.update_job(job_id, lambda job: _schedule_now(job, current_time))
        if updated is None:
            return None
        scheduler = CronScheduler(store=self.store)
        await scheduler.tick(host_context=host_context, now=current_time)
        return self.store.get_job(job_id)

    def format_list_text(self) -> str:
        jobs = self.list_jobs()
        if not jobs:
            corrupt_text = self._format_corrupt_notice()
            return corrupt_text or "暂无 cron 任务。"
        lines = [
            "id | name | schedule | next_run_at | enabled | state | last_status",
            "--- | --- | --- | --- | --- | --- | ---",
        ]
        lines.extend(
            " | ".join(
                [
                    job.id,
                    job.name,
                    job.schedule_display,
                    job.next_run_at.isoformat(),
                    str(job.enabled).lower(),
                    job.state,
                    job.last_status or "-",
                ]
            )
            for job in jobs
        )
        corrupt_text = self._format_corrupt_notice()
        if corrupt_text:
            lines.append("")
            lines.append(corrupt_text)
        return "\n".join(lines)

    def format_detail_text(self, job_id: str) -> str:
        job = self.get_job(job_id)
        if job is None:
            return f"未找到 cron 任务：{job_id}"
        payload = job.to_dict()
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _format_corrupt_notice(self) -> str:
        backup_path = self.store.last_corrupt_backup_path
        if backup_path is None:
            return ""
        return f"注意：jobs.json 已损坏，已备份到 {backup_path}。"


def job_to_summary(job: CronJob) -> dict[str, Any]:
    return {
        "id": job.id,
        "name": job.name,
        "schedule_display": job.schedule_display,
        "next_run_at": job.next_run_at.isoformat(),
        "enabled": job.enabled,
        "state": job.state,
        "last_status": job.last_status,
        "last_error": job.last_error,
        "last_delivery_error": job.last_delivery_error,
        "last_archive_path": job.last_run.archive_path if job.last_run else None,
    }


def job_to_json(job: CronJob | list[CronJob] | None) -> str:
    if job is None:
        return json.dumps({"ok": False, "error": "job not found"}, ensure_ascii=False, indent=2)
    if isinstance(job, list):
        payload: Any = [job_to_summary(item) for item in job]
    else:
        payload = job.to_dict()
    return json.dumps({"ok": True, "data": payload}, ensure_ascii=False, indent=2)


def _set_enabled(job: CronJob, *, enabled: bool) -> CronJob:
    job.enabled = enabled
    job.updated_at = utc_now()
    return job


def _schedule_now(job: CronJob, now: datetime) -> CronJob:
    job.next_run_at = now
    if job.state == "completed" and job.schedule.kind != "once":
        job.state = "scheduled"
    job.updated_at = now
    return job


def _coerce_source(value: str) -> str:
    normalized = (value or "local").strip().lower()
    if normalized not in {"local", "remote"}:
        raise ValueError("source must be local or remote")
    return normalized


def _coerce_delivery(value: str) -> CronDelivery:
    normalized = (value or "local").strip().lower()
    if normalized not in {"local", "remote"}:
        raise ValueError("delivery must be local or remote")
    return normalized  # type: ignore[return-value]


def _coerce_execution_mode(value: str) -> CronExecutionMode:
    normalized = (value or "enqueue_current_session").strip().lower()
    if normalized not in {"enqueue_current_session", "fresh_agent_background"}:
        raise ValueError(
            "execution_mode must be enqueue_current_session or fresh_agent_background"
        )
    return normalized  # type: ignore[return-value]
