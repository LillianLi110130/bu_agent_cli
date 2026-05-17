"""Agent tool for managing scheduled cron jobs."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from agent_core.tools import Depends, tool
from cli.im_bridge import resolve_session_binding_id
from cron.jobs import CronJobStore
from cron.service import CronService, job_to_json
from tools.sandbox import SandboxContext, get_sandbox_context


@tool("Create, inspect, update, pause, resume, run, or remove scheduled cron jobs")
async def cronjob(
    ctx: Annotated[SandboxContext, Depends(get_sandbox_context)],
    action: str,
    job_id: str | None = None,
    name: str | None = None,
    prompt: str | None = None,
    schedule: str | None = None,
    delivery: str | None = None,
    execution_mode: str | None = None,
    repeat_times: int | None = None,
) -> str:
    """
    Manage Agent-native scheduled jobs.

    Args:
        ctx: Sandbox context for resolving the current workspace.
        action: One of create, list, get, update, pause, resume, run, remove.
        job_id: Existing job id for get/update/pause/resume/run/remove.
        name: Optional display name.
        prompt: Prompt to run for create/update.
        schedule: Schedule text, such as 30m, every 2h, cron, or ISO timestamp.
        delivery: local or remote.
        execution_mode: enqueue_current_session or fresh_agent_background.
        repeat_times: Optional repeat limit.
    """
    service = CronService(CronJobStore())
    normalized_action = action.strip().lower()
    workspace_root = Path(ctx.working_dir).resolve()
    session_binding_id = resolve_session_binding_id(getattr(ctx, "session_id", None))

    if normalized_action == "create":
        if prompt is None or schedule is None:
            return "Error: create requires prompt and schedule"
        job = service.create_job(
            name=name,
            prompt=prompt,
            schedule=schedule,
            workspace_root=workspace_root,
            session_binding_id=session_binding_id,
            delivery=delivery or "local",
            execution_mode=execution_mode or "enqueue_current_session",
            repeat_times=repeat_times,
        )
        return job_to_json(job)

    if normalized_action == "list":
        return job_to_json(service.list_jobs())

    if normalized_action == "get":
        if not job_id:
            return "Error: get requires job_id"
        return job_to_json(service.get_job(job_id))

    if normalized_action == "remove":
        if not job_id:
            return "Error: remove requires job_id"
        removed = service.remove_job(job_id)
        return "Removed" if removed else f"Error: job not found: {job_id}"

    if normalized_action == "pause":
        if not job_id:
            return "Error: pause requires job_id"
        return job_to_json(service.pause_job(job_id))

    if normalized_action == "resume":
        if not job_id:
            return "Error: resume requires job_id"
        return job_to_json(service.resume_job(job_id))

    if normalized_action == "update":
        if not job_id:
            return "Error: update requires job_id"
        return job_to_json(
            service.update_job(
                job_id,
                name=name,
                prompt=prompt,
                schedule=schedule,
                delivery=delivery,
                execution_mode=execution_mode,
                repeat_times=repeat_times,
            )
        )

    if normalized_action == "run":
        return "Error: run is only supported by the WorkerRunner scheduler host"

    return "Error: action must be create, list, get, update, pause, resume, run, or remove"
