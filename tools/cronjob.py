"""Agent tool for managing scheduled cron jobs."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from agent_core.tools import Depends, tool
from cli.im_bridge import resolve_session_binding_id
from cron.jobs import CronJobStore
from cron.service import CronService, job_to_json
from tools.sandbox import SandboxContext, get_sandbox_context


@tool(
    "Create and manage Agent-native scheduled cron jobs. For create/update, the prompt is "
    "the short instruction the future agent should run. Do not paste long scripts, source "
    "files, or generated code into prompt. If scheduled work needs a script, first create "
    "or edit that script as a workspace file, then set prompt to reference the file path "
    "and the intended command or task."
)
async def cronjob(
    ctx: Annotated[SandboxContext, Depends(get_sandbox_context)],
    action: str,
    job_id: str | None = None,
    name: str | None = None,
    prompt: str | None = None,
    schedule: str | None = None,
    repeat_times: int | None = None,
) -> str:
    """
    Manage Agent-native scheduled jobs.

    Args:
        action: One of create, list, get, update, run, remove.
        job_id: Existing job id for get/update/run/remove.
        name: Optional display name.
        prompt: Jobs run in a fresh session with no current-chat context, so prompts
            must be self-contained. Do not inline long scripts or source code. For
            script-based jobs, write the script to a file first, then reference that
            file path and command here.
        schedule: Schedule text, such as 30m, every 2h, cron, or ISO timestamp.
        repeat_times: Optional repeat limit.
    """
    service = CronService(CronJobStore())
    normalized_action = action.strip().lower()
    workspace_root = Path(ctx.working_dir).resolve()
    bridge_store = getattr(ctx, "bridge_store", None)
    session_binding_id = getattr(bridge_store, "session_binding_id", None)
    if not session_binding_id:
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
            delivery= "remote",
            execution_mode="fresh_agent_background",
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
                delivery="remote",
                repeat_times=repeat_times,
            )
        )

    if normalized_action == "run":
        return "Error: run is only supported by the WorkerRunner scheduler host"

    return "Error: action must be create, list, get, update, pause, resume, run, or remove"
