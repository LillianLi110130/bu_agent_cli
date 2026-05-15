"""Team coordination tools for the primary lead agent."""

from __future__ import annotations

import json
from typing import Annotated

from agent_core.team import team_experiment_disabled_message
from agent_core.tools import Depends, tool
from tools.sandbox import SandboxContext, get_sandbox_context


def _disabled_error() -> str:
    return f"Error: {team_experiment_disabled_message()}"


def _runtime_or_error(ctx: SandboxContext):
    if ctx.team_runtime is None:
        return None, _disabled_error()
    return ctx.team_runtime, None


@tool("Create a filesystem-backed agent team led by the primary CLI and make it active.", name="team_create")
async def team_create(
    ctx: Annotated[SandboxContext, Depends(get_sandbox_context)],
    name: str,
    goal: str,
    make_active: bool = True,
) -> str:
    runtime, error = _runtime_or_error(ctx)
    if error is not None:
        return error
    if make_active:
        try:
            runtime.ensure_can_start_team()
        except ValueError as exc:
            return f"Error: {exc}"
    team = runtime.create_team(name=name, goal=goal)
    if make_active:
        runtime.set_active_team(team.team_id)
    return json.dumps(team.to_dict(), ensure_ascii=False, indent=2)


@tool("Spawn one teammate process for an existing team.", name="team_spawn_member")
async def team_spawn_member(
    ctx: Annotated[SandboxContext, Depends(get_sandbox_context)],
    team_id: str,
    member_id: str,
    agent_type: str,
) -> str:
    runtime, error = _runtime_or_error(ctx)
    if error is not None:
        return error
    member = runtime.spawn_member(
        team_id=team_id,
        member_id=member_id,
        agent_type=agent_type,
    )
    return json.dumps(member.to_dict(), ensure_ascii=False, indent=2)


@tool("Create a shared team task.", name="team_create_task")
async def team_create_task(
    ctx: Annotated[SandboxContext, Depends(get_sandbox_context)],
    team_id: str,
    title: str,
    description: str,
    assigned_to: str | None = None,
    depends_on: list[str] | None = None,
    write_scope: list[str] | None = None,
) -> str:
    runtime, error = _runtime_or_error(ctx)
    if error is not None:
        return error
    task = runtime.create_task(
        team_id=team_id,
        title=title,
        description=description,
        assigned_to=assigned_to,
        depends_on=depends_on,
        write_scope=write_scope,
    )
    return json.dumps(task.to_dict(), ensure_ascii=False, indent=2)


@tool("List tasks for a team.", name="team_list_tasks")
async def team_list_tasks(
    ctx: Annotated[SandboxContext, Depends(get_sandbox_context)],
    team_id: str,
) -> str:
    runtime, error = _runtime_or_error(ctx)
    if error is not None:
        return error
    return json.dumps(
        [task.to_dict() for task in runtime.list_tasks(team_id)],
        ensure_ascii=False,
        indent=2,
    )


@tool("Update an existing shared team task.", name="team_update_task")
async def team_update_task(
    ctx: Annotated[SandboxContext, Depends(get_sandbox_context)],
    team_id: str,
    task_id: str,
    title: str | None = None,
    description: str | None = None,
    status: str | None = None,
    assigned_to: str | None = None,
    depends_on: list[str] | None = None,
    write_scope: list[str] | None = None,
    result: str | None = None,
    error: str | None = None,
) -> str:
    runtime, runtime_error = _runtime_or_error(ctx)
    if runtime_error is not None:
        return runtime_error
    try:
        task = runtime.update_task(
            team_id=team_id,
            task_id=task_id,
            title=title,
            description=description,
            status=status,
            assigned_to=assigned_to,
            depends_on=depends_on,
            write_scope=write_scope,
            result=result,
            error=error,
        )
    except ValueError as exc:
        return f"Error: {exc}"
    return json.dumps(task.to_dict(), ensure_ascii=False, indent=2)


@tool("Send a team message between the lead and teammates.", name="team_send_message")
async def team_send_message(
    ctx: Annotated[SandboxContext, Depends(get_sandbox_context)],
    team_id: str,
    recipient: str,
    body: str,
    sender: str = "lead",
    type: str = "message",
    metadata: dict | None = None,
) -> str:
    runtime, error = _runtime_or_error(ctx)
    if error is not None:
        return error
    message = runtime.send_message(
        team_id=team_id,
        sender=sender,
        recipient=recipient,
        body=body,
        type=type,
        metadata=metadata,
    )
    return json.dumps(message.to_dict(), ensure_ascii=False, indent=2)


@tool("Get team status including members, heartbeats, and tasks.", name="team_status")
async def team_status(
    ctx: Annotated[SandboxContext, Depends(get_sandbox_context)],
    team_id: str,
) -> str:
    runtime, error = _runtime_or_error(ctx)
    if error is not None:
        return error
    return json.dumps(runtime.status(team_id), ensure_ascii=False, indent=2)


@tool("Get an orchestration-friendly team snapshot.", name="team_snapshot")
async def team_snapshot(
    ctx: Annotated[SandboxContext, Depends(get_sandbox_context)],
    team_id: str,
    include_inbox: bool = True,
    ack_inbox: bool = False,
    stale_after_seconds: int = 300,
) -> str:
    runtime, error = _runtime_or_error(ctx)
    if error is not None:
        return error
    return json.dumps(
        runtime.snapshot(
            team_id,
            include_inbox=include_inbox,
            ack_inbox=ack_inbox,
            stale_after_seconds=stale_after_seconds,
        ),
        ensure_ascii=False,
        indent=2,
    )


@tool("Shutdown a team and request all teammate processes to stop.", name="team_shutdown")
async def team_shutdown(
    ctx: Annotated[SandboxContext, Depends(get_sandbox_context)],
    team_id: str,
) -> str:
    runtime, error = _runtime_or_error(ctx)
    if error is not None:
        return error
    runtime.shutdown_team(team_id)
    return f"Team '{team_id}' shutdown requested."
