"""
Task output tool for reading background shell task logs.
"""

from __future__ import annotations

import json
from typing import Annotated

from agent_core.tools import Depends, tool

from tools.sandbox import SandboxContext, get_sandbox_context


@tool("Read output/logs from a background shell task", context_policy="summarize")
async def task_output(
    ctx: Annotated[SandboxContext, Depends(get_sandbox_context)],
    task_id: str,
    wait_for: str | None = None,
    timeout: float = 30.0,
    max_chars: int = 4000,
) -> str:
    """
    Read output from a background shell task.

    Args:
        ctx: Sandbox context containing shell_task_manager
        task_id: Background shell task id returned by bash with run_in_background=true
        wait_for: Optional substring to wait for in the task output
        timeout: Maximum wait time in seconds when wait_for is provided
        max_chars: Maximum number of output characters to return
    """
    if ctx.shell_task_manager is None:
        return "Error: Shell task manager not initialized"

    task = ctx.shell_task_manager.get_task(task_id)
    if task is None:
        return f"Error: Task '{task_id}' not found"

    matched = False
    if wait_for:
        output, matched = await ctx.shell_task_manager.wait_for_output(
            task_id,
            pattern=wait_for,
            timeout=timeout,
            max_chars=max(max_chars, 12000),
        )
    else:
        output = ctx.shell_task_manager.read_output(task_id, max_chars=max_chars)

    payload = {
        "task_id": task.task_id,
        "status": task.status,
        "command": task.command,
        "cwd": task.cwd,
        "returncode": task.returncode,
        "wait_for": wait_for,
        "matched": matched,
        "log_path": str(task.log_path),
        "output": output[-max_chars:] if len(output) > max_chars else output,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
