"""
Task status tool for checking background task status.
"""

from typing import Annotated
import json

from agent_core.tools import Depends, tool

from tools.sandbox import SandboxContext, get_sandbox_context


@tool("Get background task status")
async def task_status(
    ctx: Annotated[SandboxContext, Depends(get_sandbox_context)],
    task_id: str | None = None,
) -> str:
    """
    Get status of background tasks.

    Use this to check the progress of background tasks created via `call`.

    Args:
        ctx: Sandbox context containing subagent_executor
        task_id: Specific task ID to query, or None to list all tasks

    Returns:
        Task status information. If task_id is None, returns a list of all tasks.
        If task_id is provided, returns detailed information about that specific task.

    Examples:
        task_status()  # List all tasks
        task_status(task_id="abc123")  # Get details for specific task
    """
    shell_manager = ctx.shell_task_manager
    subagent_executor = ctx.subagent_executor

    if task_id:
        if shell_manager is not None:
            shell_task = shell_manager.get_task(task_id)
            if shell_task is not None:
                return json.dumps(shell_task.to_dict(), ensure_ascii=False, indent=2)
        if subagent_executor is None:
            return f"Error: Task '{task_id}' not found"
        result = subagent_executor.get_task_status(task_id)
        if result is None:
            return f"Error: Task '{task_id}' not found"
        return result
    else:
        shell_tasks = []
        if shell_manager is not None:
            shell_tasks = [task.to_dict() for task in shell_manager.list_tasks()]
        subagent_tasks = None
        if subagent_executor is not None:
            subagent_tasks = subagent_executor.list_all_tasks()
        return json.dumps(
            {
                "shell_tasks": shell_tasks,
                "subagent_tasks": subagent_tasks,
            },
            ensure_ascii=False,
            indent=2,
        )
