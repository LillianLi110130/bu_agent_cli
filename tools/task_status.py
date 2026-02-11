"""
Task status tool for checking background task status.
"""

from typing import Annotated

from bu_agent_sdk.tools import Depends, tool

from tools.sandbox import SandboxContext, get_sandbox_context


@tool("Get background task status")
async def task_status(
    ctx: Annotated[SandboxContext, Depends(get_sandbox_context)],
    task_id: str | None = None,
) -> str:
    """
    Get status of background tasks.

    Use this to check the progress of tasks created with async_task.

    Args:
        ctx: Sandbox context containing subagent_manager
        task_id: Specific task ID to query, or None to list all tasks

    Returns:
        Task status information. If task_id is None, returns a list of all tasks.
        If task_id is provided, returns detailed information about that specific task.

    Examples:
        task_status()  # List all tasks
        task_status(task_id="abc123")  # Get details for specific task
    """
    if ctx.subagent_manager is None:
        return "Error: Subagent manager not initialized"

    if task_id:
        result = ctx.subagent_manager.get_task_status(task_id)
        if result is None:
            return f"Error: Task '{task_id}' not found"
        return result
    else:
        return ctx.subagent_manager.list_all_tasks()
