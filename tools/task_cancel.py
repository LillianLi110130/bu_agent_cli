"""
Task cancel tool for cancelling background tasks.
"""

from typing import Annotated

from bu_agent_sdk.tools import Depends, tool

from tools.sandbox import SandboxContext, get_sandbox_context


@tool("Cancel a background task")
async def task_cancel(
    ctx: Annotated[SandboxContext, Depends(get_sandbox_context)],
    task_id: str,
) -> str:
    """
    Cancel a running background task.

    Use this to stop a task that is currently executing.
    Completed or failed tasks cannot be cancelled.

    Args:
        ctx: Sandbox context containing subagent_manager
        task_id: The task ID to cancel

    Returns:
        Status message indicating whether cancellation was successful.

    Examples:
        task_cancel(task_id="abc123")
    """
    if ctx.subagent_manager is None:
        return "Error: Subagent manager not initialized"

    return await ctx.subagent_manager.cancel_task(task_id)
