"""
Async task tool for creating background subagent tasks.
"""

from typing import Annotated

from bu_agent_sdk.tools import Depends, tool

from tools.sandbox import SandboxContext, get_sandbox_context


@tool("Create a background task")
async def async_task(
    ctx: Annotated[SandboxContext, Depends(get_sandbox_context)],
    subagent_name: str,
    prompt: str,
    label: str | None = None,
) -> str:
    """
    Create a background subagent task that runs asynchronously.

    The task will execute in the background without blocking the main agent.
    Results will be available when the task completes via the `/task` command.

    Use this for:
    - Long-running tasks that can run independently
    - Parallel execution of multiple tasks
    - Tasks that don't require immediate results

    Args:
        ctx: Sandbox context containing subagent_manager
        subagent_name: The name of the subagent to use (must have mode='subagent' or mode='all')
        prompt: The task description/prompt for the subagent
        label: Optional human-readable label for the task (for display)

    Returns:
        Status message with task ID. Use `/task <id>` to check status.

    Examples:
        async_task(subagent_name="explorer", prompt="Find all Python files in the project", label="Explore project")
        async_task(subagent_name="code_reviewer", prompt="Review the auth module", label="Review auth")
    """
    if ctx.subagent_manager is None:
        return "Error: Subagent manager not initialized"

    return await ctx.subagent_manager.spawn(subagent_name, prompt, label)
