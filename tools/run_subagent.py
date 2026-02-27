"""
Blocking subagent tool for synchronous execution.
"""

import asyncio
import logging
from typing import Annotated

from bu_agent_sdk.tools import Depends, tool

from tools.sandbox import SandboxContext, get_sandbox_context

logger = logging.getLogger("bu_agent_sdk.run_subagent")


@tool("Run a subagent and wait for completion")
async def run_subagent(
    ctx: Annotated[SandboxContext, Depends(get_sandbox_context)],
    subagent_name: str,
    prompt: str,
    label: str | None = None,
    timeout: Annotated[float, "Timeout in seconds (default: 300)"] = 300.0,
) -> str:
    """
    Run a subagent and BLOCK until it completes.

    This is useful when you need the result of a subagent before continuing.

    Unlike async_task (which returns immediately with a task_id),
    this tool waits for the subagent to finish and returns the actual result.

    Use this for:
    - Tasks whose result you need immediately
    - Sequential tasks where the next step depends on the previous result
    - When you need to make decisions based on the subagent output

    Use async_task instead for:
    - Parallel execution of multiple independent tasks
    - Long-running background tasks where you don't need immediate results

    Args:
        ctx: Sandbox context containing subagent_manager
        subagent_name: The name of the subagent to use (must have mode='subagent' or mode='all')
        prompt: The task description/prompt for the subagent
        label: Optional human-readable label for the task (for display)
        timeout: Maximum time to wait for completion in seconds (default: 300)

    Returns:
        The final response from the subagent. If the subagent fails,
        returns an error message describing the failure.

    Examples:
        # Run a code reviewer and wait for result
        review = run_subagent("code_reviewer", "Review this file: example.py")

        # Run a test generator with the code review result
        tests = run_subagent("test_generator", f"Generate tests for: {review}")

    Comparison with async_task:
        async_task("code_reviewer", "Review example.py")
        # Returns: "Background task started (id: abc123). Use `/task abc123` to check status."

        run_subagent("code_reviewer", "Review example.py")
        # Returns: "The code review found 3 issues: ..."
    """
    if ctx.subagent_manager is None:
        return "Error: Subagent manager not initialized"

    try:
        # Use run_and_wait to spawn and wait for completion
        result = await ctx.subagent_manager.run_and_wait(
            subagent_name=subagent_name,
            prompt=prompt,
            label=label,
            timeout=timeout,
        )

        # Return the actual result based on status
        if result.status == "completed":
            return result.final_response or f"Task completed but returned empty result."
        elif result.status == "failed":
            return f"Subagent failed: {result.error or 'Unknown error'}"
        elif result.status == "cancelled":
            return f"Subagent was cancelled: {result.error or 'Task was cancelled'}"
        else:
            return f"Subagent finished with unexpected status: {result.status}"

    except asyncio.TimeoutError:
        return f"Subagent timed out after {timeout} seconds"
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        logger.error(f"Failed to run subagent: {e}")
        return f"Error running subagent: {e}"
