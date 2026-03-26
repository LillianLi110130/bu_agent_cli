"""
Parallel subagent tool for running multiple subagents concurrently.
"""

import logging
from typing import Annotated

from agent_core.tools import Depends, tool

from tools.sandbox import SandboxContext, get_sandbox_context

logger = logging.getLogger("agent_core.run_parallel_subagents")


@tool("Run multiple subagents in parallel and wait for all to complete")
async def run_parallel_subagents(
    ctx: Annotated[SandboxContext, Depends(get_sandbox_context)],
    tasks: Annotated[
        list[dict[str, str]],
        "List of tasks, each with 'subagent_name', 'prompt', and optional 'label'",
    ],
    timeout: Annotated[float, "Timeout in seconds for each task (default: 300)"] = 300.0,
) -> str:
    """
    Run multiple subagents in parallel and BLOCK until all complete.

    This is useful when you need to execute multiple independent tasks concurrently
    and wait for all results.

    The tasks will run in parallel (at the same time), which is much faster than
    running them sequentially. The main agent will wait for ALL tasks to complete
    before returning.

    Use this for:
    - Running multiple independent analyses at once
    - Processing multiple files simultaneously
    - Getting multiple perspectives on the same code

    Use run_subagent instead for:
    - Single task execution
    - Sequential tasks where the next depends on the previous

    Args:
        ctx: Sandbox context containing subagent_manager
        tasks: List of task specifications. Each task must be a dict with:
            - subagent_name (required): Name of the subagent to use
            - prompt (required): Task description/prompt for the subagent
            - label (optional): Human-readable label for the task
        timeout: Maximum time to wait for EACH task in seconds (default: 300)

    Returns:
        A summary of all task results. Each result includes:
        - subagent_name: The subagent that was used
        - label: The task label
        - task_id: The unique task ID
        - status: 'completed', 'failed', 'cancelled', or 'timeout'
        - result: The actual output or error message
        - execution_time_ms: How long the task took

    Examples:
        # Run multiple code reviews in parallel
        results = run_parallel_subagents(tasks=[
            {
                "subagent_name": "code_reviewer",
                "prompt": "Review auth.py",
                "label": "Review auth module"
            },
            {
                "subagent_name": "code_reviewer",
                "prompt": "Review database.py",
                "label": "Review database module"
            },
            {
                "subagent_name": "test_generator",
                "prompt": "Generate tests for utils.py",
                "label": "Generate utils tests"
            }
        ])

    Comparison:
        run_subagent("code_reviewer", "Review auth.py")
        # Returns: "The code review found 3 issues..."
        # Then runs the next task sequentially

        run_parallel_subagents(tasks=[...])
        # Returns all results at once after ALL tasks complete
        # Tasks run in parallel (faster)
    """
    if ctx.subagent_manager is None:
        return "Error: Subagent manager not initialized"

    if not tasks:
        return "Error: No tasks provided. Please provide at least one task."

    # Validate task structure
    for i, task in enumerate(tasks):
        if "subagent_name" not in task:
            return f"Error: Task at index {i} is missing 'subagent_name'"
        if "prompt" not in task:
            return f"Error: Task at index {i} is missing 'prompt'"

    try:
        results = await ctx.subagent_manager.run_parallel_subagents(
            tasks=tasks,
            timeout=timeout,
        )

        # Format results for display
        output_lines = [
            f"## Parallel Subagent Results ({len(results)} tasks)",
            "",
        ]

        for i, result in enumerate(results, 1):
            status_emoji = {
                "completed": "✓",
                "failed": "✗",
                "cancelled": "⊘",
                "timeout": "⏱",
                "error": "⚠",
            }.get(result["status"], "?")

            label = result["label"] or result["subagent_name"]
            output_lines.append(f"{i}. [{status_emoji}] {label} (task: {result['task_id']})")
            output_lines.append(f"   Status: {result['status']}")
            output_lines.append(f"   Time: {result['execution_time_ms']}ms")

            # Truncate long results
            result_text = result["result"] or "(no output)"
            if len(result_text) > 500:
                result_text = result_text[:497] + "..."
            output_lines.append(f"   Result: {result_text}")
            output_lines.append("")

        return "\n".join(output_lines)

    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        logger.error(f"Failed to run parallel subagents: {e}")
        return f"Error running parallel subagents: {e}"
