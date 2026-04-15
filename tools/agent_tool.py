"""Unified task delegation tool entrypoint."""

from typing import Annotated, Any

from agent_core.task import SubagentCallRequest
from agent_core.tools import Depends, tool

from tools.sandbox import SandboxContext, get_current_agent, get_sandbox_context


@tool("Delegate a task to a named agent or a forked child agent", name="delegate")
async def delegate(
    ctx: Annotated[SandboxContext, Depends(get_sandbox_context)],
    prompt: str = "",
    description: str = "",
    current_agent: Annotated[Any, Depends(get_current_agent)] = None,
    subagent_type: str | None = None,
    model: str | None = None,
    run_in_background: bool | None = None,
) -> str:
    """Unified agent delegation entrypoint."""
    executor = ctx.subagent_executor
    if executor is None:
        return "Error: Subagent executor not initialized"
    if (
        current_agent is not None
        and getattr(current_agent, "is_fork_child", False)
        and subagent_type is None
    ):
        return "Error: fork child agents cannot create nested forks."

    return await executor.call(
        parent_agent=current_agent,
        request=SubagentCallRequest(
            prompt=prompt,
            description=description,
            subagent_type=subagent_type,
            model=model,
            run_in_background=run_in_background,
        ),
    )
