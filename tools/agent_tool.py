"""Unified task delegation tool entrypoint."""

import json
from typing import Annotated, Any

from pydantic import BaseModel, Field, field_validator

from agent_core.task import SubagentCallRequest
from agent_core.tools import Depends, tool

from tools.sandbox import SandboxContext, get_current_agent, get_sandbox_context


class DelegateParallelItem(BaseModel):
    """One foreground delegated agent invocation."""

    prompt: str = Field(description="Full task prompt for this child agent.")
    description: str = Field(description="Short label shown in the task dashboard.")
    subagent_type: str | None = Field(
        default=None,
        description=(
            "Optional named agent to use. Leave empty to fork the current agent "
            "with the current conversation context."
        ),
    )
    model: str | None = Field(
        default=None,
        description=(
            "Ignored by runtime policy. Named subagents use their markdown config, "
            "`inherit` uses the parent model, and forks always use the parent model."
        ),
    )

    @field_validator("prompt", "description")
    @classmethod
    def _strip_required_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("subagent_type", "model", mode="before")
    @classmethod
    def _strip_optional_text(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value


class DelegateParallelParams(BaseModel):
    """Arguments for parallel foreground delegation."""

    agents: list[DelegateParallelItem] = Field(
        min_length=2,
        description=(
            "Two or more new foreground delegated agents to run in parallel. "
            "Use this only when you need multiple foreground agents at once."
        ),
    )


@tool(
    "Delegate exactly one agent task. "
    "Use this when you need one child agent, either in the foreground or in the background. "
    "If you need multiple new foreground agents at the same time, use `delegate_parallel` instead. "
    "Set `subagent_type` to call a named agent such as `code_reviewer` or `code_writer`. "
    "Leave `subagent_type` empty to fork the current agent with the current conversation context. "
    "Named subagents use the model defined in markdown config, and `model: inherit` means use the parent agent model. "
    "Forked children always use the parent agent model. "
    "Any `model` argument on this tool is ignored by runtime policy. "
    "`prompt` must contain the full task for the child agent. "
    "`description` must be a short human-readable label shown in task status and the CLI dashboard. "
    "Set `run_in_background=true` only for long-running work that does not need an immediate answer. "
    "Leave `run_in_background` unset or false for a normal foreground call.",
    name="delegate",
)
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

    return await executor.run_local_agent_task(
        parent_agent=current_agent,
        request=SubagentCallRequest(
            prompt=prompt,
            description=description,
            subagent_type=subagent_type,
            model=None,
            run_in_background=run_in_background,
        ),
    )


@tool(
    "Start two or more new foreground agent tasks in parallel. "
    "Use this only when you need multiple foreground agents at the same time. "
    "Do not use this for a single agent, and do not use this for background work. "
    "For exactly one child agent, use `delegate` instead. "
    "Each item follows the same agent selection rules as `delegate`: "
    "named subagents read model from markdown config, `model: inherit` uses the parent model, "
    "and forked children always use the parent model. "
    "Any `model` field provided here is ignored by runtime policy. "
    "Fork child agents cannot use this tool to create nested forks.",
    name="delegate_parallel",
    args_schema=DelegateParallelParams,
)
async def delegate_parallel(
    params: DelegateParallelParams,
    ctx: Annotated[SandboxContext, Depends(get_sandbox_context)],
    current_agent: Annotated[Any, Depends(get_current_agent)] = None,
) -> str:
    """Run multiple foreground delegated agents concurrently."""
    executor = ctx.subagent_executor
    if executor is None:
        return "Error: Subagent executor not initialized"

    if current_agent is not None and getattr(current_agent, "is_fork_child", False):
        if any(item.subagent_type is None for item in params.agents):
            return "Error: fork child agents cannot create nested forks."

    requests = [
        SubagentCallRequest(
            prompt=item.prompt,
            description=item.description,
            subagent_type=item.subagent_type,
            model=None,
            run_in_background=False,
        )
        for item in params.agents
    ]
    results = await executor.run_parallel_foreground(
        parent_agent=current_agent,
        requests=requests,
    )
    return json.dumps(
        {
            "results": [result.to_dict() for result in results],
        },
        ensure_ascii=False,
        indent=2,
    )
