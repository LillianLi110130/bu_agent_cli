import json
import time
import sys
from typing import Annotated, Any
from dataclasses import dataclass

from bu_agent_sdk.tools import Depends, tool
from bu_agent_sdk.agent import Agent
from bu_agent_sdk.agent.config import AgentConfig
from bu_agent_sdk.agent.events import (
    ToolCallEvent, ToolResultEvent, ThinkingEvent, TextEvent, FinalResponseEvent,
)
from tools.sandbox import SandboxContext, get_sandbox_context
from config.model_config import get_model_config


@dataclass
class SubagentResult:
    """Subagent执行的结构化结果"""
    agent_name: str
    prompt: str
    final_response: str
    tool_calls: list[dict[str, Any]]
    execution_time_ms: float
    mode: str
    tools_used: list[str]


def _create_subagent_agent(
    config: AgentConfig,
    parent_ctx: SandboxContext,
    all_tools: list,
) -> Agent:
    """创建subagent实例"""
    from bu_agent_sdk.llm import ChatOpenAI

    model, base_url, api_key = get_model_config(config.model)

    llm = ChatOpenAI(model=model, api_key=api_key, base_url=base_url)

    agent = Agent(
        llm=llm,
        tools=all_tools,
        system_prompt=config.system_prompt,
        mode=config.mode,
        agent_config=config,
        dependency_overrides={get_sandbox_context: lambda: parent_ctx},
    )
    return agent


async def get_agent_registry_for_tool() -> Any:
    """获取AgentRegistry实例供工具使用"""
    from bu_agent_sdk.agent.registry import get_agent_registry
    return get_agent_registry()


@tool("Launch a subagent to handle a specific task")
async def task(
    subagent_name: str,
    prompt: str,
    registry: Annotated[Any, Depends(get_agent_registry_for_tool)],
    ctx: Annotated[SandboxContext, Depends(get_sandbox_context)],
) -> str:
    """
    Launch a subagent of the specified type to handle a task autonomously.

    This tool can call agents with mode='subagent' or mode='all'.

    Args:
        subagent_name: The name of the subagent to launch (e.g., code_reviewer)
        prompt: The task description/prompt for the subagent

    Returns:
        Structured result including tool calls, final response, and metadata
    """
    config = registry.get_config(subagent_name)
    if not config:
        return json.dumps({
            "error": f"Agent '{subagent_name}' not found",
            "available_agents": registry.list_callable_agents()
        }, indent=2)

    if config.mode not in ("subagent", "all"):
        return json.dumps({
            "error": f"Agent '{subagent_name}' has mode '{config.mode}', cannot be called by task tool"
        }, indent=2)

    from tools import ALL_TOOLS

    subagent = _create_subagent_agent(config, ctx, ALL_TOOLS)

    start_time = time.time()
    tool_calls = []
    tools_used = set()

    final_response = ""

    async for event in subagent.query_stream(prompt):
        if isinstance(event, ToolCallEvent):
            tool_calls.append({
                "tool": event.tool,
                "args": event.args,
                "timestamp": time.time()
            })
            sys.stdout.write(f"  [{subagent_name}] 🔧 Calling: {event.tool}\n")
            sys.stdout.flush()

        elif isinstance(event, ToolResultEvent):
            tools_used.add(event.tool)
            status = "✓" if not event.is_error else "✗"
            result_preview = str(event.result)[:100]
            if len(str(event.result)) > 100:
                result_preview += "..."
            sys.stdout.write(f"  [{subagent_name}] {status} {event.tool}: {result_preview}\n")
            sys.stdout.flush()

        elif isinstance(event, ThinkingEvent):
            thinking_preview = event.content[:50]
            if len(event.content) > 50:
                thinking_preview += "..."
            sys.stdout.write(f"  [{subagent_name}] 🧠 {thinking_preview}\n")
            sys.stdout.flush()

        elif isinstance(event, TextEvent):
            text_preview = event.content[:80]
            if len(event.content) > 80:
                text_preview += "..."
            sys.stdout.write(f"  [{subagent_name}] 💬 {text_preview}\n")
            sys.stdout.flush()

        elif isinstance(event, FinalResponseEvent):
            final_response = event.content
            final_preview = event.content[:100]
            if len(event.content) > 100:
                final_preview += "..."
            sys.stdout.write(f"  [{subagent_name}] ✅ Complete: {final_preview}\n")
            sys.stdout.flush()

    execution_time_ms = (time.time() - start_time) * 1000

    sys.stdout.write(f"  [{subagent_name}] ⏱️  Executed in {execution_time_ms:.0f}ms\n\n")
    sys.stdout.flush()

    result = SubagentResult(
        agent_name=subagent_name,
        prompt=prompt,
        final_response=final_response,
        tool_calls=tool_calls,
        execution_time_ms=execution_time_ms,
        mode=config.mode,
        tools_used=list(tools_used),
    )

    return json.dumps(result.__dict__, indent=2, default=str)
