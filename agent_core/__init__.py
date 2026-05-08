"""
A framework for building agentic applications with LLMs.

Example:
    from agent_core import Agent
    from agent_core.llm import ChatOpenAI
    from agent_core.tools import tool

    @tool("Add two numbers")
    async def add(a: int, b: int) -> int:
        return a + b

    agent = Agent(
        llm=ChatOpenAI(model="gpt-4o"),
        tools=[add],
    )

    result = await agent.query("What is 2 + 3?")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_core.agent import Agent, AgentConfig
    from agent_core.agent.registry import get_agent_registry
    from agent_core.observability import Laminar, observe, observe_debug

__all__ = [
    "Agent",
    "AgentConfig",
    "get_agent_registry",
    "Laminar",
    "observe",
    "observe_debug",
]

_AGENT_EXPORTS = {"Agent", "AgentConfig", "get_agent_registry"}
_OBSERVABILITY_EXPORTS = {"Laminar", "observe", "observe_debug"}


def __getattr__(name: str) -> Any:
    if name in _AGENT_EXPORTS:
        from agent_core.agent import Agent, AgentConfig, get_agent_registry

        exports = {
            "Agent": Agent,
            "AgentConfig": AgentConfig,
            "get_agent_registry": get_agent_registry,
        }
    elif name in _OBSERVABILITY_EXPORTS:
        from agent_core.observability import Laminar, observe, observe_debug

        exports = {
            "Laminar": Laminar,
            "observe": observe,
            "observe_debug": observe_debug,
        }
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    globals().update(exports)
    return exports[name]


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
