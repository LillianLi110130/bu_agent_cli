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

from agent_core.agent import Agent, AgentConfig, get_agent_registry
from agent_core.observability import Laminar, observe, observe_debug

__all__ = [
    "Agent",
    "AgentConfig",
    "get_agent_registry",
    "Laminar",
    "observe",
    "observe_debug",
]
