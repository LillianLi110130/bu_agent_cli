"""
FastAPI HTTP Server for agent_core.

This module provides a REST API wrapper around the agent_core,
allowing the agent to be invoked via HTTP requests while keeping
the core SDK logic intact.

Example:
    from agent_core.server import create_server
    from agent_core.llm import ChatOpenAI
    from agent_core.tools import tool

    @tool("Add two numbers")
    async def add(a: int, b: int) -> int:
        return a + b

    # Configure agent factory
    def agent_factory():
        return Agent(
            llm=ChatOpenAI(model="gpt-4o"),
            tools=[add],
        )

    app = create_server(agent_factory=agent_factory)

    # Run with uvicorn
    # uvicorn agent_core.server:app --reload
"""

from agent_core.server.app import create_app, create_server, ServerConfig
from agent_core.server.models import (
    QueryRequest,
    QueryResponse,
    StreamEvent,
    ErrorResponse,
)

__all__ = [
    "create_app",
    "create_server",
    "ServerConfig",
    "QueryRequest",
    "QueryResponse",
    "StreamEvent",
    "ErrorResponse",
]
