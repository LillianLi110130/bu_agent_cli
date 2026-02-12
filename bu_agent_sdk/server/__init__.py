"""
FastAPI HTTP Server for bu_agent_sdk.

This module provides a REST API wrapper around the bu_agent_sdk,
allowing the agent to be invoked via HTTP requests while keeping
the core SDK logic intact.

Example:
    from bu_agent_sdk.server import create_server
    from bu_agent_sdk.llm import ChatOpenAI
    from bu_agent_sdk.tools import tool

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
    # uvicorn bu_agent_sdk.server:app --reload
"""

from bu_agent_sdk.server.app import create_app, create_server, ServerConfig
from bu_agent_sdk.server.models import (
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
