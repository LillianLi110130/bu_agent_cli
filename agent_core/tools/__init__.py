"""
Tools framework for building agentic applications.

This module provides:
- @tool decorator for creating type-safe tools from functions
- Depends for dependency injection

Example:
    from agent_core.tools import tool, Depends

    # Define a simple tool
    @tool("Add two numbers together")
    async def add(a: int, b: int) -> int:
        return a + b

    # Define a tool with dependency injection
    async def get_db():
        return DatabaseConnection()

    @tool("Query the database")
    async def query(sql: str, db: Depends(get_db)) -> str:
        return await db.execute(sql)
"""

from agent_core.tools.decorator import Tool, ToolContent, tool
from agent_core.tools.depends import DependencyOverrides, Depends

__all__ = [
    "tool",
    "Tool",
    "ToolContent",
    "Depends",
    "DependencyOverrides",
]
