"""MCP-backed external tool invocation."""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, Field

from agent_core.mcp.formatter import error_payload, format_status, format_tool_result
from agent_core.tools import Depends, tool
from tools.sandbox import SandboxContext, get_sandbox_context


class MCPParams(BaseModel):
    """Arguments for the unified MCP tool."""

    server: str = Field(
        description="Configured MCP server name, for example 'codegraph'.",
    )
    mcp_tool: str = Field(
        alias="tool",
        description=(
            "MCP tool name exposed by that server, for example 'codegraph_status' "
            "or 'codegraph_explore'. Use /mcp tools to inspect available tools."
        ),
    )
    arguments: dict[str, Any] | None = Field(
        default=None,
        description="JSON object arguments to pass to the MCP tool. Use {} when no arguments are needed.",
    )


@tool(
    (
        "Call a configured MCP server tool. MCP servers are configured with Claude Code-style "
        "mcpServers and are started automatically at agent startup unless disabled. Use this for "
        "external MCP capabilities such as CodeGraph. Inspect available servers and tools with "
        "/mcp status and /mcp tools."
    ),
    context_policy="trim",
    context_max_inline_chars=12000,
    args_schema=MCPParams,
)
async def mcp(
    server: str,
    mcp_tool: Annotated[str, Field(alias="tool")],
    ctx: Annotated[SandboxContext, Depends(get_sandbox_context)],
    arguments: dict[str, Any] | None = None,
) -> str:
    manager = getattr(ctx, "mcp_manager", None)
    if manager is None:
        return error_payload("MCP manager is not initialized.", tool="mcp", server=server)
    try:
        result = await manager.call_tool(server, mcp_tool, arguments or {})
        return format_tool_result(server=server, tool=mcp_tool, result=result)
    except Exception as exc:
        return error_payload(
            str(exc) or exc.__class__.__name__,
            tool="mcp",
            server=server,
            mcpTool=mcp_tool,
        )


@tool("Show configured MCP servers, status, and discovered tools.", context_policy="trim")
async def mcp_status(
    ctx: Annotated[SandboxContext, Depends(get_sandbox_context)],
) -> str:
    manager = getattr(ctx, "mcp_manager", None)
    if manager is None:
        return error_payload("MCP manager is not initialized.", tool="mcp_status")
    return format_status(manager.status())
