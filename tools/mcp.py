"""MCP-backed external tool invocation."""

from __future__ import annotations

from typing import Annotated

from agent_core.mcp.formatter import error_payload, format_status
from agent_core.tools import Depends, tool
from tools.sandbox import SandboxContext, get_sandbox_context


@tool("Show configured MCP servers, status, and discovered tools.", context_policy="trim")
async def mcp_status(
    ctx: Annotated[SandboxContext, Depends(get_sandbox_context)],
) -> str:
    manager = getattr(ctx, "mcp_manager", None)
    if manager is None:
        return error_payload("MCP manager is not initialized.", tool="mcp_status")
    return format_status(manager.status())
