"""Dynamic registration of discovered MCP tools into an Agent."""

from __future__ import annotations

import re
from typing import Annotated, Any

from agent_core.mcp.formatter import error_payload, format_tool_result
from agent_core.mcp.instructions import sync_mcp_instruction_reminders
from agent_core.tools import Depends, Tool, tool
from tools.sandbox import SandboxContext, get_sandbox_context

_DYNAMIC_PREFIX = "mcp__"
_NAME_RE = re.compile(r"[^A-Za-z0-9_]+")


def bind_mcp_dynamic_tools(agent: Any, ctx: SandboxContext) -> None:
    """Keep an agent's dynamic MCP proxy tools in sync with the MCP manager."""
    manager = getattr(ctx, "mcp_manager", None)
    if manager is None:
        return

    def sync() -> None:
        sync_mcp_dynamic_tools(agent, ctx)
        sync_mcp_instruction_reminders(agent, ctx)

    manager.on_tools_changed = sync
    sync()


def sync_mcp_dynamic_tools(agent: Any, ctx: SandboxContext) -> None:
    manager = getattr(ctx, "mcp_manager", None)
    if manager is None:
        return

    static_tools = [
        existing
        for existing in getattr(agent, "tools", [])
        if not getattr(existing, "name", "").startswith(_DYNAMIC_PREFIX)
    ]
    dynamic_tools = [_build_proxy_tool(tool_info) for tool_info in manager.list_tools()]
    agent.tools = [*static_tools, *dynamic_tools]
    agent._tool_map = {tool_item.name: tool_item for tool_item in agent.tools}


def _build_proxy_tool(tool_info) -> Tool:
    server_name = tool_info.server_name
    mcp_tool_name = tool_info.name
    dynamic_name = f"{_DYNAMIC_PREFIX}{_sanitize(server_name)}__{_sanitize(mcp_tool_name)}"
    description = _build_description(tool_info)

    async def proxy(
        ctx: Annotated[SandboxContext, Depends(get_sandbox_context)],
        arguments: dict[str, Any] | None = None,
    ) -> str:
        manager = getattr(ctx, "mcp_manager", None)
        if manager is None:
            return error_payload(
                "MCP manager is not initialized.",
                tool=dynamic_name,
                server=server_name,
                mcpTool=mcp_tool_name,
            )
        try:
            result = await manager.call_tool(server_name, mcp_tool_name, arguments or {})
            return format_tool_result(server=server_name, tool=mcp_tool_name, result=result)
        except Exception as exc:
            return error_payload(
                str(exc) or exc.__class__.__name__,
                tool=dynamic_name,
                server=server_name,
                mcpTool=mcp_tool_name,
            )

    proxy.__name__ = dynamic_name
    return tool(
        description,
        name=dynamic_name,
        context_policy="trim",
        context_max_inline_chars=12000,
    )(proxy)


def _build_description(tool_info) -> str:
    base = tool_info.description.strip() or f"Call MCP tool {tool_info.name}."
    if len(base) > 1200:
        base = base[:1200].rstrip() + "..."
    return (
        f"MCP tool from server '{tool_info.server_name}': {tool_info.name}. "
        f"{base} Pass the MCP tool input as a JSON object in the 'arguments' parameter."
    )


def _sanitize(name: str) -> str:
    cleaned = _NAME_RE.sub("_", name).strip("_")
    return cleaned or "tool"
