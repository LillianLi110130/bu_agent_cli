"""Format MCP responses into compact JSON payloads."""

from __future__ import annotations

import json
from typing import Any


def success_payload(**fields: Any) -> str:
    return json.dumps({"ok": True, **fields}, ensure_ascii=False, indent=2)


def error_payload(error: str, **fields: Any) -> str:
    return json.dumps({"ok": False, **fields, "error": error}, ensure_ascii=False, indent=2)


def format_status(status: dict[str, Any]) -> str:
    return success_payload(tool="mcp_status", **status)


def format_tool_result(
    *,
    server: str,
    tool: str,
    result: Any,
) -> str:
    """Return the full MCP tool result.

    Large MCP outputs are handled by the agent tool-context policy, which can
    persist the complete response as an artifact and keep only a preview in the
    model context. Truncating here would lose data before that layer can save it.
    """
    return success_payload(
        tool="mcp",
        server=server,
        mcpTool=tool,
        result=result,
    )
