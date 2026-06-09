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
    max_text_chars: int = 12000,
) -> str:
    truncated = False
    compact = result
    if isinstance(result, dict):
        compact = dict(result)
        content = compact.get("content")
        if isinstance(content, list):
            new_content = []
            remaining = max_text_chars
            for item in content:
                if not isinstance(item, dict):
                    new_content.append(item)
                    continue
                copied = dict(item)
                text = copied.get("text")
                if isinstance(text, str) and len(text) > remaining:
                    copied["text"] = text[: max(0, remaining)]
                    truncated = True
                    remaining = 0
                elif isinstance(text, str):
                    remaining -= len(text)
                new_content.append(copied)
            compact["content"] = new_content
    return success_payload(
        tool="mcp",
        server=server,
        mcpTool=tool,
        result=compact,
        truncated=truncated,
    )
