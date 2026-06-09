"""JSON-RPC 2.0 helpers for MCP stdio transport."""

from __future__ import annotations

import asyncio
import json
from typing import Any


class MCPProtocolError(RuntimeError):
    """Raised when an MCP message cannot be decoded."""


def make_request(request_id: int, method: str, params: Any | None = None) -> dict[str, Any]:
    message: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
    if params is not None:
        message["params"] = params
    return message


def make_notification(method: str, params: Any | None = None) -> dict[str, Any]:
    message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        message["params"] = params
    return message


async def read_message(reader: asyncio.StreamReader) -> dict[str, Any] | None:
    line = await reader.readline()
    if line == b"":
        return None
    try:
        message = json.loads(line.decode("utf-8"))
    except Exception as exc:
        raise MCPProtocolError(f"Invalid MCP JSON message: {exc}") from exc
    if not isinstance(message, dict):
        raise MCPProtocolError("MCP message must be a JSON object")
    if message.get("jsonrpc") != "2.0":
        raise MCPProtocolError("MCP message must use JSON-RPC 2.0")
    return message


async def write_message(writer: asyncio.StreamWriter, message: dict[str, Any]) -> None:
    payload = json.dumps(message, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    writer.write(payload + b"\n")
    await writer.drain()
