"""JSON-RPC 2.0 and LSP stdio framing helpers."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Literal

MessageKind = Literal["request", "response", "notification", "invalid"]


class LSPProtocolError(Exception):
    """Raised when the LSP wire framing or JSON-RPC envelope is malformed."""


def make_request(request_id: int, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": params or {},
    }


def make_notification(method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "method": method,
        "params": params or {},
    }


def make_response(request_id: Any, result: Any) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": result,
    }


def make_error_response(
    request_id: Any,
    code: int,
    message: str,
    data: Any = None,
) -> dict[str, Any]:
    error: dict[str, Any] = {
        "code": code,
        "message": message,
    }
    if data is not None:
        error["data"] = data
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": error,
    }


def classify_message(message: dict[str, Any]) -> tuple[MessageKind, Any]:
    if message.get("jsonrpc") != "2.0":
        return "invalid", None

    has_id = "id" in message
    has_method = "method" in message
    has_result = "result" in message
    has_error = "error" in message

    if has_id and has_method:
        return "request", message.get("id")
    if has_id and (has_result or has_error):
        return "response", message.get("id")
    if has_method and not has_id:
        return "notification", message.get("method")
    return "invalid", None


def encode_message(message: dict[str, Any]) -> bytes:
    body = json.dumps(message, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
    return header + body


async def read_message(reader: asyncio.StreamReader) -> dict[str, Any] | None:
    headers: dict[str, str] = {}
    header_bytes = 0
    while True:
        try:
            line = await reader.readuntil(b"\r\n")
        except asyncio.IncompleteReadError as exc:
            if not exc.partial and not headers:
                return None
            raise LSPProtocolError(
                f"LSP stream closed while reading headers: partial={exc.partial!r}"
            ) from exc

        header_bytes += len(line)
        if header_bytes > 8192:
            raise LSPProtocolError("LSP header block exceeded 8 KiB")

        line = line[:-2]
        if not line:
            break

        try:
            decoded = line.decode("ascii").strip()
        except UnicodeDecodeError as exc:
            raise LSPProtocolError(f"Non-ASCII LSP header: {line!r}") from exc
        if ":" not in decoded:
            raise LSPProtocolError(f"Malformed LSP header line: {line!r}")
        key, value = decoded.split(":", 1)
        headers[key.strip().lower()] = value.strip()

    raw_length = headers.get("content-length")
    if raw_length is None:
        raise LSPProtocolError("LSP message missing Content-Length header")
    try:
        content_length = int(raw_length)
    except ValueError as exc:
        raise LSPProtocolError(f"Invalid LSP Content-Length: {raw_length}") from exc
    if content_length < 0 or content_length > 64 * 1024 * 1024:
        raise LSPProtocolError(f"Unreasonable LSP Content-Length: {content_length}")

    try:
        body = await reader.readexactly(content_length)
    except asyncio.IncompleteReadError as exc:
        raise LSPProtocolError(
            f"Truncated LSP body: expected {content_length} bytes, got {len(exc.partial)}"
        ) from exc
    try:
        payload = json.loads(body.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise LSPProtocolError("LSP message body must be UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise LSPProtocolError(f"LSP message body is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise LSPProtocolError("LSP message body must be a JSON object")
    return payload
