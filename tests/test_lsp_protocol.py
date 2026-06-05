from __future__ import annotations

import asyncio

import pytest

from agent_core.lsp.protocol import (
    LSPProtocolError,
    classify_message,
    encode_message,
    make_error_response,
    make_notification,
    make_request,
    make_response,
    read_message,
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def test_make_request_uses_json_rpc_2() -> None:
    request = make_request(7, "textDocument/definition", {"x": 1})

    assert request == {
        "jsonrpc": "2.0",
        "id": 7,
        "method": "textDocument/definition",
        "params": {"x": 1},
    }


def test_make_notification_has_no_id() -> None:
    notification = make_notification("initialized", {})

    assert notification["jsonrpc"] == "2.0"
    assert "id" not in notification


def test_make_response_uses_json_rpc_2() -> None:
    response = make_response("server-request-1", None)

    assert response == {
        "jsonrpc": "2.0",
        "id": "server-request-1",
        "result": None,
    }


def test_make_error_response_can_include_data() -> None:
    response = make_error_response(3, -32601, "Method not found", {"method": "x"})

    assert response == {
        "jsonrpc": "2.0",
        "id": 3,
        "error": {
            "code": -32601,
            "message": "Method not found",
            "data": {"method": "x"},
        },
    }


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (
            {"jsonrpc": "2.0", "id": 1, "method": "workspace/configuration"},
            ("request", 1),
        ),
        (
            {"jsonrpc": "2.0", "id": "a", "result": None},
            ("response", "a"),
        ),
        (
            {"jsonrpc": "2.0", "id": 2, "error": {"code": -32601}},
            ("response", 2),
        ),
        (
            {"jsonrpc": "2.0", "method": "textDocument/publishDiagnostics"},
            ("notification", "textDocument/publishDiagnostics"),
        ),
        (
            {"id": 1, "result": None},
            ("invalid", None),
        ),
    ],
)
def test_classify_message(message: dict[str, object], expected: tuple[str, object]) -> None:
    assert classify_message(message) == expected


@pytest.mark.anyio
async def test_encode_and_read_message_counts_utf8_bytes() -> None:
    message = {"jsonrpc": "2.0", "id": 1, "result": {"text": "中文"}}
    encoded = encode_message(message)
    reader = asyncio.StreamReader()
    reader.feed_data(encoded)
    reader.feed_eof()

    decoded = await read_message(reader)

    assert decoded == message
    assert b"Content-Length: 51" in encoded


@pytest.mark.anyio
async def test_read_message_returns_none_on_clean_eof() -> None:
    reader = asyncio.StreamReader()
    reader.feed_eof()

    assert await read_message(reader) is None


@pytest.mark.anyio
async def test_read_message_raises_protocol_error_for_malformed_header() -> None:
    reader = asyncio.StreamReader()
    reader.feed_data(b"Bad header\r\n\r\n{}")
    reader.feed_eof()

    with pytest.raises(LSPProtocolError, match="Malformed"):
        await read_message(reader)


@pytest.mark.anyio
async def test_read_message_raises_protocol_error_for_truncated_body() -> None:
    reader = asyncio.StreamReader()
    reader.feed_data(b"Content-Length: 10\r\n\r\n{}")
    reader.feed_eof()

    with pytest.raises(LSPProtocolError, match="Truncated"):
        await read_message(reader)
