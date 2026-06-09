from __future__ import annotations

import asyncio
import json

import pytest

from agent_core.mcp.protocol import make_notification, make_request, read_message, write_message


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_mcp_protocol_reads_and_writes_newline_json() -> None:
    reader = asyncio.StreamReader()
    reader.feed_data(b'{"jsonrpc":"2.0","method":"ping"}\n')
    reader.feed_eof()

    message = await read_message(reader)

    assert message == {"jsonrpc": "2.0", "method": "ping"}


def test_mcp_protocol_builds_request_and_notification() -> None:
    assert make_request(1, "tools/list") == {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list",
    }
    assert make_notification("notifications/initialized") == {
        "jsonrpc": "2.0",
        "method": "notifications/initialized",
    }


@pytest.mark.anyio
async def test_mcp_protocol_write_message() -> None:
    chunks: list[bytes] = []

    class FakeWriter:
        def write(self, data: bytes) -> None:
            chunks.append(data)

        async def drain(self) -> None:
            return None

    await write_message(FakeWriter(), {"jsonrpc": "2.0", "id": 1, "result": {}})  # type: ignore[arg-type]

    assert chunks
    assert chunks[0].endswith(b"\n")
    assert json.loads(chunks[0]) == {"jsonrpc": "2.0", "id": 1, "result": {}}
