from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import pytest

from agent_core.lsp.client import LSPClient
from agent_core.lsp.types import LSPServerConfig


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_ensure_document_synced_sends_open_then_change(tmp_path: Path) -> None:
    target = tmp_path / "main.py"
    target.write_text("x = 1\n", encoding="utf-8")
    client = LSPClient(
        server_config=LSPServerConfig(
            name="python",
            command="pyright-langserver",
            args=["--stdio"],
            extensions=[".py"],
            language_id="python",
            root_markers=[],
        ),
        root_dir=tmp_path,
        request_timeout_seconds=1,
        diagnostics_settle_ms=1,
    )
    notifications: list[tuple[str, dict]] = []

    async def fake_start() -> None:
        return None

    async def fake_notify(method: str, params: dict | None = None) -> None:
        notifications.append((method, params or {}))

    client.start = fake_start  # type: ignore[method-assign]
    client.notify = fake_notify  # type: ignore[method-assign]

    await client.ensure_document_synced(target)
    await client.ensure_document_synced(target)
    target.write_text("x = 2\n", encoding="utf-8")
    await client.ensure_document_synced(target)

    assert [method for method, _params in notifications] == [
        "workspace/didChangeWatchedFiles",
        "textDocument/didOpen",
        "workspace/didChangeWatchedFiles",
        "textDocument/didChange",
    ]
    assert notifications[0][1]["changes"][0]["type"] == 1
    assert notifications[1][1]["textDocument"]["version"] == 1
    assert notifications[2][1]["changes"][0]["type"] == 2
    assert notifications[3][1]["textDocument"]["version"] == 2


@pytest.mark.anyio
async def test_save_document_sends_did_save(tmp_path: Path) -> None:
    target = tmp_path / "main.py"
    target.write_text("x = 1\n", encoding="utf-8")
    client = _make_client(tmp_path)
    notifications: list[tuple[str, dict]] = []

    async def fake_start() -> None:
        return None

    async def fake_notify(method: str, params: dict | None = None) -> None:
        notifications.append((method, params or {}))

    client.start = fake_start  # type: ignore[method-assign]
    client.notify = fake_notify  # type: ignore[method-assign]

    await client.save_document(target)

    assert notifications == [
        ("textDocument/didSave", {"textDocument": {"uri": target.resolve().as_uri()}})
    ]


@pytest.mark.anyio
async def test_diagnostics_saves_and_waits_for_new_publish(tmp_path: Path) -> None:
    target = tmp_path / "main.py"
    target.write_text("x = 1\n", encoding="utf-8")
    client = _make_client(tmp_path, request_timeout_seconds=1, diagnostics_settle_ms=0)
    notifications: list[str] = []

    async def fake_start() -> None:
        return None

    async def fake_notify(method: str, params: dict | None = None) -> None:
        notifications.append(method)
        if method == "textDocument/didSave":
            asyncio.create_task(
                client._handle_incoming_message(  # noqa: SLF001
                    {
                        "jsonrpc": "2.0",
                        "method": "textDocument/publishDiagnostics",
                        "params": {
                            "uri": target.resolve().as_uri(),
                            "version": 1,
                            "diagnostics": [
                                {
                                    "range": {
                                        "start": {"line": 0, "character": 0},
                                        "end": {"line": 0, "character": 1},
                                    },
                                    "severity": 1,
                                    "message": "sample",
                                }
                            ],
                        },
                    }
                )
            )

    client.start = fake_start  # type: ignore[method-assign]
    client.notify = fake_notify  # type: ignore[method-assign]

    diagnostics = await client.diagnostics(target)

    assert "textDocument/didSave" in notifications
    assert len(diagnostics) == 1
    assert diagnostics[0].message == "sample"


@pytest.mark.anyio
async def test_wait_for_diagnostics_accepts_unversioned_new_push(tmp_path: Path) -> None:
    target = tmp_path / "main.py"
    target.write_text("x = 1\n", encoding="utf-8")
    client = _make_client(tmp_path, request_timeout_seconds=1, diagnostics_settle_ms=0)
    wait_task = asyncio.create_task(client.wait_for_diagnostics(target, version=3))
    await asyncio.sleep(0)

    await client._handle_incoming_message(  # noqa: SLF001
        {
            "jsonrpc": "2.0",
            "method": "textDocument/publishDiagnostics",
            "params": {
                "uri": target.resolve().as_uri(),
                "diagnostics": [],
            },
        }
    )

    await asyncio.wait_for(wait_task, timeout=1)


@pytest.mark.anyio
async def test_wait_for_diagnostics_ignores_stale_versioned_push(tmp_path: Path) -> None:
    target = tmp_path / "main.py"
    target.write_text("x = 1\n", encoding="utf-8")
    client = _make_client(tmp_path, request_timeout_seconds=1, diagnostics_settle_ms=0)
    wait_task = asyncio.create_task(client.wait_for_diagnostics(target, version=3))
    await asyncio.sleep(0)

    await client._handle_incoming_message(  # noqa: SLF001
        {
            "jsonrpc": "2.0",
            "method": "textDocument/publishDiagnostics",
            "params": {
                "uri": target.resolve().as_uri(),
                "version": 2,
                "diagnostics": [],
            },
        }
    )
    await asyncio.sleep(0.05)
    assert not wait_task.done()

    await client._handle_incoming_message(  # noqa: SLF001
        {
            "jsonrpc": "2.0",
            "method": "textDocument/publishDiagnostics",
            "params": {
                "uri": target.resolve().as_uri(),
                "version": 3,
                "diagnostics": [],
            },
        }
    )

    await asyncio.wait_for(wait_task, timeout=1)


@pytest.mark.anyio
async def test_server_requests_are_handled_without_crashing(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    client.settings = {
        "python": {
            "analysis": {
                "typeCheckingMode": "basic",
            }
        }
    }
    responses: list[dict] = []

    async def fake_write(message: dict) -> None:
        responses.append(message)

    client._write_message = fake_write  # type: ignore[method-assign]

    await client._handle_incoming_message(  # noqa: SLF001
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "workspace/configuration",
            "params": {
                "items": [
                    {"section": "python.analysis"},
                    {"section": "python.missing"},
                    {},
                ]
            },
        }
    )
    await client._handle_incoming_message(  # noqa: SLF001
        {"jsonrpc": "2.0", "id": 2, "method": "workspace/workspaceFolders", "params": {}}
    )
    await client._handle_incoming_message(  # noqa: SLF001
        {"jsonrpc": "2.0", "id": 3, "method": "window/workDoneProgress/create", "params": {}}
    )
    await client._handle_incoming_message(  # noqa: SLF001
        {"jsonrpc": "2.0", "id": 4, "method": "unknown/request", "params": {}}
    )
    await client._handle_incoming_message(  # noqa: SLF001
        {"jsonrpc": "2.0", "id": 5, "method": "workspace/configuration", "params": "bad"}
    )

    assert responses[0]["result"] == [
        {"typeCheckingMode": "basic"},
        None,
        client.settings,
    ]
    assert responses[1]["result"] == [{"uri": tmp_path.resolve().as_uri(), "name": tmp_path.name}]
    assert responses[2]["result"] is None
    assert responses[3]["error"]["code"] == -32601
    assert responses[4]["result"] == []


@pytest.mark.anyio
async def test_invalid_message_is_logged_and_dropped(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = _make_client(tmp_path)

    with caplog.at_level(logging.WARNING, logger="agent_core.lsp.client"):
        await client._handle_incoming_message({"id": 1, "result": None})  # noqa: SLF001

    assert "dropping invalid message" in caplog.text
    assert "python" in caplog.text


def _make_client(
    tmp_path: Path,
    *,
    request_timeout_seconds: float = 1,
    diagnostics_settle_ms: int = 1,
) -> LSPClient:
    return LSPClient(
        server_config=LSPServerConfig(
            name="python",
            command="pyright-langserver",
            args=["--stdio"],
            extensions=[".py"],
            language_id="python",
            root_markers=[],
        ),
        root_dir=tmp_path,
        request_timeout_seconds=request_timeout_seconds,
        diagnostics_settle_ms=diagnostics_settle_ms,
    )
