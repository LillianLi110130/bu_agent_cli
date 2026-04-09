from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pytest

from cli.im_bridge import FileBridgeStore, get_bridge_store
from tools import SandboxContext, get_sandbox_context
from tools.message import message


@pytest.fixture
def workspace_root() -> Path:
    root = Path(".pytest_tmp") / f"message-tool-{uuid.uuid4().hex}"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    try:
        yield root.resolve()
    finally:
        if root.exists():
            shutil.rmtree(root)


def _build_overrides(workspace_root: Path) -> tuple[SandboxContext, FileBridgeStore, dict]:
    ctx = SandboxContext.create(workspace_root)
    store = FileBridgeStore(workspace_root, session_binding_id="message-tool-test")
    store.initialize()
    overrides = {
        get_sandbox_context: lambda: ctx,
        get_bridge_store: lambda: store,
    }
    return ctx, store, overrides


@pytest.mark.asyncio
async def test_message_tool_enqueues_text_outbound_event(workspace_root: Path) -> None:
    _ctx, store, overrides = _build_overrides(workspace_root)

    result = await message.execute(
        _overrides=overrides,
        action="text",
        text="hello outbound",
    )

    assert "hello outbound" in result
    event = store.claim_next_pending_outbound()
    assert event is not None
    assert event.action == "text"
    assert event.text == "hello outbound"


@pytest.mark.asyncio
async def test_message_tool_rejects_unsupported_attachment_suffix(workspace_root: Path) -> None:
    _ctx, store, overrides = _build_overrides(workspace_root)
    attachment_path = workspace_root / "payload.exe"
    attachment_path.write_bytes(b"bad")

    result = await message.execute(
        _overrides=overrides,
        action="attachment",
        file_path=str(attachment_path),
    )

    assert "Unsupported attachment format" in result
    assert store.claim_next_pending_outbound() is None


@pytest.mark.asyncio
async def test_message_tool_rejects_attachment_larger_than_five_mb(workspace_root: Path) -> None:
    _ctx, store, overrides = _build_overrides(workspace_root)
    attachment_path = workspace_root / "big.txt"
    attachment_path.write_bytes(b"x" * (5 * 1024 * 1024 + 1))

    result = await message.execute(
        _overrides=overrides,
        action="attachment",
        file_path=str(attachment_path),
    )

    assert "File too large" in result
    assert "5 MB" in result
    assert store.claim_next_pending_outbound() is None


@pytest.mark.asyncio
async def test_message_tool_enqueues_valid_attachment_outbound_event(workspace_root: Path) -> None:
    _ctx, store, overrides = _build_overrides(workspace_root)
    attachment_path = workspace_root / "report.pdf"
    attachment_path.write_bytes(b"%PDF-1.4 content")

    result = await message.execute(
        _overrides=overrides,
        action="attachment",
        file_path=str(attachment_path),
    )

    assert "report.pdf" in result
    event = store.claim_next_pending_outbound()
    assert event is not None
    assert event.action == "attachment"
    assert event.file_path == str(attachment_path)
    assert event.file_name == "report.pdf"
    assert event.mime_type == "application/pdf"
    assert event.file_size == attachment_path.stat().st_size
