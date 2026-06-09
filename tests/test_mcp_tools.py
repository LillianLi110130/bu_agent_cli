from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

if "croniter" not in sys.modules:
    croniter_stub = types.ModuleType("croniter")
    croniter_stub.croniter = object
    sys.modules["croniter"] = croniter_stub

from tools.mcp import mcp, mcp_status
from tools.sandbox import SandboxContext


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class FakeManager:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict]] = []

    def status(self) -> dict:
        return {"workspaceRoot": "/workspace", "servers": [], "clients": []}

    async def call_tool(self, server: str, tool: str, arguments: dict):
        self.calls.append((server, tool, arguments))
        return {"content": [{"type": "text", "text": "ok"}]}


def test_mcp_tool_schema_uses_claude_style_tool_alias() -> None:
    definition = mcp.definition
    properties = definition.parameters["properties"]

    assert "mcpServers" in definition.description
    assert "server" in properties
    assert "tool" in properties
    assert "arguments" in properties


@pytest.mark.anyio
async def test_mcp_tool_calls_manager(tmp_path: Path) -> None:
    ctx = SandboxContext.create(tmp_path)
    manager = FakeManager()
    ctx.mcp_manager = manager

    result = json.loads(
        await mcp.func(server="codegraph", mcp_tool="codegraph_status", arguments={}, ctx=ctx)
    )

    assert result["ok"] is True
    assert result["server"] == "codegraph"
    assert manager.calls == [("codegraph", "codegraph_status", {})]


@pytest.mark.anyio
async def test_mcp_status_returns_manager_status(tmp_path: Path) -> None:
    ctx = SandboxContext.create(tmp_path)
    ctx.mcp_manager = FakeManager()

    result = json.loads(await mcp_status.func(ctx=ctx))

    assert result["ok"] is True
    assert result["tool"] == "mcp_status"
