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

from agent_core.mcp.formatter import format_tool_result
from tools import ALL_TOOLS
from tools.mcp import mcp_status
from tools.sandbox import SandboxContext


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class FakeManager:
    def status(self) -> dict:
        return {"workspaceRoot": "/workspace", "servers": [], "clients": []}


def test_unified_mcp_tool_is_not_exposed_by_default() -> None:
    assert "mcp" not in {tool.name for tool in ALL_TOOLS}


def test_mcp_tool_result_is_not_truncated_by_formatter() -> None:
    long_text = "x" * 20000
    result = json.loads(
        format_tool_result(
            server="codegraph",
            tool="codegraph_dump",
            result={"content": [{"type": "text", "text": long_text}]},
        )
    )

    assert result["ok"] is True
    assert "truncated" not in result
    assert result["result"]["content"][0]["text"] == long_text


@pytest.mark.anyio
async def test_mcp_status_returns_manager_status(tmp_path: Path) -> None:
    ctx = SandboxContext.create(tmp_path)
    ctx.mcp_manager = FakeManager()

    result = json.loads(await mcp_status.func(ctx=ctx))

    assert result["ok"] is True
    assert result["tool"] == "mcp_status"
