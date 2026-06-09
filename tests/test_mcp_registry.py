from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

if "croniter" not in sys.modules:
    croniter_stub = types.ModuleType("croniter")
    croniter_stub.croniter = object
    sys.modules["croniter"] = croniter_stub

from agent_core.mcp.registry import sync_mcp_dynamic_tools
from agent_core.mcp.types import MCPTool
from agent_core.tools import tool
from tools.sandbox import SandboxContext, get_sandbox_context


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@tool("Static tool")
async def static_tool() -> str:
    return "static"


class FakeManager:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict]] = []

    def list_tools(self):
        return [
            MCPTool(
                server_name="codegraph",
                name="codegraph_status",
                description="Return CodeGraph status.",
                input_schema={"type": "object"},
            )
        ]

    async def call_tool(self, server: str, tool_name: str, arguments: dict):
        self.calls.append((server, tool_name, arguments))
        return {"content": [{"type": "text", "text": "ok"}]}


@pytest.mark.anyio
async def test_sync_mcp_dynamic_tools_adds_proxy_tool(tmp_path: Path) -> None:
    ctx = SandboxContext.create(tmp_path)
    manager = FakeManager()
    ctx.mcp_manager = manager
    agent = SimpleNamespace(
        tools=[static_tool],
        _tool_map={static_tool.name: static_tool},
        dependency_overrides={get_sandbox_context: lambda: ctx},
    )

    sync_mcp_dynamic_tools(agent, ctx)

    assert "static_tool" in agent._tool_map
    assert "mcp__codegraph__codegraph_status" in agent._tool_map

    proxy = agent._tool_map["mcp__codegraph__codegraph_status"]
    result = json.loads(
        await proxy.execute(
            _overrides=agent.dependency_overrides,
            arguments={"verbose": True},
        )
    )

    assert result["ok"] is True
    assert manager.calls == [("codegraph", "codegraph_status", {"verbose": True})]
