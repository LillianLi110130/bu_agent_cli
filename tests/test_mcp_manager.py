from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest

from agent_core.mcp.config import parse_mcp_config
from agent_core.mcp.config import set_mcp_server_disabled
from agent_core.mcp.manager import MCPManager


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _write_fake_mcp_server(path: Path) -> None:
    path.write_text(
        textwrap.dedent(
            """
            import json
            import sys

            for line in sys.stdin:
                message = json.loads(line)
                method = message.get("method")
                request_id = message.get("id")
                if method == "initialize":
                    print(json.dumps({
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {"tools": {}},
                            "serverInfo": {"name": "fake"},
                            "instructions": "Prefer fake tools."
                        }
                    }), flush=True)
                elif method == "notifications/initialized":
                    pass
                elif method == "tools/list":
                    print(json.dumps({
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "tools": [{
                                "name": "echo",
                                "description": "Echo input",
                                "inputSchema": {"type": "object"}
                            }]
                        }
                    }), flush=True)
                elif method == "tools/call":
                    print(json.dumps({
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [{
                                "type": "text",
                                "text": json.dumps(message.get("params", {}).get("arguments", {}))
                            }]
                        }
                    }), flush=True)
                elif method == "shutdown":
                    print(json.dumps({"jsonrpc": "2.0", "id": request_id, "result": None}), flush=True)
                elif method == "exit":
                    break
            """
        ),
        encoding="utf-8",
    )


@pytest.mark.anyio
async def test_mcp_manager_starts_server_and_calls_tool(tmp_path: Path) -> None:
    script = tmp_path / "fake_mcp.py"
    _write_fake_mcp_server(script)
    config = parse_mcp_config(
        {
            "mcpServers": {
                "fake": {
                    "command": sys.executable,
                    "args": [str(script)],
                }
            }
        }
    )
    manager = MCPManager(workspace_root=tmp_path, config=config)

    await manager.start_enabled_servers()
    result = await manager.call_tool("fake", "echo", {"hello": "world"})

    assert result["content"][0]["text"] == '{"hello": "world"}'
    assert manager.list_tools()[0].name == "echo"
    assert manager.status()["servers"][0]["state"] == "running"
    await manager.shutdown_all()


@pytest.mark.anyio
async def test_mcp_manager_skips_disabled_server(tmp_path: Path) -> None:
    config = parse_mcp_config(
        {
            "mcpServers": {
                "fake": {
                    "command": sys.executable,
                    "disabled": True,
                }
            }
        }
    )
    manager = MCPManager(workspace_root=tmp_path, config=config)

    await manager.start_enabled_servers()

    assert manager.status()["servers"][0]["state"] == "disabled"
    assert manager.list_tools() == []


@pytest.mark.anyio
async def test_mcp_manager_reload_clears_disabled_state_after_enable(tmp_path: Path) -> None:
    project_dir = tmp_path / ".tg_agent"
    project_dir.mkdir()
    (project_dir / "mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "fake": {
                        "command": sys.executable,
                        "disabled": True,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    manager = MCPManager.from_settings(tmp_path)
    assert manager.status()["servers"][0]["state"] == "disabled"

    set_mcp_server_disabled(
        workspace_root=tmp_path,
        server_name="fake",
        source="project",
        disabled=False,
    )
    manager.reload_config()

    assert manager.status()["servers"][0]["state"] == "configured"


@pytest.mark.anyio
async def test_mcp_manager_failed_server_does_not_block_others(tmp_path: Path) -> None:
    script = tmp_path / "fake_mcp.py"
    _write_fake_mcp_server(script)
    config = parse_mcp_config(
        {
            "mcpServers": {
                "missing": {"command": "definitely-missing-mcp-command"},
                "fake": {"command": sys.executable, "args": [str(script)]},
            }
        }
    )
    manager = MCPManager(workspace_root=tmp_path, config=config)

    await manager.start_enabled_servers()
    states = {server["name"]: server["state"] for server in manager.status()["servers"]}

    assert states["missing"] == "failed"
    assert states["fake"] == "running"
    await manager.shutdown_all()
