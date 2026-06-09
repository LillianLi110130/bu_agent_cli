from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console

import pytest

from agent_core.mcp.types import MCPTool
from cli.mcp_handler import MCPSlashHandler
from cli.slash_commands import SlashCommandRegistry


class FakeClient:
    @property
    def tools(self):
        return [MCPTool("codegraph", "codegraph_status", "Status", {})]


class FakeManager:
    def __init__(self) -> None:
        self.started: list[str] = []
        self.stopped: list[str] = []
        self.reloaded = False
        self.workspace_root = Path("/workspace")
        self.state = "running"
        self.source = "project"

    def status(self):
        return {
            "workspaceRoot": "/workspace",
            "userConfigPath": "/home/.tg_agent/settings.json",
            "projectConfigPath": "/workspace/.tg_agent/mcp.json",
            "servers": [
                {
                    "name": "codegraph",
                    "state": self.state,
                    "source": self.source,
                    "type": "stdio",
                    "command": "codegraph",
                    "args": ["serve", "--mcp"],
                    "toolCount": 1,
                    "instructions": True,
                    "disabled": self.state == "disabled",
                }
            ],
            "clients": [{"name": "codegraph"}],
        }

    def list_tools(self, server_name=None):
        return [MCPTool("codegraph", "codegraph_status", "Status", {})]

    async def start_server(self, name: str):
        self.started.append(name)
        self.state = "running"
        return FakeClient()

    async def stop_server(self, name: str):
        self.stopped.append(name)
        self.state = "stopped"

    async def restart_server(self, name: str):
        self.stopped.append(name)
        self.started.append(name)
        return FakeClient()

    def reload_config(self):
        self.reloaded = True

    def instructions(self, server_name=None):
        return {"codegraph": "Prefer codegraph_explore."}


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def test_slash_registry_contains_mcp_command() -> None:
    registry = SlashCommandRegistry()

    command = registry.get("mcp")

    assert command is not None
    assert command.usage.startswith("/mcp")
    assert "instructions" in command.usage
    assert "start <server>" not in command.usage
    assert "reconnect <server>" in command.usage


@pytest.mark.anyio
async def test_mcp_slash_handler_prints_status() -> None:
    console = Console(record=True, force_terminal=False, width=120)
    handler = MCPSlashHandler(manager=FakeManager(), console=console)  # type: ignore[arg-type]

    handled = await handler.handle(["status"])

    output = console.export_text()
    assert handled is True
    assert "MCP Service" in output
    assert "codegraph" in output
    assert "project" in output
    assert ".tg_agent/mcp.json" in output
    assert "/mcp tools codegraph" in output
    assert "/mcp instructions codegraph" in output


@pytest.mark.anyio
async def test_mcp_slash_handler_enters_interactive_menu() -> None:
    console = Console(record=True, force_terminal=False, width=120)
    handler = MCPSlashHandler(manager=FakeManager(), console=console)  # type: ignore[arg-type]

    handled = await handler.handle([])

    output = console.export_text()
    assert handled is True
    assert handler.active is True
    assert "MCP Servers" in output
    assert "codegraph" in output


@pytest.mark.anyio
async def test_mcp_slash_handler_disabled_server_menu_only_offers_enable() -> None:
    console = Console(record=True, force_terminal=False, width=120)
    manager = FakeManager()
    manager.state = "disabled"
    handler = MCPSlashHandler(manager=manager, console=console)  # type: ignore[arg-type]
    await handler.handle([])

    handled = await handler.handle_input("1")

    output = console.export_text()
    assert handled is True
    assert "Enable" in output
    assert "Tools" not in output
    assert "Disable" not in output


@pytest.mark.anyio
async def test_mcp_slash_handler_prints_tools() -> None:
    console = Console(record=True, force_terminal=False, width=120)
    handler = MCPSlashHandler(manager=FakeManager(), console=console)  # type: ignore[arg-type]

    handled = await handler.handle(["tools", "codegraph"])

    output = console.export_text()
    assert handled is True
    assert "codegraph_status" in output


@pytest.mark.anyio
async def test_mcp_slash_handler_prints_instructions() -> None:
    console = Console(record=True, force_terminal=False, width=120)
    handler = MCPSlashHandler(manager=FakeManager(), console=console)  # type: ignore[arg-type]

    handled = await handler.handle(["instructions", "codegraph"])

    output = console.export_text()
    assert handled is True
    assert "Prefer codegraph_explore" in output


@pytest.mark.anyio
async def test_mcp_slash_handler_reconnects_server() -> None:
    console = Console(record=True, force_terminal=False, width=120)
    manager = FakeManager()
    handler = MCPSlashHandler(manager=manager, console=console)  # type: ignore[arg-type]

    handled = await handler.handle(["reconnect", "codegraph"])

    output = console.export_text()
    assert handled is True
    assert manager.stopped == ["codegraph"]
    assert manager.started == ["codegraph"]
    assert "MCP server 已重启" in output


@pytest.mark.anyio
async def test_mcp_slash_handler_disables_server_in_project_config(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    project_dir = workspace / ".tg_agent"
    project_dir.mkdir(parents=True)
    config_path = project_dir / "mcp.json"
    config_path.write_text(
        json.dumps({"mcpServers": {"codegraph": {"command": "codegraph"}}}),
        encoding="utf-8",
    )
    console = Console(record=True, force_terminal=False, width=120)
    manager = FakeManager()
    manager.workspace_root = workspace
    handler = MCPSlashHandler(manager=manager, console=console)  # type: ignore[arg-type]

    handled = await handler.handle(["disable", "codegraph"])

    data = json.loads(config_path.read_text(encoding="utf-8"))
    output = console.export_text()
    assert handled is True
    assert manager.stopped == ["codegraph"]
    assert manager.reloaded is True
    assert data["mcpServers"]["codegraph"]["disabled"] is True
    assert "MCP server 已禁用" in output


@pytest.mark.anyio
async def test_mcp_slash_handler_enables_and_starts_server_in_project_config(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    project_dir = workspace / ".tg_agent"
    project_dir.mkdir(parents=True)
    config_path = project_dir / "mcp.json"
    config_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "codegraph": {
                        "command": "codegraph",
                        "disabled": True,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    console = Console(record=True, force_terminal=False, width=120)
    manager = FakeManager()
    manager.workspace_root = workspace
    manager.state = "disabled"
    handler = MCPSlashHandler(manager=manager, console=console)  # type: ignore[arg-type]

    handled = await handler.handle(["enable", "codegraph"])

    data = json.loads(config_path.read_text(encoding="utf-8"))
    output = console.export_text()
    assert handled is True
    assert manager.started == ["codegraph"]
    assert manager.reloaded is True
    assert data["mcpServers"]["codegraph"]["disabled"] is False
    assert "MCP server 已启用并启动" in output


@pytest.mark.anyio
async def test_mcp_slash_handler_enable_reports_already_running(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    project_dir = workspace / ".tg_agent"
    project_dir.mkdir(parents=True)
    config_path = project_dir / "mcp.json"
    config_path.write_text(
        json.dumps({"mcpServers": {"codegraph": {"command": "codegraph"}}}),
        encoding="utf-8",
    )
    console = Console(record=True, force_terminal=False, width=120)
    manager = FakeManager()
    manager.workspace_root = workspace
    manager.state = "running"
    handler = MCPSlashHandler(manager=manager, console=console)  # type: ignore[arg-type]

    handled = await handler.handle(["enable", "codegraph"])

    output = console.export_text()
    assert handled is True
    assert manager.started == ["codegraph"]
    assert "MCP server 已启用，且已在运行" in output
    assert "MCP server 已启用并启动" not in output


@pytest.mark.anyio
async def test_mcp_interactive_enable_starts_and_returns_to_normal_actions(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    project_dir = workspace / ".tg_agent"
    project_dir.mkdir(parents=True)
    config_path = project_dir / "mcp.json"
    config_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "codegraph": {
                        "command": "codegraph",
                        "disabled": True,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    console = Console(record=True, force_terminal=False, width=120)
    manager = FakeManager()
    manager.workspace_root = workspace
    manager.state = "disabled"
    handler = MCPSlashHandler(manager=manager, console=console)  # type: ignore[arg-type]
    await handler.handle([])
    await handler.handle_input("1")

    handled = await handler.handle_input("1")

    output = console.export_text()
    assert handled is True
    assert manager.started == ["codegraph"]
    assert "MCP server 已启用并启动" in output
    assert "Tools" in output
    assert "Reconnect" in output
