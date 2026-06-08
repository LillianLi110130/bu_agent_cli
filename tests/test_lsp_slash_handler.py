from __future__ import annotations

from rich.console import Console

import pytest

from cli.lsp_handler import LspSlashHandler
from cli.slash_commands import SlashCommandRegistry


class FakeClient:
    def status(self) -> dict[str, object]:
        return {"name": "python", "root": "/workspace"}


class FakeManager:
    def __init__(self) -> None:
        self.started: list[str] = []

    def status(self) -> dict[str, object]:
        return {
            "enabled": True,
            "workspaceRoot": "/workspace",
            "servers": [
                {
                    "name": "python",
                    "command": "pyright-langserver",
                    "extensions": [".py"],
                }
            ],
            "clients": [],
        }

    async def start_server(self, name: str) -> FakeClient:
        self.started.append(name)
        return FakeClient()


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def test_slash_registry_contains_lsp_command() -> None:
    registry = SlashCommandRegistry()

    command = registry.get("lsp")

    assert command is not None
    assert command.usage == "/lsp [status|start <server>]"


@pytest.mark.anyio
async def test_lsp_slash_handler_starts_named_server() -> None:
    console = Console(record=True, force_terminal=False, width=120)
    manager = FakeManager()
    handler = LspSlashHandler(manager=manager, console=console)  # type: ignore[arg-type]

    handled = await handler.handle(["start", "python"])

    output = console.export_text()
    assert handled is True
    assert manager.started == ["python"]
    assert "LSP server 已启动" in output
    assert "python" in output


@pytest.mark.anyio
async def test_lsp_slash_handler_prints_status() -> None:
    console = Console(record=True, force_terminal=False, width=120)
    handler = LspSlashHandler(manager=FakeManager(), console=console)  # type: ignore[arg-type]

    handled = await handler.handle(["status"])

    output = console.export_text()
    assert handled is True
    assert "LSP Service" in output
    assert "pyright-langserver" in output
    assert "/lsp start python" in output
