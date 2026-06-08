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

from tools.lsp import lsp, lsp_definition, lsp_status, lsp_workspace_symbols
from tools.sandbox import SandboxContext


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def test_only_unified_lsp_tool_is_publicly_registered() -> None:
    from tools import ALL_TOOLS

    tool_names = {tool.name for tool in ALL_TOOLS}

    assert "lsp" in tool_names
    assert "lsp_status" not in tool_names
    assert "lsp_diagnostics" not in tool_names
    assert "lsp_definition" not in tool_names
    assert "lsp_references" not in tool_names
    assert "lsp_hover" not in tool_names
    assert "lsp_document_symbols" not in tool_names
    assert "lsp_workspace_symbols" not in tool_names


def test_lsp_tool_schema_describes_operations_and_parameters() -> None:
    definition = lsp.definition
    properties = definition.parameters["properties"]

    assert "Use definition as Go to Definition" in definition.description
    assert "do not set character to 1" in definition.description
    assert "Do not call document_symbols repeatedly" in definition.description
    assert "not pagination" in definition.description
    assert "not '.' or a directory" in definition.description
    assert "workspace_symbols" in properties["operation"]["description"]
    assert "selects the language server" in properties["file_path"]["description"]
    assert "not '.' and not a directory" in properties["file_path"]["description"]
    assert "1-based line number" in properties["line"]["description"]
    assert "must point inside the target symbol text" in properties["character"]["description"]
    assert "symbol-index search, not grep" in properties["query"]["description"]


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Path, int, int]] = []
        self.workspace_symbol_queries: list[str] = []

    async def definition(self, path: Path, line: int, character: int):
        self.calls.append(("definition", path, line, character))
        return {
            "uri": path.as_uri(),
            "range": {
                "start": {"line": line, "character": character},
                "end": {"line": line, "character": character + 1},
            },
        }

    async def workspace_symbols(self, query: str):
        self.workspace_symbol_queries.append(query)
        return [
            {
                "name": "Agent",
                "kind": 5,
                "location": {
                    "uri": (Path.cwd() / "main.py").resolve().as_uri(),
                    "range": {
                        "start": {"line": 0, "character": 0},
                        "end": {"line": 0, "character": 5},
                    },
                },
            }
        ]


class FakeManager:
    def __init__(self, client: FakeClient) -> None:
        self.client = client
        self.enabled = True
        self.workspace_symbol_calls: list[Path | None] = []

    def status(self):
        return {"enabled": True, "clients": []}

    async def for_file(self, path: Path):
        return self.client

    async def workspace_symbols(self, query: str, path: Path):
        self.workspace_symbol_calls.append(path)
        return await self.client.workspace_symbols(query)


@pytest.mark.anyio
async def test_lsp_status_returns_manager_status(tmp_path: Path) -> None:
    ctx = SandboxContext.create(tmp_path)
    ctx.lsp_manager = SimpleNamespace(status=lambda: {"enabled": True, "clients": []})

    result = json.loads(await lsp_status.func(ctx=ctx))

    assert result["ok"] is True
    assert result["enabled"] is True


@pytest.mark.anyio
async def test_lsp_definition_converts_one_based_position(tmp_path: Path) -> None:
    target = tmp_path / "main.py"
    target.write_text("x = 1\n", encoding="utf-8")
    ctx = SandboxContext.create(tmp_path)
    client = FakeClient()
    ctx.lsp_manager = FakeManager(client)

    result = json.loads(await lsp_definition.func("main.py", 1, 3, ctx=ctx))

    assert result["ok"] is True
    assert client.calls[0][2:] == (0, 2)
    assert result["results"][0]["line"] == 1
    assert result["results"][0]["character"] == 3


@pytest.mark.anyio
async def test_unified_lsp_dispatches_position_operation(tmp_path: Path) -> None:
    target = tmp_path / "main.py"
    target.write_text("x = 1\n", encoding="utf-8")
    ctx = SandboxContext.create(tmp_path)
    client = FakeClient()
    ctx.lsp_manager = FakeManager(client)

    result = json.loads(
        await lsp.func(
            operation="definition",
            file_path="main.py",
            line=1,
            character=3,
            ctx=ctx,
        )
    )

    assert result["ok"] is True
    assert client.calls[0][2:] == (0, 2)


@pytest.mark.anyio
async def test_unified_lsp_validates_required_position_params(tmp_path: Path) -> None:
    ctx = SandboxContext.create(tmp_path)
    ctx.lsp_manager = FakeManager(FakeClient())

    result = json.loads(await lsp.func(operation="hover", file_path="main.py", ctx=ctx))

    assert result["ok"] is False
    assert "requires line, character" in result["error"]


@pytest.mark.anyio
async def test_lsp_workspace_symbols_uses_file_path_as_language_context(
    tmp_path: Path,
) -> None:
    target = tmp_path / "main.py"
    target.write_text("x = 1\n", encoding="utf-8")
    ctx = SandboxContext.create(tmp_path)
    client = FakeClient()
    manager = FakeManager(client)
    ctx.lsp_manager = manager

    result = json.loads(
        await lsp_workspace_symbols.func(query="Agent", ctx=ctx, file_path="main.py")
    )

    assert result["ok"] is True
    assert manager.workspace_symbol_calls == [target.resolve()]
    assert client.workspace_symbol_queries == ["Agent"]


@pytest.mark.anyio
async def test_lsp_workspace_symbols_requires_file_path(
    tmp_path: Path,
) -> None:
    ctx = SandboxContext.create(tmp_path)
    client = FakeClient()
    manager = FakeManager(client)
    ctx.lsp_manager = manager

    result = json.loads(await lsp.func(operation="workspace_symbols", query="Agent", ctx=ctx))

    assert result["ok"] is False
    assert "requires file_path" in result["error"]
    assert manager.workspace_symbol_calls == []


@pytest.mark.anyio
async def test_lsp_workspace_symbols_rejects_directory_file_path(
    tmp_path: Path,
) -> None:
    ctx = SandboxContext.create(tmp_path)
    manager = FakeManager(FakeClient())
    ctx.lsp_manager = manager

    result = json.loads(
        await lsp.func(
            operation="workspace_symbols",
            query="Agent",
            file_path=".",
            ctx=ctx,
        )
    )

    assert result["ok"] is False
    assert "concrete source file" in result["error"]
    assert "not '.' or a directory" in result["error"]
    assert manager.workspace_symbol_calls == []
