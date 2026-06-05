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

from tools.lsp import lsp_definition, lsp_status
from tools.sandbox import SandboxContext


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Path, int, int]] = []

    async def definition(self, path: Path, line: int, character: int):
        self.calls.append(("definition", path, line, character))
        return {
            "uri": path.as_uri(),
            "range": {
                "start": {"line": line, "character": character},
                "end": {"line": line, "character": character + 1},
            },
        }


class FakeManager:
    def __init__(self, client: FakeClient) -> None:
        self.client = client

    def status(self):
        return {"enabled": True, "clients": []}

    async def for_file(self, path: Path):
        return self.client


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
