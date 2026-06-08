from __future__ import annotations

import asyncio
from types import SimpleNamespace
from pathlib import Path

import pytest

import agent_core.lsp as lsp_module
from agent_core.lsp import shutdown_lsp_manager
from agent_core.lsp.client import LSPClient
from agent_core.lsp.config import parse_lsp_config
from agent_core.lsp.manager import LSPManager


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def test_manager_resolves_root_marker_without_crossing_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    package = workspace / "packages" / "web"
    source = package / "src"
    source.mkdir(parents=True)
    (package / "package.json").write_text("{}", encoding="utf-8")
    target = source / "App.tsx"
    target.write_text("export const App = () => null\n", encoding="utf-8")
    config = parse_lsp_config(
        {
            "enabled": True,
            "servers": {
                "typescript": {
                    "command": "typescript-language-server",
                    "args": ["--stdio"],
                    "extensions": [".ts", ".tsx", ".js", ".jsx"],
                    "languageId": "typescript",
                    "rootMarkers": ["tsconfig.json", "package.json", ".git"],
                }
            },
        }
    )
    manager = LSPManager(workspace_root=workspace, config=config)
    server = manager._server_for_path(target)  # noqa: SLF001

    root = manager._resolve_root(target, server)  # noqa: SLF001

    assert root == package.resolve()


def test_manager_rejects_unknown_extension(tmp_path: Path) -> None:
    config = parse_lsp_config({"enabled": True})
    manager = LSPManager(workspace_root=tmp_path, config=config)

    with pytest.raises(RuntimeError, match="No LSP server configured"):
        manager._server_for_path(tmp_path / "notes.txt")  # noqa: SLF001


def test_manager_selects_default_java_server(tmp_path: Path) -> None:
    config = parse_lsp_config({"enabled": True})
    manager = LSPManager(workspace_root=tmp_path, config=config)

    server = manager._server_for_path(tmp_path / "Main.java")  # noqa: SLF001

    assert server.name == "java"
    assert server.command == "jdtls"


@pytest.mark.anyio
async def test_manager_auto_start_false_blocks_new_client(tmp_path: Path) -> None:
    target = tmp_path / "main.py"
    target.write_text("x = 1\n", encoding="utf-8")
    config = parse_lsp_config({"enabled": True, "autoStart": False})
    manager = LSPManager(workspace_root=tmp_path, config=config)

    with pytest.raises(RuntimeError, match="autoStart is disabled"):
        await manager.for_file(target)


@pytest.mark.anyio
async def test_manager_start_server_ignores_auto_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = parse_lsp_config({"enabled": True, "autoStart": False})
    manager = LSPManager(workspace_root=tmp_path, config=config)
    starts = 0

    async def fake_start(self: LSPClient) -> None:
        nonlocal starts
        starts += 1

    monkeypatch.setattr(LSPClient, "start", fake_start)

    first = await manager.start_server("python")
    second = await manager.start_server("python")

    assert first is second
    assert starts == 1
    assert manager.status()["clients"]


@pytest.mark.anyio
async def test_manager_start_server_rejects_unknown_name(tmp_path: Path) -> None:
    manager = LSPManager(workspace_root=tmp_path, config=parse_lsp_config({"enabled": True}))

    with pytest.raises(RuntimeError, match="No LSP server configured with name"):
        await manager.start_server("missing")


@pytest.mark.anyio
async def test_manager_deduplicates_concurrent_spawns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "main.py"
    target.write_text("x = 1\n", encoding="utf-8")
    config = parse_lsp_config({"enabled": True})
    manager = LSPManager(workspace_root=tmp_path, config=config)
    starts = 0

    async def fake_start(self: LSPClient) -> None:
        nonlocal starts
        starts += 1
        await asyncio.sleep(0.01)

    monkeypatch.setattr(LSPClient, "start", fake_start)

    first, second = await asyncio.gather(manager.for_file(target), manager.for_file(target))

    assert first is second
    assert starts == 1


@pytest.mark.anyio
async def test_manager_marks_failed_spawn_as_broken(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "main.py"
    target.write_text("x = 1\n", encoding="utf-8")
    config = parse_lsp_config({"enabled": True})
    manager = LSPManager(workspace_root=tmp_path, config=config)
    starts = 0

    async def fake_start(self: LSPClient) -> None:
        nonlocal starts
        starts += 1
        raise RuntimeError("boom")

    monkeypatch.setattr(LSPClient, "start", fake_start)

    with pytest.raises(RuntimeError, match="boom"):
        await manager.for_file(target)
    with pytest.raises(RuntimeError, match="marked broken"):
        await manager.for_file(target)

    assert starts == 1
    assert manager.status()["broken"]


@pytest.mark.anyio
async def test_manager_passes_server_settings_init_options_and_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "main.py"
    target.write_text("x = 1\n", encoding="utf-8")
    config = parse_lsp_config(
        {
            "enabled": True,
            "servers": {
                "python": {
                    "env": {"TEST_LSP_ENV": "1"},
                    "settings": {"python": {"analysis": {"diagnosticMode": "workspace"}}},
                    "initializationOptions": {"cache": True},
                }
            },
        }
    )
    manager = LSPManager(workspace_root=tmp_path, config=config)

    async def fake_start(self: LSPClient) -> None:
        return None

    monkeypatch.setattr(LSPClient, "start", fake_start)

    client = await manager.for_file(target)

    assert client.env == {"TEST_LSP_ENV": "1"}
    assert client.settings == {"python": {"analysis": {"diagnosticMode": "workspace"}}}
    assert client.initialization_options == {"cache": True}


@pytest.mark.anyio
async def test_manager_shutdown_all_closes_cached_clients(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "main.py"
    target.write_text("x = 1\n", encoding="utf-8")
    manager = LSPManager(workspace_root=tmp_path, config=parse_lsp_config({"enabled": True}))
    shutdowns = 0

    async def fake_start(self: LSPClient) -> None:
        return None

    async def fake_shutdown(self: LSPClient) -> None:
        nonlocal shutdowns
        shutdowns += 1

    monkeypatch.setattr(LSPClient, "start", fake_start)
    monkeypatch.setattr(LSPClient, "shutdown", fake_shutdown)

    await manager.for_file(target)
    await manager.shutdown_all()

    assert shutdowns == 1
    assert manager.status()["clients"] == []


@pytest.mark.anyio
async def test_shutdown_lsp_manager_helper_discards_registered_manager(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = LSPManager(workspace_root=tmp_path, config=parse_lsp_config({"enabled": True}))
    ctx = SimpleNamespace(lsp_manager=manager)
    shutdowns = 0

    async def fake_shutdown_all() -> None:
        nonlocal shutdowns
        shutdowns += 1

    monkeypatch.setattr(manager, "shutdown_all", fake_shutdown_all)
    lsp_module._registered_managers.add(manager)  # noqa: SLF001

    await shutdown_lsp_manager(ctx)

    assert shutdowns == 1
    assert manager not in lsp_module._registered_managers  # noqa: SLF001


def test_manager_skips_disabled_server(tmp_path: Path) -> None:
    config = parse_lsp_config({"enabled": True, "servers": {"python": {"disabled": True}}})
    manager = LSPManager(workspace_root=tmp_path, config=config)

    with pytest.raises(RuntimeError, match="No LSP server configured"):
        manager._server_for_path(tmp_path / "main.py")  # noqa: SLF001
