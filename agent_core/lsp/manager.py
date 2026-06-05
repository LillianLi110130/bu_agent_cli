"""Workspace-level LSP client manager."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path

from agent_core.lsp.client import LSPDisabledError, LSPError, LSPClient
from agent_core.lsp.config import load_lsp_config
from agent_core.lsp.types import LSPConfig, LSPDiagnostic, LSPServerConfig

logger = logging.getLogger("agent_core.lsp.manager")


class LSPManager:
    """Create and cache LSP clients by server and resolved root."""

    def __init__(self, *, workspace_root: Path, config: LSPConfig) -> None:
        self.workspace_root = workspace_root.resolve()
        self.config = config
        self._clients: dict[tuple[str, Path], LSPClient] = {}
        self._spawning: dict[tuple[str, Path], asyncio.Task[LSPClient]] = {}
        self._broken: set[tuple[str, Path]] = set()

    @classmethod
    def from_settings(cls, workspace_root: Path) -> "LSPManager":
        return cls(workspace_root=workspace_root, config=load_lsp_config())

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    async def for_file(self, path: Path) -> LSPClient:
        self._ensure_enabled()
        resolved = path.resolve()
        server = self._server_for_path(resolved)
        root = self._resolve_root(resolved, server)
        key = (server.name, root)
        if key in self._broken:
            raise LSPError(f"LSP server is marked broken for {server.name}: {root}")
        client = self._clients.get(key)
        if client is None:
            if not self.config.auto_start:
                raise LSPError(
                    f"LSP autoStart is disabled and no client is running for {server.name}"
                )
            spawning = self._spawning.get(key)
            if spawning is not None:
                return await spawning
            spawning = asyncio.create_task(self._spawn_client(key, server, root))
            self._spawning[key] = spawning
            try:
                return await spawning
            finally:
                self._spawning.pop(key, None)
        return client

    async def diagnostics(self, path: Path | None = None) -> list[LSPDiagnostic]:
        self._ensure_enabled()
        if path is not None:
            client = await self.for_file(path)
            return await client.diagnostics(path)
        return [
            diagnostic
            for client in self._clients.values()
            for diagnostic in await client.diagnostics(None)
        ]

    async def shutdown_all(self) -> None:
        spawning = list(self._spawning.values())
        self._spawning.clear()
        for task in spawning:
            task.cancel()
        if spawning:
            await asyncio.gather(*spawning, return_exceptions=True)

        clients = list(self._clients.values())
        self._clients.clear()
        for client in clients:
            await client.shutdown()

    def terminate_all_nowait(self) -> None:
        """Best-effort process cleanup when async shutdown cannot run."""
        clients = list(self._clients.values())
        self._clients.clear()
        self._spawning.clear()
        for client in clients:
            client.terminate_nowait()

    def status(self) -> dict[str, object]:
        return {
            "enabled": self.config.enabled,
            "workspaceRoot": str(self.workspace_root),
            "servers": [
                {
                    "name": server.name,
                    "command": server.command,
                    "extensions": server.extensions,
                    "languageId": server.language_id,
                }
                for server in self.config.servers.values()
            ],
            "clients": [client.status() for client in self._clients.values()],
            "broken": [
                {"name": name, "root": str(root)}
                for name, root in sorted(self._broken, key=lambda item: (item[0], str(item[1])))
            ],
        }

    def _ensure_enabled(self) -> None:
        if not self.config.enabled:
            raise LSPDisabledError(
                "LSP is disabled by settings. Set settings.lsp.enabled to true in ~/.tg_agent/settings.json."
            )

    def _server_for_path(self, path: Path) -> LSPServerConfig:
        suffix = path.suffix.lower()
        for server in self.config.servers.values():
            if server.disabled:
                continue
            if suffix in server.extensions:
                return server
        raise LSPError(f"No LSP server configured for extension {suffix or '(none)'}")

    async def _spawn_client(
        self,
        key: tuple[str, Path],
        server: LSPServerConfig,
        root: Path,
    ) -> LSPClient:
        client = LSPClient(
            server_config=server,
            root_dir=root,
            request_timeout_seconds=self.config.request_timeout_seconds,
            diagnostics_settle_ms=self.config.diagnostics_settle_ms,
            settings=server.settings,
            initialization_options=server.initialization_options,
            env=server.env,
        )
        try:
            await client.start()
        except Exception as exc:
            logger.warning(
                "[%s] spawn/initialize failed for %s: %s",
                server.name,
                root,
                exc,
            )
            self._broken.add(key)
            with contextlib.suppress(Exception):
                await client.shutdown()
            raise
        self._clients[key] = client
        return client

    def _resolve_root(self, path: Path, server: LSPServerConfig) -> Path:
        current = path.parent.resolve()
        while True:
            for marker in server.root_markers:
                if (current / marker).exists():
                    return current
            if current == self.workspace_root or current.parent == current:
                return self.workspace_root
            if not _is_same_or_parent(self.workspace_root, current.parent):
                return self.workspace_root
            current = current.parent


def _is_same_or_parent(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False
