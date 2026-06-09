"""Workspace-level MCP client manager."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path
from typing import Any, Callable

from agent_core.mcp.client import MCPClient, MCPError
from agent_core.mcp.config import load_mcp_config
from agent_core.mcp.types import MCPConfig, MCPServerConfig, MCPTool

logger = logging.getLogger("agent_core.mcp.manager")


class MCPManager:
    """Create and manage configured MCP clients."""

    def __init__(
        self,
        *,
        workspace_root: Path,
        config: MCPConfig,
        request_timeout_seconds: float = 10.0,
    ) -> None:
        self.workspace_root = workspace_root.resolve()
        self.config = config
        self.request_timeout_seconds = request_timeout_seconds
        self._clients: dict[str, MCPClient] = {}
        self._spawning: dict[str, asyncio.Task[MCPClient]] = {}
        self._states: dict[str, str] = {
            name: "disabled" if server.disabled else "configured"
            for name, server in config.servers.items()
        }
        self._errors: dict[str, str] = {}
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self.on_tools_changed: Callable[[], None] | None = None

    @classmethod
    def from_settings(cls, workspace_root: Path) -> "MCPManager":
        return cls(workspace_root=workspace_root, config=load_mcp_config(workspace_root))

    def reload_config(self) -> None:
        """Reload MCP config from disk while preserving running clients."""
        self.config = load_mcp_config(self.workspace_root)
        configured_names = set(self.config.servers)
        for name in list(self._states):
            if name not in configured_names:
                self._states.pop(name, None)
                self._errors.pop(name, None)
        for name, server in self.config.servers.items():
            if server.disabled:
                self._states[name] = "disabled"
            elif name in self._clients:
                self._states[name] = "running"
            else:
                state = self._states.get(name)
                self._states[name] = state if state in {"starting", "failed", "stopped"} else "configured"

    def start_enabled_servers_background(self) -> None:
        for server in self.config.servers.values():
            if server.disabled:
                self._states[server.name] = "disabled"
                continue
            task = asyncio.create_task(self._start_server_safely(server.name))
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

    async def start_enabled_servers(self) -> None:
        await asyncio.gather(
            *[
                self._start_server_safely(server.name)
                for server in self.config.servers.values()
                if not server.disabled
            ],
            return_exceptions=True,
        )

    async def start_server(self, name: str) -> MCPClient:
        server = self._server_by_name(name)
        if server.disabled:
            self._states[server.name] = "disabled"
            raise MCPError(f"MCP server is disabled: {server.name}")
        client = self._clients.get(server.name)
        if client is not None:
            return client
        spawning = self._spawning.get(server.name)
        if spawning is not None:
            return await spawning
        self._states[server.name] = "starting"
        self._errors.pop(server.name, None)
        task = asyncio.create_task(self._spawn_client(server))
        self._spawning[server.name] = task
        try:
            return await task
        finally:
            self._spawning.pop(server.name, None)

    async def stop_server(self, name: str) -> None:
        server = self._server_by_name(name)
        task = self._spawning.pop(server.name, None)
        if task is not None:
            task.cancel()
            with contextlib.suppress(BaseException):
                await task
        client = self._clients.pop(server.name, None)
        if client is not None:
            await client.shutdown()
        self._states[server.name] = "stopped"
        self._emit_tools_changed()

    async def restart_server(self, name: str) -> MCPClient:
        await self.stop_server(name)
        return await self.start_server(name)

    def list_tools(self, server_name: str | None = None) -> list[MCPTool]:
        clients = (
            [self._clients[server_name]]
            if server_name is not None and server_name in self._clients
            else list(self._clients.values())
        )
        return [tool for client in clients for tool in client.tools]

    def instructions(self, server_name: str | None = None) -> dict[str, str]:
        clients = (
            [self._clients[server_name]]
            if server_name is not None and server_name in self._clients
            else list(self._clients.values())
        )
        result: dict[str, str] = {}
        for client in clients:
            if client.instructions:
                result[client.server_config.name] = client.instructions
        return result

    async def call_tool(self, server: str, tool: str, arguments: dict[str, Any] | None) -> Any:
        client = self._clients.get(server)
        if client is None:
            state = self._states.get(server)
            if state == "failed":
                raise MCPError(f"MCP server {server} failed: {self._errors.get(server, '')}")
            client = await self.start_server(server)
        return await client.call_tool(tool, arguments or {})

    async def shutdown_all(self) -> None:
        tasks = list(self._background_tasks) + list(self._spawning.values())
        self._background_tasks.clear()
        self._spawning.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        clients = list(self._clients.values())
        self._clients.clear()
        for client in clients:
            await client.shutdown()
        for name, server in self.config.servers.items():
            self._states[name] = "disabled" if server.disabled else "stopped"

    def terminate_all_nowait(self) -> None:
        clients = list(self._clients.values())
        self._clients.clear()
        self._spawning.clear()
        self._background_tasks.clear()
        for client in clients:
            client.terminate_nowait()

    def status(self) -> dict[str, Any]:
        servers = []
        for name, server in self.config.servers.items():
            client = self._clients.get(name)
            status = client.status() if client is not None else {}
            servers.append(
                {
                    "name": name,
                    "state": self._states.get(name, "configured"),
                    "type": server.type,
                    "command": server.command,
                    "args": server.args,
                    "disabled": server.disabled,
                    "source": server.source,
                    "toolCount": status.get("toolCount", 0),
                    "tools": status.get("tools", []),
                    "instructions": status.get("instructions", False),
                    "error": self._errors.get(name),
                }
            )
        return {
            "workspaceRoot": str(self.workspace_root),
            "projectConfigPath": self.config.project_config_path,
            "userConfigPath": self.config.user_config_path,
            "servers": servers,
            "clients": [client.status() for client in self._clients.values()],
        }

    async def _start_server_safely(self, name: str) -> None:
        with contextlib.suppress(Exception):
            await self.start_server(name)

    async def _spawn_client(self, server: MCPServerConfig) -> MCPClient:
        client = MCPClient(
            server_config=server,
            request_timeout_seconds=self.request_timeout_seconds,
        )
        try:
            await client.start()
        except BaseException as exc:
            with contextlib.suppress(BaseException):
                await client.shutdown()
            if isinstance(exc, asyncio.CancelledError):
                raise
            logger.warning("[%s] MCP start failed: %s", server.name, exc)
            self._states[server.name] = "failed"
            self._errors[server.name] = str(exc) or exc.__class__.__name__
            self._emit_tools_changed()
            raise
        self._clients[server.name] = client
        self._states[server.name] = "running"
        self._emit_tools_changed()
        return client

    def _emit_tools_changed(self) -> None:
        if self.on_tools_changed is None:
            return
        with contextlib.suppress(Exception):
            self.on_tools_changed()

    def _server_by_name(self, name: str) -> MCPServerConfig:
        server = self.config.servers.get(name)
        if server is None:
            raise MCPError(f"No MCP server configured with name: {name}")
        return server
