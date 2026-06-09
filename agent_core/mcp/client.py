"""Async stdio client for one MCP server process."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import shutil
from typing import Any

from agent_core.mcp.protocol import (
    MCPProtocolError,
    make_notification,
    make_request,
    read_message,
    write_message,
)
from agent_core.mcp.types import MCPServerConfig, MCPTool
from agent_core.version import get_cli_version

logger = logging.getLogger("agent_core.mcp.client")

MCP_PROTOCOL_VERSION = "2024-11-05"


class MCPError(RuntimeError):
    """Base MCP error."""


class MCPDisabledError(MCPError):
    """Raised when MCP use is disabled."""


class MCPCommandNotFoundError(MCPError):
    """Raised when the configured MCP server command cannot be found."""


class MCPClient:
    """Manage one MCP server subprocess over stdio."""

    def __init__(
        self,
        *,
        server_config: MCPServerConfig,
        request_timeout_seconds: float = 10.0,
    ) -> None:
        self.server_config = server_config
        self.request_timeout_seconds = request_timeout_seconds
        self.process: asyncio.subprocess.Process | None = None
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._read_task: asyncio.Task[None] | None = None
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._request_id = 0
        self._tools: dict[str, MCPTool] = {}
        self.instructions: str | None = None
        self.server_info: dict[str, Any] | None = None
        self.capabilities: dict[str, Any] = {}
        self._initialized = False

    @property
    def tools(self) -> list[MCPTool]:
        return list(self._tools.values())

    async def start(self) -> None:
        if shutil.which(self.server_config.command) is None:
            raise MCPCommandNotFoundError(
                f"MCP server command not found: {self.server_config.command}"
            )
        env = os.environ.copy()
        if self.server_config.env:
            env.update(self.server_config.env)
        self.process = await asyncio.create_subprocess_exec(
            self.server_config.command,
            *self.server_config.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env=env,
        )
        if self.process.stdin is None or self.process.stdout is None:
            raise MCPError("MCP process did not expose stdio pipes")
        self._reader = self.process.stdout
        self._writer = self.process.stdin
        self._read_task = asyncio.create_task(self._read_loop())
        await self.initialize()
        await self.list_tools()

    async def initialize(self) -> dict[str, Any]:
        result = await self.request(
            "initialize",
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "clientInfo": {
                    "name": "bu-agent-cli",
                    "version": get_cli_version(),
                },
            },
        )
        if not isinstance(result, dict):
            raise MCPError("MCP initialize result must be an object")
        self.server_info = result.get("serverInfo") if isinstance(result.get("serverInfo"), dict) else None
        capabilities = result.get("capabilities")
        self.capabilities = capabilities if isinstance(capabilities, dict) else {}
        instructions = result.get("instructions")
        self.instructions = instructions if isinstance(instructions, str) else None
        await self.notify("notifications/initialized")
        self._initialized = True
        return result

    async def list_tools(self) -> list[MCPTool]:
        result = await self.request("tools/list")
        if not isinstance(result, dict):
            raise MCPError("MCP tools/list result must be an object")
        raw_tools = result.get("tools", [])
        if not isinstance(raw_tools, list):
            raise MCPError("MCP tools/list result.tools must be a list")
        tools: dict[str, MCPTool] = {}
        for raw in raw_tools:
            if not isinstance(raw, dict):
                continue
            name = raw.get("name")
            if not isinstance(name, str) or not name:
                continue
            description = raw.get("description")
            input_schema = raw.get("inputSchema")
            tools[name] = MCPTool(
                server_name=self.server_config.name,
                name=name,
                description=description if isinstance(description, str) else "",
                input_schema=input_schema if isinstance(input_schema, dict) else {},
            )
        self._tools = tools
        return list(tools.values())

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        if name not in self._tools:
            await self.list_tools()
        return await self.request(
            "tools/call",
            {"name": name, "arguments": arguments or {}},
        )

    async def request(
        self,
        method: str,
        params: Any | None = None,
        *,
        timeout: float | None = None,
    ) -> Any:
        if self._writer is None:
            raise MCPError("MCP client is not started")
        self._request_id += 1
        request_id = self._request_id
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        self._pending[request_id] = future
        await write_message(self._writer, make_request(request_id, method, params))
        try:
            return await asyncio.wait_for(
                future,
                timeout=timeout or self.request_timeout_seconds,
            )
        except TimeoutError as exc:
            self._pending.pop(request_id, None)
            raise MCPError(f"MCP request timed out: {method}") from exc

    async def notify(self, method: str, params: Any | None = None) -> None:
        if self._writer is None:
            raise MCPError("MCP client is not started")
        await write_message(self._writer, make_notification(method, params))

    async def shutdown(self) -> None:
        if self._writer is not None and self.process is not None and self._initialized:
            with contextlib.suppress(Exception):
                await self.request("shutdown", timeout=2.0)
            with contextlib.suppress(Exception):
                await self.notify("exit")
        if self._writer is not None:
            await self._close_writer()
        if self._read_task is not None:
            self._read_task.cancel()
            with contextlib.suppress(BaseException):
                await self._read_task
            self._read_task = None
        if self.process is not None and self.process.returncode is None:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                self.process.kill()
                with contextlib.suppress(Exception):
                    await self.process.wait()
            except Exception:
                pass
        self._fail_pending(MCPError("MCP client shut down"))
        self.process = None
        self._reader = None
        self._writer = None
        self._initialized = False

    def terminate_nowait(self) -> None:
        process = self.process
        writer = self._writer
        if writer is not None:
            with contextlib.suppress(Exception):
                writer.close()
        if self._read_task is not None:
            self._read_task.cancel()
        if process is not None and process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.terminate()
        self._fail_pending(MCPError("MCP client terminated"))
        self.process = None
        self._reader = None
        self._writer = None
        self._initialized = False

    async def _close_writer(self) -> None:
        writer = self._writer
        self._writer = None
        if writer is None:
            return
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()

    def status(self) -> dict[str, Any]:
        running = self.process is not None and self.process.returncode is None
        return {
            "name": self.server_config.name,
            "state": "running" if running else "stopped",
            "type": self.server_config.type,
            "command": self.server_config.command,
            "args": self.server_config.args,
            "toolCount": len(self._tools),
            "tools": [tool.name for tool in self.tools],
            "instructions": bool(self.instructions),
            "serverInfo": self.server_info,
        }

    async def _read_loop(self) -> None:
        assert self._reader is not None
        try:
            while True:
                message = await read_message(self._reader)
                if message is None:
                    logger.debug("[%s] server clean EOF", self.server_config.name)
                    self._fail_pending(MCPError("MCP server exited"))
                    return
                self._handle_message(message)
        except asyncio.CancelledError:
            raise
        except MCPProtocolError as exc:
            logger.warning("[%s] MCP protocol error: %s", self.server_config.name, exc)
            self._fail_pending(exc)
        except Exception as exc:
            logger.warning("[%s] MCP read loop failed: %s", self.server_config.name, exc)
            self._fail_pending(exc)

    def _handle_message(self, message: dict[str, Any]) -> None:
        if "id" in message and ("result" in message or "error" in message):
            request_id = message.get("id")
            if not isinstance(request_id, int):
                return
            future = self._pending.pop(request_id, None)
            if future is None or future.done():
                return
            if "error" in message:
                error = message.get("error")
                if isinstance(error, dict):
                    msg = str(error.get("message") or error)
                else:
                    msg = str(error)
                future.set_exception(MCPError(msg))
            else:
                future.set_result(message.get("result"))
            return

        method = message.get("method")
        if method == "notifications/tools/list_changed":
            asyncio.create_task(self._refresh_tools_safely())

    async def _refresh_tools_safely(self) -> None:
        with contextlib.suppress(Exception):
            await self.list_tools()

    def _fail_pending(self, exc: Exception) -> None:
        pending = list(self._pending.values())
        self._pending.clear()
        for future in pending:
            if not future.done():
                future.set_exception(exc)
