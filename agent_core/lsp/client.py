"""Async stdio client for one LSP server process."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import os
import shutil
from pathlib import Path
from typing import Any

from agent_core.lsp.protocol import (
    LSPProtocolError,
    classify_message,
    encode_message,
    make_error_response,
    make_notification,
    make_request,
    make_response,
    read_message,
)
from agent_core.lsp.types import LSPDiagnostic, LSPServerConfig, SyncedDocument

logger = logging.getLogger("agent_core.lsp.client")


class LSPError(RuntimeError):
    """Base error for LSP operations."""


class LSPDisabledError(LSPError):
    """Raised when LSP is disabled by settings."""


class LSPCommandNotFoundError(LSPError):
    """Raised when the configured language server command cannot be found."""


class LSPRequestTimeoutError(LSPError):
    """Raised when an LSP request times out."""


class LSPClient:
    """Manage one language server process and its synchronized documents."""

    def __init__(
        self,
        *,
        server_config: LSPServerConfig,
        root_dir: Path,
        request_timeout_seconds: float,
        diagnostics_settle_ms: int,
        settings: dict[str, Any] | None = None,
        initialization_options: dict[str, Any] | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        self.server_config = server_config
        self.root_dir = root_dir.resolve()
        self.request_timeout_seconds = request_timeout_seconds
        self.diagnostics_settle_ms = diagnostics_settle_ms
        self.settings = settings or {}
        self.initialization_options = initialization_options or {}
        self.env = env or {}
        self.process: asyncio.subprocess.Process | None = None
        self._request_id = 0
        self._pending: dict[int | str, asyncio.Future[Any]] = {}
        self._synced_documents: dict[Path, SyncedDocument] = {}
        self._diagnostics_by_uri: dict[str, list[LSPDiagnostic]] = {}
        self._published_version_by_uri: dict[str, int] = {}
        self._diagnostic_push_counter_by_uri: dict[str, int] = {}
        self._diagnostic_push_has_version_by_uri: dict[str, bool] = {}
        self._push_event = asyncio.Event()
        self._push_counter = 0
        self._read_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._write_lock = asyncio.Lock()
        self._initialized = False

    @property
    def is_running(self) -> bool:
        return self.process is not None and self.process.returncode is None

    @property
    def synced_document_count(self) -> int:
        return len(self._synced_documents)

    @property
    def diagnostics_count(self) -> int:
        return sum(len(items) for items in self._diagnostics_by_uri.values())

    async def start(self) -> None:
        if self.is_running and self._initialized:
            return
        if shutil.which(self.server_config.command) is None:
            raise LSPCommandNotFoundError(
                f"LSP server command not found: {self.server_config.command}"
            )
        self.process = await asyncio.create_subprocess_exec(
            self.server_config.command,
            *self.server_config.args,
            cwd=str(self.root_dir),
            env={**os.environ, **self.env},
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        if self.process.stdout is None or self.process.stdin is None:
            raise LSPError("LSP process did not expose stdio pipes")
        self._read_task = asyncio.create_task(self._read_loop())
        self._stderr_task = asyncio.create_task(self._drain_stderr())
        await self.initialize()

    async def initialize(self) -> None:
        if self._initialized:
            return
        root_uri = self.root_dir.as_uri()
        await self.request(
            "initialize",
            {
                "processId": os.getpid(),
                "rootUri": root_uri,
                "initializationOptions": self.initialization_options,
                "workspaceFolders": [
                    {
                        "uri": root_uri,
                        "name": self.root_dir.name,
                    }
                ],
                "capabilities": {
                    "textDocument": {
                        "synchronization": {
                            "didSave": True,
                            "dynamicRegistration": False,
                        },
                        "definition": {"dynamicRegistration": False},
                        "references": {"dynamicRegistration": False},
                        "hover": {"dynamicRegistration": False},
                        "documentSymbol": {"dynamicRegistration": False},
                        "publishDiagnostics": {"relatedInformation": True},
                    },
                    "workspace": {
                        "configuration": True,
                        "workspaceFolders": True,
                        "symbol": {"dynamicRegistration": False},
                        "didChangeWatchedFiles": {"dynamicRegistration": False},
                    },
                },
            },
        )
        await self.notify("initialized", {})
        self._initialized = True

    async def shutdown(self) -> None:
        if self.process is None:
            return
        if self.is_running and self._initialized:
            try:
                await self.request("shutdown", {}, timeout=2.0)
                await self.notify("exit", {})
            except Exception:
                pass
        if self.process.returncode is None:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()
        if self.process.stdin is not None:
            self.process.stdin.close()
            with contextlib.suppress(Exception):
                await self.process.stdin.wait_closed()
        if self._read_task is not None:
            self._read_task.cancel()
            try:
                await self._read_task
            except asyncio.CancelledError:
                pass
        if self._stderr_task is not None:
            self._stderr_task.cancel()
            try:
                await self._stderr_task
            except asyncio.CancelledError:
                pass
        self._reject_pending(LSPError("LSP client shut down"))
        self.process = None
        self._initialized = False

    def terminate_nowait(self) -> None:
        """Best-effort process cleanup for synchronous shutdown fallbacks."""
        if self.process is not None and self.process.stdin is not None:
            with contextlib.suppress(Exception):
                self.process.stdin.close()
        if self.process is not None and self.process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                self.process.terminate()
        if self._read_task is not None:
            self._read_task.cancel()
        if self._stderr_task is not None:
            self._stderr_task.cancel()
        self._reject_pending(LSPError("LSP client terminated"))
        self.process = None
        self._initialized = False

    async def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> Any:
        if not self.is_running:
            if method == "initialize":
                pass
            else:
                await self.start()
        self._request_id += 1
        request_id = self._request_id
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        self._pending[request_id] = future
        await self._write_message(make_request(request_id, method, params))
        try:
            return await asyncio.wait_for(
                future,
                timeout=timeout or self.request_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            self._pending.pop(request_id, None)
            logger.warning(
                "[%s] request timed out: %s",
                self.server_config.name,
                method,
            )
            raise LSPRequestTimeoutError(f"LSP request timed out: {method}") from exc

    async def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        if not self.is_running:
            await self.start()
        await self._write_message(make_notification(method, params))

    async def ensure_document_synced(self, path: Path) -> int:
        await self.start()
        resolved = path.resolve()
        content = resolved.read_text(encoding="utf-8")
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        existing = self._synced_documents.get(resolved)
        uri = resolved.as_uri()

        if existing is None:
            document = SyncedDocument(
                uri=uri,
                language_id=self.server_config.language_id,
                version=1,
                content_hash=content_hash,
                path=resolved,
            )
            await self._notify_watched_file(uri, change_type=1)
            await self.notify(
                "textDocument/didOpen",
                {
                    "textDocument": {
                        "uri": uri,
                        "languageId": self.server_config.language_id,
                        "version": document.version,
                        "text": content,
                    }
                },
            )
            self._synced_documents[resolved] = document
            return document.version

        if existing.content_hash == content_hash:
            return existing.version

        existing.version += 1
        existing.content_hash = content_hash
        await self._notify_watched_file(existing.uri, change_type=2)
        await self.notify(
            "textDocument/didChange",
            {
                "textDocument": {
                    "uri": existing.uri,
                    "version": existing.version,
                },
                "contentChanges": [{"text": content}],
            },
        )
        return existing.version

    async def save_document(self, path: Path) -> None:
        await self.start()
        await self.notify(
            "textDocument/didSave",
            {"textDocument": {"uri": path.resolve().as_uri()}},
        )

    async def diagnostics(self, path: Path | None = None) -> list[LSPDiagnostic]:
        if path is not None:
            version = await self.ensure_document_synced(path)
            await self.save_document(path)
            await self.wait_for_diagnostics(path, version)
            return list(self._diagnostics_by_uri.get(path.resolve().as_uri(), []))
        return [
            diagnostic
            for diagnostics in self._diagnostics_by_uri.values()
            for diagnostic in diagnostics
        ]

    async def definition(self, path: Path, line: int, character: int) -> Any:
        await self.ensure_document_synced(path)
        return await self.request(
            "textDocument/definition",
            self._text_document_position_params(path, line, character),
        )

    async def references(self, path: Path, line: int, character: int) -> Any:
        await self.ensure_document_synced(path)
        params = self._text_document_position_params(path, line, character)
        params["context"] = {"includeDeclaration": True}
        return await self.request("textDocument/references", params)

    async def hover(self, path: Path, line: int, character: int) -> Any:
        await self.ensure_document_synced(path)
        return await self.request(
            "textDocument/hover",
            self._text_document_position_params(path, line, character),
        )

    async def document_symbols(self, path: Path) -> Any:
        await self.ensure_document_synced(path)
        return await self.request(
            "textDocument/documentSymbol",
            {"textDocument": {"uri": path.resolve().as_uri()}},
        )

    async def workspace_symbols(self, query: str) -> Any:
        await self.start()
        return await self.request("workspace/symbol", {"query": query})

    async def wait_for_diagnostics(self, path: Path, version: int) -> None:
        uri = path.resolve().as_uri()
        initial_counter = self._diagnostic_push_counter_by_uri.get(uri, 0)
        deadline = asyncio.get_running_loop().time() + self.request_timeout_seconds

        while True:
            if self._has_suitable_diagnostic_push(uri, version, initial_counter):
                await self._settle_diagnostics()
                return

            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                logger.warning(
                    "[%s] diagnostics timed out for %s",
                    self.server_config.name,
                    path,
                )
                return

            try:
                await asyncio.wait_for(self._push_event.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                logger.warning(
                    "[%s] diagnostics timed out for %s",
                    self.server_config.name,
                    path,
                )
                return
            finally:
                self._push_event.clear()

    def status(self) -> dict[str, Any]:
        return {
            "name": self.server_config.name,
            "command": self.server_config.command,
            "root": str(self.root_dir),
            "running": self.is_running,
            "pid": self.process.pid if self.process is not None else None,
            "syncedDocuments": self.synced_document_count,
            "diagnostics": self.diagnostics_count,
        }

    def _text_document_position_params(
        self,
        path: Path,
        line: int,
        character: int,
    ) -> dict[str, Any]:
        return {
            "textDocument": {"uri": path.resolve().as_uri()},
            "position": {"line": line, "character": character},
        }

    async def _settle_diagnostics(self) -> None:
        if self.diagnostics_settle_ms <= 0:
            return
        await asyncio.sleep(self.diagnostics_settle_ms / 1000)

    async def _notify_watched_file(self, uri: str, *, change_type: int) -> None:
        await self.notify(
            "workspace/didChangeWatchedFiles",
            {"changes": [{"uri": uri, "type": change_type}]},
        )

    async def _write_message(self, message: dict[str, Any]) -> None:
        if self.process is None or self.process.stdin is None:
            raise LSPError("LSP process is not running")
        encoded = encode_message(message)
        async with self._write_lock:
            self.process.stdin.write(encoded)
            await self.process.stdin.drain()

    async def _read_loop(self) -> None:
        assert self.process is not None
        assert self.process.stdout is not None
        try:
            while True:
                message = await read_message(self.process.stdout)
                if message is None:
                    logger.debug(
                        "[%s] server closed stdout cleanly",
                        self.server_config.name,
                    )
                    self._reject_pending(LSPError("LSP stream closed"))
                    return
                await self._handle_incoming_message(message)
        except asyncio.CancelledError:
            raise
        except LSPProtocolError as exc:
            logger.warning(
                "[%s] protocol error in reader loop: %s",
                self.server_config.name,
                exc,
            )
            self._reject_pending(LSPError(str(exc)))
        except Exception as exc:
            logger.warning(
                "[%s] reader loop failed: %s",
                self.server_config.name,
                exc,
            )
            self._reject_pending(exc)

    async def _drain_stderr(self) -> None:
        if self.process is None or self.process.stderr is None:
            return
        try:
            while True:
                chunk = await self.process.stderr.read(4096)
                if not chunk:
                    return
        except asyncio.CancelledError:
            raise

    async def _handle_incoming_message(self, message: dict[str, Any]) -> None:
        kind, key = classify_message(message)
        if kind == "response":
            request_id = key
            future = self._pending.pop(request_id, None)
            if future is None or future.done():
                return
            if "error" in message:
                error = message.get("error") or {}
                future.set_exception(
                    LSPError(str(error.get("message") or f"LSP error response: {error}"))
                )
            else:
                future.set_result(message.get("result"))
            return

        if kind == "request":
            await self._handle_server_request(message)
            return

        if kind != "notification":
            logger.warning(
                "[%s] dropping invalid message: %s",
                self.server_config.name,
                _short_repr(message),
            )
            return

        method = key
        if method == "textDocument/publishDiagnostics":
            params = message.get("params") or {}
            uri = params.get("uri")
            raw_diagnostics = params.get("diagnostics", [])
            if isinstance(uri, str) and isinstance(raw_diagnostics, list):
                self._diagnostics_by_uri[uri] = [
                    self._parse_diagnostic(uri, item)
                    for item in raw_diagnostics
                    if isinstance(item, dict)
                ]
                version = params.get("version")
                if isinstance(version, int):
                    self._published_version_by_uri[uri] = version
                    self._diagnostic_push_has_version_by_uri[uri] = True
                else:
                    self._diagnostic_push_has_version_by_uri[uri] = False
                self._diagnostic_push_counter_by_uri[uri] = (
                    self._diagnostic_push_counter_by_uri.get(uri, 0) + 1
                )
                self._push_counter += 1
                self._push_event.set()
            return

    async def _handle_server_request(self, message: dict[str, Any]) -> None:
        request_id = message.get("id")
        method = message.get("method")
        try:
            if not isinstance(request_id, (int, str)):
                return
            if not isinstance(method, str):
                await self._write_message(make_error_response(request_id, -32600, "Invalid Request"))
                return
            result = self._server_request_result(method, message.get("params"))
            if result is _METHOD_NOT_FOUND:
                await self._write_message(
                    make_error_response(request_id, -32601, f"Method not found: {method}")
                )
                return
            await self._write_message(make_response(request_id, result))
        except Exception as exc:
            if isinstance(request_id, (int, str)):
                await self._write_message(
                    make_error_response(
                        request_id,
                        -32603,
                        str(exc) or exc.__class__.__name__,
                    )
                )

    def _server_request_result(self, method: str, params: Any) -> Any:
        if method == "workspace/configuration":
            return self._workspace_configuration(params)
        if method == "workspace/workspaceFolders":
            return [{"uri": self.root_dir.as_uri(), "name": self.root_dir.name}]
        if method == "window/workDoneProgress/create":
            return None
        return _METHOD_NOT_FOUND

    def _workspace_configuration(self, params: Any) -> list[Any]:
        items = params.get("items") if isinstance(params, dict) else []
        if not isinstance(items, list):
            items = []
        return [self._configuration_item(item) for item in items]

    def _configuration_item(self, item: Any) -> Any:
        if not isinstance(item, dict):
            return None
        section = item.get("section")
        if section is None or section == "":
            return self.settings if self.settings else None
        if not isinstance(section, str):
            return None
        current: Any = self.settings
        for part in section.split("."):
            if not isinstance(current, dict) or part not in current:
                return None
            current = current[part]
        return current

    def _has_suitable_diagnostic_push(
        self,
        uri: str,
        version: int,
        initial_counter: int,
    ) -> bool:
        if self._diagnostic_push_counter_by_uri.get(uri, 0) <= initial_counter:
            return False
        if not self._diagnostic_push_has_version_by_uri.get(uri, False):
            return True
        published_version = self._published_version_by_uri.get(uri)
        return published_version is not None and published_version >= version

    def _parse_diagnostic(self, uri: str, item: dict[str, Any]) -> LSPDiagnostic:
        return LSPDiagnostic(
            uri=uri,
            range=item.get("range") if isinstance(item.get("range"), dict) else {},
            severity=item.get("severity") if isinstance(item.get("severity"), int) else None,
            code=item.get("code") if isinstance(item.get("code"), (str, int)) else None,
            source=item.get("source") if isinstance(item.get("source"), str) else None,
            message=str(item.get("message") or ""),
        )

    def _reject_pending(self, error: Exception) -> None:
        for future in self._pending.values():
            if not future.done():
                future.set_exception(error)
        self._pending.clear()


def _short_repr(value: Any, *, limit: int = 1000) -> str:
    rendered = repr(value)
    if len(rendered) <= limit:
        return rendered
    return rendered[: limit - 3] + "..."


_METHOD_NOT_FOUND = object()
