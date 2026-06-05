"""LSP-backed code intelligence tools."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from agent_core.lsp.client import LSPError
from agent_core.lsp.formatter import (
    error_payload,
    format_diagnostics,
    format_hover,
    format_locations,
    format_status,
    format_symbols,
)
from agent_core.tools import Depends, tool
from tools.path_resolution import AmbiguousPathError, PathNotFoundError, resolve_target_path
from tools.sandbox import SandboxContext, get_sandbox_context


@tool("Show LSP server configuration and currently running clients.", context_policy="trim")
async def lsp_status(
    ctx: Annotated[SandboxContext, Depends(get_sandbox_context)],
) -> str:
    manager = _get_manager(ctx)
    if manager is None:
        return error_payload("lsp_status", "LSP manager is not initialized.")
    return format_status(manager.status())


@tool(
    "Return LSP diagnostics for a file, or cached diagnostics for started LSP clients when "
    "file_path is omitted.",
    context_policy="trim",
    context_max_inline_chars=6400,
)
async def lsp_diagnostics(
    ctx: Annotated[SandboxContext, Depends(get_sandbox_context)],
    file_path: str | None = None,
) -> str:
    manager = _get_manager(ctx)
    if manager is None:
        return error_payload("lsp_diagnostics", "LSP manager is not initialized.")
    try:
        path = _resolve_optional_file(file_path, ctx)
        diagnostics = await manager.diagnostics(path)
        return format_diagnostics(diagnostics)
    except Exception as exc:
        return error_payload("lsp_diagnostics", _error_message(exc))


@tool(
    "Find the semantic definition for the symbol at a 1-based file line and character.",
    context_policy="trim",
    context_max_inline_chars=6400,
)
async def lsp_definition(
    file_path: str,
    line: int,
    character: int,
    ctx: Annotated[SandboxContext, Depends(get_sandbox_context)],
) -> str:
    return await _position_request(
        tool_name="lsp_definition",
        file_path=file_path,
        line=line,
        character=character,
        ctx=ctx,
        method_name="definition",
    )


@tool(
    "Find semantic references for the symbol at a 1-based file line and character.",
    context_policy="trim",
    context_max_inline_chars=6400,
)
async def lsp_references(
    file_path: str,
    line: int,
    character: int,
    ctx: Annotated[SandboxContext, Depends(get_sandbox_context)],
) -> str:
    return await _position_request(
        tool_name="lsp_references",
        file_path=file_path,
        line=line,
        character=character,
        ctx=ctx,
        method_name="references",
    )


@tool(
    "Return hover/type information for the symbol at a 1-based file line and character.",
    context_policy="trim",
    context_max_inline_chars=6400,
)
async def lsp_hover(
    file_path: str,
    line: int,
    character: int,
    ctx: Annotated[SandboxContext, Depends(get_sandbox_context)],
) -> str:
    manager = _get_manager(ctx)
    if manager is None:
        return error_payload("lsp_hover", "LSP manager is not initialized.")
    try:
        path = _resolve_file(file_path, ctx)
        lsp_line, lsp_character = _to_lsp_position(line, character)
        client = await manager.for_file(path)
        raw = await client.hover(path, lsp_line, lsp_character)
        return format_hover(raw)
    except Exception as exc:
        return error_payload("lsp_hover", _error_message(exc))


@tool("Return symbols declared in one source file using LSP.", context_policy="trim")
async def lsp_document_symbols(
    file_path: str,
    ctx: Annotated[SandboxContext, Depends(get_sandbox_context)],
) -> str:
    manager = _get_manager(ctx)
    if manager is None:
        return error_payload("lsp_document_symbols", "LSP manager is not initialized.")
    try:
        path = _resolve_file(file_path, ctx)
        client = await manager.for_file(path)
        raw = await client.document_symbols(path)
        return format_symbols("lsp_document_symbols", raw, workspace_root=ctx.root_dir)
    except Exception as exc:
        return error_payload("lsp_document_symbols", _error_message(exc))


@tool("Search workspace symbols using started/configured LSP servers.", context_policy="trim")
async def lsp_workspace_symbols(
    query: str,
    ctx: Annotated[SandboxContext, Depends(get_sandbox_context)],
) -> str:
    manager = _get_manager(ctx)
    if manager is None:
        return error_payload("lsp_workspace_symbols", "LSP manager is not initialized.")
    try:
        if not manager.enabled:
            raise LSPError(
                "LSP is disabled by settings. Set settings.lsp.enabled to true in ~/.tg_agent/settings.json."
            )
        results = []
        for server in manager.config.servers.values():
            probe = _find_probe_file(ctx.root_dir, server.extensions)
            if probe is None:
                continue
            client = await manager.for_file(probe)
            raw = await client.workspace_symbols(query)
            results.append(raw)
        flattened = [item for raw in results if isinstance(raw, list) for item in raw]
        return format_symbols("lsp_workspace_symbols", flattened, workspace_root=ctx.root_dir)
    except Exception as exc:
        return error_payload("lsp_workspace_symbols", _error_message(exc))


async def _position_request(
    *,
    tool_name: str,
    file_path: str,
    line: int,
    character: int,
    ctx: SandboxContext,
    method_name: str,
) -> str:
    manager = _get_manager(ctx)
    if manager is None:
        return error_payload(tool_name, "LSP manager is not initialized.")
    try:
        path = _resolve_file(file_path, ctx)
        lsp_line, lsp_character = _to_lsp_position(line, character)
        client = await manager.for_file(path)
        method = getattr(client, method_name)
        raw = await method(path, lsp_line, lsp_character)
        return format_locations(tool_name, raw, workspace_root=ctx.root_dir)
    except Exception as exc:
        return error_payload(tool_name, _error_message(exc))


def _get_manager(ctx: SandboxContext):
    return getattr(ctx, "lsp_manager", None)


def _resolve_optional_file(file_path: str | None, ctx: SandboxContext) -> Path | None:
    if file_path is None or not str(file_path).strip():
        return None
    return _resolve_file(str(file_path), ctx)


def _resolve_file(file_path: str, ctx: SandboxContext) -> Path:
    try:
        return resolve_target_path(file_path, ctx, kind="file")
    except (PathNotFoundError, AmbiguousPathError):
        raise
    except Exception as exc:
        raise LSPError(f"Security error: {exc}") from exc


def _to_lsp_position(line: int, character: int) -> tuple[int, int]:
    if line <= 0 or character <= 0:
        raise LSPError("line and character must be positive 1-based integers")
    return line - 1, character - 1


def _find_probe_file(root: Path, extensions: list[str]) -> Path | None:
    try:
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in extensions:
                return path
    except OSError:
        return None
    return None


def _error_message(exc: Exception) -> str:
    if isinstance(exc, (PathNotFoundError, AmbiguousPathError, LSPError)):
        return str(exc)
    return str(exc) or exc.__class__.__name__
