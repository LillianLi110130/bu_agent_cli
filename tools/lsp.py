"""LSP-backed code intelligence tools."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field

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

LSPOperation = Literal[
    "status",
    "diagnostics",
    "definition",
    "references",
    "hover",
    "document_symbols",
    "workspace_symbols",
]


class LSPParams(BaseModel):
    """Arguments for the unified LSP tool."""

    operation: LSPOperation = Field(
        description=(
            "LSP operation to run. Use status for server state; diagnostics for errors; "
            "definition/references/hover are editor cursor-position operations for the symbol "
            "under file_path + line + character; "
            "document_symbols for a stable symbol outline of one file; workspace_symbols "
            "to search indexed workspace symbols by query."
        ),
    )
    file_path: str | None = Field(
        default=None,
        description=(
            "Path to a concrete source file, not '.' and not a directory. Required for "
            "definition, references, hover, document_symbols, and workspace_symbols. Optional "
            "for diagnostics. Not used for status. For workspace_symbols, pass an actual file "
            "such as 'agent_core/agent/service.py'; this selects the language server "
            "and workspace root."
        ),
    )
    line: int | None = Field(
        default=None,
        description=(
            "1-based line number. Required only for definition, references, and hover. "
            "It must be the exact editor line containing the target symbol."
        ),
    )
    character: int | None = Field(
        default=None,
        description=(
            "1-based character column. Required only for definition, references, and hover. "
            "It must point inside the target symbol text, not merely to the start of the line. "
            "Do not set this to 1 unless the symbol actually starts at column 1. If unsure, "
            "read the file line first and count the 1-based column of the symbol."
        ),
    )
    query: str | None = Field(
        default=None,
        description=(
            "Search text for workspace_symbols only. Ignored by status, diagnostics, "
            "definition, references, hover, and document_symbols. This is symbol-index search, not grep."
        ),
    )


@tool(
    (
        "Run Language Server Protocol code intelligence operations. "
        "Use status to inspect configured servers and active clients. "
        "Use diagnostics to return language-server diagnostics. "
        "Use definition as Go to Definition: given a concrete source file and an exact cursor "
        "position inside a symbol, return where an editor would jump. "
        "Use references to find usages of the symbol at an exact cursor position. "
        "Use hover to get type, signature, or documentation for the symbol at an exact cursor "
        "position. "
        "Use document_symbols to list the stable symbol outline declared in one file. "
        "Do not call document_symbols repeatedly for the same file unless the file changed "
        "or the previous call failed; repeated calls return the same snapshot. If truncated "
        "is true, it is not pagination and another call will return the same first results; "
        "use read or grep for code details instead. "
        "Use workspace_symbols to search indexed workspace symbols by query, with file_path "
        "selecting the language server. file_path must be a real source file, not '.' or a "
        "directory. Lines and characters are 1-based. "
        "For definition/references/hover, character must point inside the target symbol text; "
        "do not set character to 1 unless the symbol starts at column 1. If you only know a "
        "symbol name, use workspace_symbols first. If you know the line but not the character, "
        "read the file line first and count the 1-based column. query is only used by "
        "workspace_symbols."
    ),
    context_policy="trim",
    context_max_inline_chars=6400,
    args_schema=LSPParams,
)
async def lsp(
    operation: LSPOperation,
    ctx: Annotated[SandboxContext, Depends(get_sandbox_context)],
    file_path: str | None = None,
    line: int | None = None,
    character: int | None = None,
    query: str | None = None,
) -> str:
    if operation == "status":
        return await lsp_status.func(ctx=ctx)
    if operation == "diagnostics":
        return await lsp_diagnostics.func(ctx=ctx, file_path=file_path)
    if operation == "document_symbols":
        if file_path is None:
            return error_payload("lsp", "operation document_symbols requires file_path.")
        return await lsp_document_symbols.func(file_path=file_path, ctx=ctx)
    if operation == "workspace_symbols":
        if query is None or not str(query).strip():
            return error_payload("lsp", "operation workspace_symbols requires query.")
        if file_path is None:
            return error_payload("lsp", "operation workspace_symbols requires file_path.")
        return await lsp_workspace_symbols.func(query=query, ctx=ctx, file_path=file_path)
    if operation in {"definition", "references", "hover"}:
        missing = []
        if file_path is None:
            missing.append("file_path")
        if line is None:
            missing.append("line")
        if character is None:
            missing.append("character")
        if missing:
            return error_payload("lsp", f"operation {operation} requires {', '.join(missing)}.")
        if operation == "definition":
            return await lsp_definition.func(file_path, line, character, ctx=ctx)
        if operation == "references":
            return await lsp_references.func(file_path, line, character, ctx=ctx)
        return await lsp_hover.func(file_path, line, character, ctx=ctx)
    return error_payload("lsp", f"Unsupported LSP operation: {operation}")


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
    "Go to Definition for the symbol under an exact 1-based cursor position. The character "
    "must point inside the target symbol text; this is not name-based search.",
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
    "Find references for the symbol under an exact 1-based cursor position. The character "
    "must point inside the target symbol text; this is not name-based search.",
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
    "Return hover/type information for the symbol under an exact 1-based cursor position. "
    "The character must point inside the target symbol text; this is not name-based search.",
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


@tool(
    "Return a stable symbol outline declared in one source file using LSP. Do not call this "
    "repeatedly for the same file unless the file changed or the previous call failed. If "
    "truncated is true, repeated calls return the same first results; use read or grep for "
    "code details instead.",
    context_policy="trim",
)
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


@tool(
    "Search workspace symbols using file_path to choose one language server.",
    context_policy="trim",
)
async def lsp_workspace_symbols(
    query: str,
    file_path: str,
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
        path = _resolve_file(file_path, ctx)
        raw = await manager.workspace_symbols(query, path)
        return format_symbols("lsp_workspace_symbols", raw, workspace_root=ctx.root_dir)
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
    stripped = file_path.strip()
    if stripped in {".", "./"} or stripped.endswith(("/", "\\")):
        raise LSPError(
            "file_path must be a concrete source file, not '.' or a directory. "
            "For workspace_symbols, pass a file for the target language, e.g. "
            "'agent_core/agent/service.py'."
        )
    try:
        direct = ctx.resolve_path(stripped)
    except Exception:
        direct = None
    if direct is not None and direct.exists() and direct.is_dir():
        raise LSPError(
            "file_path must be a concrete source file, not a directory. "
            "For workspace_symbols, pass a file for the target language, e.g. "
            "'agent_core/agent/service.py'."
        )
    try:
        return resolve_target_path(stripped, ctx, kind="file")
    except (PathNotFoundError, AmbiguousPathError):
        raise
    except Exception as exc:
        raise LSPError(f"Security error: {exc}") from exc


def _to_lsp_position(line: int, character: int) -> tuple[int, int]:
    if line <= 0 or character <= 0:
        raise LSPError("line and character must be positive 1-based integers")
    return line - 1, character - 1


def _error_message(exc: Exception) -> str:
    if isinstance(exc, (PathNotFoundError, AmbiguousPathError, LSPError)):
        return str(exc)
    return str(exc) or exc.__class__.__name__
