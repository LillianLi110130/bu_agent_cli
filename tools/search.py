"""Search tools: glob pattern matching and grep content search."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Annotated, Iterable, Literal

from agent_core.tools import Depends, tool
from tools.path_resolution import AmbiguousPathError, PathNotFoundError, resolve_target_path
from tools.sandbox import SandboxContext, get_sandbox_context
from tools.text_encoding import read_text_with_fallback

GlobKind = Literal["file", "dir", "any"]
_GLOB_RESULT_LIMIT = 50


def _iter_searchable_files(root: Path, ctx: SandboxContext) -> Iterable[Path]:
    """Yield files below ``root`` while pruning ignored paths."""
    resolved_root = root.resolve()
    if ctx.is_ignored(resolved_root):
        return
    if resolved_root.is_file():
        if not ctx.is_ignored(resolved_root):
            yield resolved_root
        return

    for current, dirs, files in os.walk(resolved_root):
        current_path = Path(current)
        dirs[:] = [name for name in dirs if not ctx.is_ignored(current_path / name)]
        for file_name in files:
            candidate = current_path / file_name
            if not ctx.is_ignored(candidate):
                yield candidate


def _matches_kind(path: Path, kind: GlobKind) -> bool:
    if kind == "file":
        return path.is_file()
    if kind == "dir":
        return path.is_dir()
    return path.exists()


def _is_searchable_path(path: Path, ctx: SandboxContext, kind: GlobKind) -> bool:
    try:
        resolved = path.resolve()
    except OSError:
        return False
    return (
        ctx.is_allowed(resolved)
        and not ctx.is_ignored(resolved)
        and _matches_kind(resolved, kind)
    )


def _sort_key_for_glob_match(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix().lower()
    except ValueError:
        return str(path).lower()


def _iter_glob_matches(root: Path, pattern: str, ctx: SandboxContext, kind: GlobKind) -> list[Path]:
    pattern_text = pattern.strip().replace("\\", "/")
    if not pattern_text:
        raise ValueError("pattern cannot be empty")
    pattern_path = Path(pattern_text)
    if pattern_text.startswith("/") or pattern_path.is_absolute() or pattern_path.drive:
        raise ValueError("pattern must be relative to the search path")

    matches = {
        candidate.resolve()
        for candidate in root.glob(pattern_text)
        if _is_searchable_path(candidate, ctx, kind)
    }
    return sorted(matches, key=lambda item: _sort_key_for_glob_match(item, root))


def _display_path(path: Path, ctx: SandboxContext) -> str:
    try:
        return str(path.relative_to(ctx.root_dir))
    except ValueError:
        return str(path)


@tool(
    "Find files or directories matching a glob pattern",
    context_policy="trim",
    context_max_inline_chars=1600,
)
async def glob_search(
    pattern: str,
    ctx: Annotated[SandboxContext, Depends(get_sandbox_context)],
    path: str | None = None,
    kind: GlobKind = "file",
) -> str:
    """Find files or directories matching a glob pattern like ``**/*.py``."""
    try:
        if path:
            search_dir = resolve_target_path(path, ctx, kind="dir")
        else:
            search_dir = ctx.working_dir
    except Exception as e:
        if isinstance(e, (PathNotFoundError, AmbiguousPathError)):
            return str(e)
        return f"Security error: {e}"

    try:
        all_matches = [
            _display_path(candidate, ctx)
            for candidate in _iter_glob_matches(search_dir, pattern, ctx, kind)
        ]
        matches = all_matches[:_GLOB_RESULT_LIMIT]
        result_kind = "path" if kind == "any" else kind
        if not matches:
            return f"No {result_kind}s match pattern: {pattern}"
        if len(all_matches) > _GLOB_RESULT_LIMIT:
            return (
                f"Found {len(all_matches)} {result_kind}(s), showing first {_GLOB_RESULT_LIMIT}:\n"
                + "\n".join(matches)
                + "\n... (truncated; refine pattern or path)"
            )
        return f"Found {len(matches)} {result_kind}(s):\n" + "\n".join(matches)
    except Exception as e:
        return f"Error: {e}"


@tool("Search file contents with regex", context_policy="trim", context_max_inline_chars=1800)
async def grep(
    pattern: str,
    ctx: Annotated[SandboxContext, Depends(get_sandbox_context)],
    path: str | None = None,
) -> str:
    """Search for a regular-expression pattern in files recursively."""
    try:
        if path:
            search_target = resolve_target_path(path, ctx, kind="any")
        else:
            search_target = ctx.working_dir
    except Exception as e:
        if isinstance(e, (PathNotFoundError, AmbiguousPathError)):
            return str(e)
        return f"Security error: {e}"

    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"Invalid regex: {e}"

    results: list[str] = []
    for file_path in _iter_searchable_files(search_target, ctx):
        try:
            content, _ = read_text_with_fallback(file_path)
            for i, line in enumerate(content.splitlines(), 1):
                if regex.search(line):
                    results.append(f"{_display_path(file_path, ctx)}:{i}: {line[:100]}")
                    if len(results) >= 50:
                        return "\n".join(results) + "\n... (truncated)"
        except Exception:
            continue
    return "\n".join(results) if results else f"No matches for: {pattern}"
