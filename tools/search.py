"""Search tools: glob pattern matching and grep content search."""

from __future__ import annotations

import os
import re
from pathlib import Path, PurePosixPath
from typing import Annotated, Iterable

from agent_core.tools import Depends, tool

from tools.path_resolution import AmbiguousPathError, PathNotFoundError, resolve_target_path
from tools.sandbox import SandboxContext, get_sandbox_context
from tools.text_encoding import read_text_with_fallback


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


def _matches_glob(root: Path, candidate: Path, pattern: str) -> bool:
    pattern_text = pattern.replace("\\", "/")
    relative = candidate.relative_to(root).as_posix()
    relative_path = PurePosixPath(relative)
    candidate_name = PurePosixPath(candidate.name)
    # Keep pathlib's native glob semantics for the common case.
    if relative_path.match(pattern_text) or candidate_name.match(pattern_text):
        return True
    if pattern_text.startswith("**/"):
        # pathlib does not treat patterns like "**/*.docx" as matching files that
        # live directly under the search root, so retry without the recursive prefix.
        pattern_without_recursive_prefix = pattern_text[3:]
        return relative_path.match(pattern_without_recursive_prefix) or candidate_name.match(
            pattern_without_recursive_prefix
        )
    return False


def _display_path(path: Path, ctx: SandboxContext) -> str:
    try:
        return str(path.relative_to(ctx.root_dir))
    except ValueError:
        return str(path)


@tool("Find files matching a glob pattern", context_policy="trim", context_max_inline_chars=1600)
async def glob_search(
    pattern: str,
    ctx: Annotated[SandboxContext, Depends(get_sandbox_context)],
    path: str | None = None,
) -> str:
    """Find files matching a glob pattern like ``**/*.py``."""
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
        matches = [
            _display_path(candidate, ctx)
            for candidate in _iter_searchable_files(search_dir, ctx)
            if _matches_glob(search_dir, candidate, pattern)
        ][:50]
        if not matches:
            return f"No files match pattern: {pattern}"
        return f"Found {len(matches)} file(s):\n" + "\n".join(matches)
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
