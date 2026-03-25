"""Search tools: glob pattern matching and grep content search."""

import re
from bu_agent_sdk.tools import Depends, tool
from typing import Annotated

from tools.path_resolution import AmbiguousPathError, PathNotFoundError, resolve_target_path
from tools.sandbox import SandboxContext, get_sandbox_context
from tools.text_encoding import read_text_with_fallback


@tool("Find files matching a glob pattern")
async def glob_search(
    pattern: str,
    ctx: Annotated[SandboxContext, Depends(get_sandbox_context)],
    path: str | None = None,
) -> str:
    """Find files matching a glob pattern like **/*.py"""
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
        matches = list(search_dir.glob(pattern))
        files = [str(m.relative_to(ctx.root_dir)) for m in matches if m.is_file()][:50]
        if not files:
            return f"No files match pattern: {pattern}"
        return f"Found {len(files)} file(s):\n" + "\n".join(files)
    except Exception as e:
        return f"Error: {e}"


@tool("Search file contents with regex")
async def grep(
    pattern: str,
    ctx: Annotated[SandboxContext, Depends(get_sandbox_context)],
    path: str | None = None,
) -> str:
    """Search for pattern in files recursively."""
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

    results = []
    if search_target.is_file():
        files_to_search = [search_target]
    else:
        files_to_search = search_target.rglob("*")

    for file_path in files_to_search:
        if not file_path.is_file():
            continue
        try:
            content, _ = read_text_with_fallback(file_path)
            for i, line in enumerate(content.splitlines(), 1):
                if regex.search(line):
                    rel_path = file_path.relative_to(ctx.root_dir)
                    results.append(f"{rel_path}:{i}: {line[:100]}")
                    if len(results) >= 50:
                        return "\n".join(results) + "\n... (truncated)"
        except Exception:
            pass
    return "\n".join(results) if results else f"No matches for: {pattern}"
