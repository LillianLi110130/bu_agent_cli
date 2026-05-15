"""Search tools: glob pattern matching and grep content search."""

from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path, PurePosixPath
from typing import Annotated, Iterable, Iterator, Literal

from agent_core.tools import Depends, tool
from tools.path_resolution import AmbiguousPathError, PathNotFoundError, resolve_target_path
from tools.sandbox import SandboxContext, get_sandbox_context
from tools.text_encoding import candidate_encodings

GlobKind = Literal["file", "dir", "any"]
_GLOB_RESULT_LIMIT = 50
_GREP_RESULT_LIMIT = 50
_SEARCH_BINARY_SAMPLE_SIZE = 4096
_SEARCH_MAX_FILE_SIZE_BYTES = 2 * 1024 * 1024


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
        dirs[:] = sorted(name for name in dirs if not ctx.is_ignored(current_path / name))
        for file_name in sorted(files):
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


def _normalize_glob_pattern(pattern: str) -> str:
    pattern_text = pattern.strip().replace("\\", "/")
    if not pattern_text:
        raise ValueError("pattern cannot be empty")
    pattern_path = Path(pattern_text)
    if pattern_text.startswith("/") or pattern_path.is_absolute() or pattern_path.drive:
        raise ValueError("pattern must be relative to the search path")
    return pattern_text


def _split_glob_parts(pattern: str) -> tuple[str, ...]:
    return tuple(part for part in PurePosixPath(pattern).parts if part not in ("", "."))


def _matches_glob_parts(path_parts: tuple[str, ...], pattern_parts: tuple[str, ...]) -> bool:
    if not pattern_parts:
        return not path_parts

    head = pattern_parts[0]
    if head == "**":
        if _matches_glob_parts(path_parts, pattern_parts[1:]):
            return True
        return bool(path_parts) and _matches_glob_parts(path_parts[1:], pattern_parts)

    return (
        bool(path_parts)
        and fnmatch.fnmatchcase(path_parts[0], head)
        and _matches_glob_parts(path_parts[1:], pattern_parts[1:])
    )


def _matches_glob_pattern(relative_path: PurePosixPath, pattern: str) -> bool:
    return _matches_glob_parts(relative_path.parts, _split_glob_parts(pattern))


def _iter_glob_candidates(root: Path, ctx: SandboxContext) -> Iterator[Path]:
    resolved_root = root.resolve()
    if ctx.is_ignored(resolved_root):
        return
    if resolved_root.is_file():
        if _is_searchable_path(resolved_root, ctx, kind="file"):
            yield resolved_root
        return

    for current, dirs, files in os.walk(resolved_root):
        current_path = Path(current)
        dirs[:] = sorted(name for name in dirs if not ctx.is_ignored(current_path / name))
        for dir_name in dirs:
            candidate = current_path / dir_name
            if _is_searchable_path(candidate, ctx, kind="dir"):
                yield candidate
        for file_name in sorted(files):
            candidate = current_path / file_name
            if _is_searchable_path(candidate, ctx, kind="file"):
                yield candidate


def _iter_glob_matches(root: Path, pattern: str, ctx: SandboxContext, kind: GlobKind) -> Iterator[Path]:
    pattern_text = _normalize_glob_pattern(pattern)
    for candidate in _iter_glob_candidates(root, ctx):
        if not _matches_kind(candidate, kind):
            continue
        try:
            relative = candidate.relative_to(root).as_posix()
        except ValueError:
            continue
        if _matches_glob_pattern(PurePosixPath(relative), pattern_text):
            yield candidate


def _take_limited_matches(matches: Iterable[Path], limit: int) -> tuple[list[Path], bool]:
    page: list[Path] = []
    for candidate in matches:
        if len(page) < limit:
            page.append(candidate)
            continue
        return page, True
    return page, False


def _display_path(path: Path, ctx: SandboxContext) -> str:
    try:
        return str(path.relative_to(ctx.root_dir))
    except ValueError:
        return str(path)


def _render_glob_results(
    matches: list[str],
    *,
    result_kind: str,
    has_more: bool,
    pattern: str,
) -> str:
    if not matches:
        return f"No {result_kind}s match pattern: {pattern}"

    if has_more:
        return (
            f"Found {len(matches)} {result_kind}(s), showing first {len(matches)}:\n"
            + "\n".join(matches)
            + "\n... (truncated; refine pattern or path)"
        )

    return f"Found {len(matches)} {result_kind}(s):\n" + "\n".join(matches)


def _is_binary_file(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            sample = handle.read(_SEARCH_BINARY_SAMPLE_SIZE)
    except OSError:
        return True
    if not sample:
        return False
    if b"\x00" in sample:
        return True
    return False


def _should_search_file(path: Path) -> bool:
    try:
        if path.stat().st_size > _SEARCH_MAX_FILE_SIZE_BYTES:
            return False
    except OSError:
        return False
    return not _is_binary_file(path)


def _iter_matching_lines(path: Path, regex: re.Pattern[str]) -> Iterator[tuple[int, str]]:
    for encoding in candidate_encodings():
        try:
            with path.open("r", encoding=encoding) as handle:
                for line_number, line in enumerate(handle, 1):
                    if regex.search(line):
                        yield line_number, line.rstrip("\r\n")
            return
        except UnicodeDecodeError:
            continue
        except OSError:
            return


def _render_grep_results(
    results: list[str],
    *,
    has_more: bool,
    pattern: str,
) -> str:
    if not results:
        return f"No matches for: {pattern}"
    if has_more:
        return "\n".join(results) + "\n... (truncated)"
    return "\n".join(results)


@tool(
    "Find files or directories matching a glob pattern",
    context_policy="trim",
    context_max_inline_chars=6400,
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
        page, has_more = _take_limited_matches(
            _iter_glob_matches(search_dir, pattern, ctx, kind),
            _GLOB_RESULT_LIMIT,
        )
        page.sort(key=lambda item: _sort_key_for_glob_match(item, search_dir))
        matches = [_display_path(candidate, ctx) for candidate in page]
        result_kind = "path" if kind == "any" else kind
        return _render_glob_results(
            matches,
            result_kind=result_kind,
            has_more=has_more,
            pattern=pattern,
        )
    except Exception as e:
        return f"Error: {e}"


@tool("Search file contents with regex", context_policy="trim", context_max_inline_chars=6400)
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
        if not _should_search_file(file_path):
            continue
        for line_number, line in _iter_matching_lines(file_path, regex):
            if len(results) < _GREP_RESULT_LIMIT:
                results.append(f"{_display_path(file_path, ctx)}:{line_number}: {line[:100]}")
                continue
            return _render_grep_results(
                results,
                has_more=True,
                pattern=pattern,
            )
    return _render_grep_results(
        results,
        has_more=False,
        pattern=pattern,
    )
