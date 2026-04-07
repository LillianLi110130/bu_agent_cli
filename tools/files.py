"""File operation tools: read, write, edit."""

import difflib
from agent_core.tools import Depends, tool
from typing import Annotated, Optional

from tools.path_resolution import AmbiguousPathError, PathNotFoundError, resolve_target_path
from tools.sandbox import SandboxContext, get_sandbox_context
from tools.text_encoding import read_text_with_fallback

# Constants
_MAX_FILE_CHARS = 128_000  # ~128KB - prevents OOM from reading huge files
_MAX_ARTIFACT_WINDOW_LINES = 200


def _build_diff_error(old_text: str, content: str, file_path: str) -> str:
    """Build a helpful error message with diff when old_text is not found."""
    lines = content.splitlines(keepends=True)
    old_lines = old_text.splitlines(keepends=True)
    window = len(old_lines)

    best_ratio, best_start = 0.0, 0
    for i in range(max(1, len(lines) - window + 1)):
        ratio = difflib.SequenceMatcher(None, old_lines, lines[i : i + window]).ratio()
        if ratio > best_ratio:
            best_ratio, best_start = ratio, i

    if best_ratio > 0.5:
        diff = "\n".join(
            difflib.unified_diff(
                old_lines,
                lines[best_start : best_start + window],
                fromfile="old_string (provided)",
                tofile=f"{file_path} (actual, line {best_start + 1})",
                lineterm="",
            )
        )
        return f"Error: old_string not found.\nBest match ({best_ratio:.0%} similar) at line {best_start + 1}:\n{diff}"
    return f"Error: old_string not found in {file_path}. No similar text found."


def _validate_artifact_window(
    *,
    path,
    ctx: SandboxContext,
    offset_line: Optional[int],
    n_lines: Optional[int],
) -> str | None:
    if not ctx.is_runtime_artifact_path(path):
        return None

    if offset_line is None or n_lines is None:
        return (
            "Error: reading runtime artifact requires explicit offset_line and n_lines. "
            "Use a narrow slice, for example offset_line=1, n_lines=80."
        )

    if n_lines <= 0:
        return "Error: n_lines must be a positive integer when reading a runtime artifact."

    if n_lines > _MAX_ARTIFACT_WINDOW_LINES:
        return (
            f"Error: runtime artifact reads are limited to {_MAX_ARTIFACT_WINDOW_LINES} lines per call. "
            f"Requested n_lines={n_lines}."
        )

    if offset_line <= 0:
        return "Error: offset_line must be a positive integer when reading a runtime artifact."

    return None


@tool("Read contents of a file", context_policy="trim", context_max_inline_chars=2200)
async def read(
    file_path: str,
    ctx: Annotated[SandboxContext, Depends(get_sandbox_context)],
    offset_line: Optional[int] = None,
    n_lines: Optional[int] = None,
) -> str:
    """Read a file with optional line range and size limit.

    Args:
        file_path: Path to the file to read.
        offset_line: 1-based line number to start reading from. Defaults to 1.
        n_lines: Number of lines to read. Defaults to reading all remaining lines.
    """
    try:
        path = resolve_target_path(file_path, ctx, kind="file")
    except Exception as e:
        if isinstance(e, (PathNotFoundError, AmbiguousPathError)):
            return str(e)
        return f"Security error: {e}"

    artifact_window_error = _validate_artifact_window(
        path=path,
        ctx=ctx,
        offset_line=offset_line,
        n_lines=n_lines,
    )
    if artifact_window_error:
        return artifact_window_error

    try:
        # Only check file size if no line range is specified
        if n_lines is None and offset_line is None:
            size = path.stat().st_size
            if size > _MAX_FILE_CHARS * 4:
                return (
                    f"Error: File too large ({size:,} bytes). "
                    f"Use offset_line and n_lines to read portions."
                )

        content, _ = read_text_with_fallback(path)
        lines = content.splitlines()
        total = len(lines)

        # Determine the slice range
        start = max(0, (offset_line or 1) - 1)  # convert 1-based to 0-based
        end = total if n_lines is None else min(total, start + n_lines)

        selected = lines[start:end]
        numbered = [f"{start + i + 1:4d}  {line}" for i, line in enumerate(selected)]

        result = f"[Lines {start + 1}-{end} of {total}]\n" + "\n".join(numbered)

        # Check result size after applying line range
        if len(result) > _MAX_FILE_CHARS:
            return result[:_MAX_FILE_CHARS] + (
                f"\n\n... (truncated — result is {len(result):,} chars, "
                f"use smaller n_lines to read less)"
            )

        return result
    except Exception as e:
        return f"Error reading file: {e}"


@tool("Write content to a file")
async def write(
    file_path: str,
    content: str,
    ctx: Annotated[SandboxContext, Depends(get_sandbox_context)],
) -> str:
    """Write content to a file, creating directories if needed."""
    try:
        path = ctx.resolve_path(file_path)
    except Exception as e:
        return f"Security error: {e}"

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} bytes to {file_path}"
    except Exception as e:
        return f"Error writing file: {e}"


@tool("Replace text in a file")
async def edit(
    file_path: str,
    old_string: str,
    new_string: str,
    ctx: Annotated[SandboxContext, Depends(get_sandbox_context)],
    replace_all: bool = False,
) -> str:
    """Replace old_string with new_string in a file.

    Args:
        file_path: Path to the file to edit.
        old_string: The exact text to find and replace.
        new_string: The text to replace with.
        replace_all: If False, only replaces first occurrence and warns on duplicates.
                     If True, replaces all occurrences.
    """
    try:
        path = resolve_target_path(file_path, ctx, kind="file")
    except Exception as e:
        if isinstance(e, (PathNotFoundError, AmbiguousPathError)):
            return str(e)
        return f"Security error: {e}"

    try:
        content, encoding = read_text_with_fallback(path)
        if old_string not in content:
            return _build_diff_error(old_string, content, file_path)

        count = content.count(old_string)

        # Safety check: warn on multiple occurrences
        if count > 1 and not replace_all:
            return (
                f"Warning: old_string appears {count} times. "
                f"Please provide more context to make it unique, or set replace_all=True."
            )

        new_content = content.replace(old_string, new_string, 1 if not replace_all else count)
        path.write_text(new_content, encoding=encoding)

        return f"Replaced {1 if not replace_all else count} occurrence(s) in {file_path}"
    except Exception as e:
        return f"Error editing file: {e}"
