"""File operation tools: read, write, edit."""

from bu_agent_sdk.tools import Depends, tool
from typing import Annotated, Optional

from tools.sandbox import SandboxContext, get_sandbox_context


@tool("Read contents of a file")
async def read(
    file_path: str,
    ctx: Annotated[SandboxContext, Depends(get_sandbox_context)],
    offset_line: Optional[int] = None,
    n_lines: Optional[int] = None,
) -> str:
    """Read a file and return its contents with line numbers.

    Args:
        file_path: Path to the file to read.
        offset_line: 1-based line number to start reading from. Defaults to 1.
        n_lines: Number of lines to read. Defaults to reading all remaining lines.
    """
    try:
        path = ctx.resolve_path(file_path)
    except Exception as e:
        return f"Security error: {e}"

    if not path.exists():
        return f"File not found: {file_path}"
    if path.is_dir():
        return f"Path is a directory: {file_path}"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        total = len(lines)

        # Determine the slice range
        start = 0
        if offset_line is not None:
            start = max(0, offset_line - 1)  # convert 1-based to 0-based

        end = total
        if n_lines is not None:
            end = min(total, start + n_lines)

        selected = lines[start:end]
        numbered = [
            f"{start + i + 1:4d}  {line}" for i, line in enumerate(selected)
        ]

        header = f"[Lines {start + 1}-{end} of {total}]"
        return header + "\n" + "\n".join(numbered)
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
) -> str:
    """Replace old_string with new_string in a file."""
    try:
        path = ctx.resolve_path(file_path)
    except Exception as e:
        return f"Security error: {e}"

    if not path.exists():
        return f"File not found: {file_path}"
    try:
        content = path.read_text(encoding="utf-8")
        if old_string not in content:
            return f"String not found in {file_path}"
        count = content.count(old_string)
        new_content = content.replace(old_string, new_string)
        path.write_text(new_content, encoding="utf-8")
        return f"Replaced {count} occurrence(s) in {file_path}"
    except Exception as e:
        return f"Error editing file: {e}"
