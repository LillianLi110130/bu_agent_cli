"""File operation tools: read, write, edit."""

from bu_agent_sdk.tools import Depends, tool
from typing import Annotated

from tools.sandbox import SandboxContext, get_sandbox_context


@tool("Read contents of a file")
async def read(
    file_path: str,
    ctx: Annotated[SandboxContext, Depends(get_sandbox_context)],
) -> str:
    """Read a file and return its contents with line numbers."""
    try:
        path = ctx.resolve_path(file_path)
    except Exception as e:
        return f"Security error: {e}"

    if not path.exists():
        return f"File not found: {file_path}"
    if path.is_dir():
        return f"Path is a directory: {file_path}"
    try:
        lines = path.read_text().splitlines()
        numbered = [f"{i + 1:4d}  {line}" for i, line in enumerate(lines)]
        return "\n".join(numbered)
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
        path.write_text(content)
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
        content = path.read_text()
        if old_string not in content:
            return f"String not found in {file_path}"
        count = content.count(old_string)
        new_content = content.replace(old_string, new_string)
        path.write_text(new_content)
        return f"Replaced {count} occurrence(s) in {file_path}"
    except Exception as e:
        return f"Error editing file: {e}"
