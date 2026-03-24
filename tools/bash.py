"""Bash command execution tool."""

import locale
import subprocess
import sys
from bu_agent_sdk.tools import Depends, tool
from typing import Annotated

from tools.sandbox import SandboxContext, get_sandbox_context


@tool("Execute a shell command and return output")
async def bash(
    command: str,
    ctx: Annotated[SandboxContext, Depends(get_sandbox_context)],
    timeout: int = 30,
) -> str:
    """Run a bash command in the sandbox working directory."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            timeout=timeout,
            cwd=str(ctx.working_dir),
        )
        stdout = _decode_process_stream(result.stdout)
        stderr = _decode_process_stream(result.stderr)
        output = stdout + stderr
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout}s"
    except Exception as e:
        return f"Error: {e}"


def _decode_process_stream(data: bytes | str | None) -> str:
    if data is None:
        return ""
    if isinstance(data, str):
        return data
    for encoding in _shell_output_encodings():
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _shell_output_encodings() -> list[str]:
    encodings = ["utf-8", "utf-8-sig"]

    preferred = locale.getpreferredencoding(False)
    if preferred:
        encodings.append(preferred)

    if sys.platform == "win32":
        codepage = _windows_console_encoding()
        if codepage:
            encodings.append(codepage)
        encodings.extend(["gb18030", "gbk", "cp936"])

    unique: list[str] = []
    seen: set[str] = set()
    for encoding in encodings:
        lowered = encoding.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        unique.append(encoding)
    return unique


def _windows_console_encoding() -> str | None:
    if sys.platform != "win32":
        return None

    try:
        import ctypes

        codepage = ctypes.windll.kernel32.GetConsoleOutputCP()
    except Exception:
        return None

    if not codepage:
        return None
    return f"cp{codepage}"
