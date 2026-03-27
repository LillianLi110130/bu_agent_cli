"""Bash command execution tool."""

import asyncio
import json
import locale
import os
import signal
import subprocess
import sys
from contextlib import suppress
from agent_core.tools import Depends, tool
from typing import Annotated

from tools.sandbox import SandboxContext, get_sandbox_context


@tool(
    "Execute a command in the current OS shell and return structured output. "
    "On Windows, use cmd or PowerShell compatible syntax and avoid Unix-only "
    "patterns such as heredoc, python3, or file."
)
async def bash(
    command: str,
    ctx: Annotated[SandboxContext, Depends(get_sandbox_context)],
    timeout: int = 30,
) -> str:
    """Run a command in the current OS shell within the sandbox working directory."""
    try:
        result = await _run_shell_command(
            command,
            cwd=str(ctx.working_dir),
            timeout=timeout,
        )
        return _format_bash_result(
            command=command,
            cwd=str(ctx.working_dir),
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            timed_out=False,
        )
    except asyncio.TimeoutError:
        return _format_bash_result(
            command=command,
            cwd=str(ctx.working_dir),
            returncode=None,
            stdout="",
            stderr=f"Command timed out after {timeout}s",
            timed_out=True,
        )
    except Exception as e:
        return f"Error: {e}"


class _AsyncShellResult:
    def __init__(self, returncode: int, stdout: str, stderr: str):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


async def _run_shell_command(command: str, cwd: str, timeout: int) -> _AsyncShellResult:
    popen_kwargs: dict[str, object] = {
        "shell": True,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "cwd": cwd,
    }
    if sys.platform == "win32":
        popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        popen_kwargs["start_new_session"] = True

    process = subprocess.Popen(command, **popen_kwargs)
    wait_task = asyncio.create_task(asyncio.to_thread(process.wait))

    try:
        done, _ = await asyncio.wait(
            {wait_task},
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if wait_task not in done:
            await _terminate_shell_process(process)
            await asyncio.to_thread(process.wait)
            _close_process_pipes(process)
            raise asyncio.TimeoutError
    except asyncio.CancelledError:
        await _terminate_shell_process(process)
        await asyncio.to_thread(process.wait)
        _close_process_pipes(process)
        raise
    finally:
        if wait_task.done():
            with suppress(Exception):
                await wait_task

    stdout, stderr = await asyncio.to_thread(process.communicate)
    return _AsyncShellResult(
        returncode=process.returncode or 0,
        stdout=_decode_process_stream(stdout),
        stderr=_decode_process_stream(stderr),
    )


async def _terminate_shell_process(process: subprocess.Popen) -> None:
    if process.returncode is not None:
        return

    try:
        if sys.platform == "win32":
            await _terminate_windows_process_tree(process.pid)
        else:
            os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    try:
        await asyncio.wait_for(asyncio.to_thread(process.wait), timeout=1)
        return
    except (asyncio.TimeoutError, ProcessLookupError):
        pass
    except asyncio.CancelledError:
        process.kill()
        raise

    try:
        process.kill()
    except ProcessLookupError:
        return

    try:
        await asyncio.wait_for(asyncio.to_thread(process.wait), timeout=1)
    except (asyncio.TimeoutError, ProcessLookupError):
        pass


async def _terminate_windows_process_tree(pid: int | None) -> None:
    if not pid:
        return

    killer = await asyncio.create_subprocess_exec(
        "taskkill",
        "/PID",
        str(pid),
        "/T",
        "/F",
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await killer.wait()


def _close_process_pipes(process: subprocess.Popen) -> None:
    for stream_name in ("stdout", "stderr", "stdin"):
        stream = getattr(process, stream_name, None)
        if stream is None:
            continue
        try:
            stream.close()
        except Exception:
            pass


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


def _format_bash_result(
    *,
    command: str,
    cwd: str,
    returncode: int | None,
    stdout: str,
    stderr: str,
    timed_out: bool,
) -> str:
    payload = {
        "ok": returncode == 0 and not timed_out,
        "command": command,
        "cwd": cwd,
        "returncode": returncode,
        "timed_out": timed_out,
        "stdout": stdout,
        "stderr": stderr,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
