"""Bash command execution tool."""

from __future__ import annotations

import asyncio
import json
import subprocess
from typing import Annotated

from agent_core.tools import Depends, tool
from tools.sandbox import SandboxContext, get_sandbox_context
from tools.shell_tasks import decode_process_stream, terminate_process_tree


@tool(
    "Execute a command in the current OS shell and return structured output. "
    "On Windows, use cmd or PowerShell compatible syntax and avoid Unix-only "
    "patterns such as heredoc, python3, or file.",
    context_policy="trim",
    context_max_inline_chars=6400,
)
async def bash(
    command: str,
    ctx: Annotated[SandboxContext, Depends(get_sandbox_context)],
    timeout: int = 30,
    run_in_background: bool = False,
) -> str:
    """Run a command in the current OS shell within the sandbox working directory."""
    try:
        run_in_background = _normalize_run_in_background(run_in_background)
        if run_in_background is None:
            return "Error: run_in_background must be a JSON boolean true/false, not a string."

        if run_in_background:
            if ctx.shell_task_manager is None:
                return "Error: Shell task manager not initialized"
            task = await ctx.shell_task_manager.start(command=command, cwd=str(ctx.working_dir))
            return _format_bash_result(
                command=command,
                cwd=str(ctx.working_dir),
                returncode=0,
                stdout="",
                stderr="",
                timed_out=False,
                interrupted=False,
                background_task_id=task.task_id,
                persisted_output_path=str(task.log_path),
            )

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
            interrupted=False,
        )
    except asyncio.TimeoutError:
        return _format_bash_result(
            command=command,
            cwd=str(ctx.working_dir),
            returncode=None,
            stdout="",
            stderr=f"Command timed out after {timeout}s",
            timed_out=True,
            interrupted=False,
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
        "stdin": subprocess.DEVNULL,
    }
    if subprocess._mswindows:
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
            await terminate_process_tree(process)
            await asyncio.to_thread(process.wait)
            _close_process_pipes(process)
            raise asyncio.TimeoutError
    except asyncio.CancelledError:
        await terminate_process_tree(process)
        await asyncio.to_thread(process.wait)
        _close_process_pipes(process)
        raise
    finally:
        if wait_task.done():
            try:
                await wait_task
            except Exception:
                pass

    stdout, stderr = await asyncio.to_thread(process.communicate)
    return _AsyncShellResult(
        returncode=process.returncode or 0,
        stdout=_decode_process_stream(stdout),
        stderr=_decode_process_stream(stderr),
    )


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
    return decode_process_stream(data)


def _normalize_run_in_background(value: bool | str) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    return None


def _format_bash_result(
    *,
    command: str,
    cwd: str,
    returncode: int | None,
    stdout: str,
    stderr: str,
    timed_out: bool,
    interrupted: bool,
    background_task_id: str | None = None,
    persisted_output_path: str | None = None,
) -> str:
    payload = {
        "ok": returncode == 0 and not timed_out and not interrupted,
        "command": command,
        "cwd": cwd,
        "returncode": returncode,
        "timed_out": timed_out,
        "interrupted": interrupted,
        "stdout": stdout,
        "stderr": stderr,
        "backgroundTaskId": background_task_id,
        "persistedOutputPath": persisted_output_path,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
