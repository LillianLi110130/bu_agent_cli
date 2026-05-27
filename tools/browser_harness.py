"""Dedicated browser-harness execution tool."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from typing import Annotated

from agent_core.tools import Depends, tool
from tools.sandbox import SandboxContext, get_sandbox_context
from tools.shell_tasks import terminate_process_tree


@tool(
    "Run Python code in browser-harness by passing it directly to stdin. "
    'Before calling this tool for browser work, load the browser skill with '
    'skill_view(name="browser") unless its SKILL.md is already in context. '
    "Use this as the repository browser automation entrypoint across platforms.",
    name="browser_harness",
    context_policy="trim",
    context_max_inline_chars=12800,
)
async def browser_harness(
    script: str,
    ctx: Annotated[SandboxContext, Depends(get_sandbox_context)],
    timeout: int = 60,
) -> str:
    """Run browser-harness without invoking the OS shell."""
    try:
        result = await _run_browser_harness(
            script=script,
            cwd=str(ctx.working_dir),
            timeout=timeout,
        )
    except FileNotFoundError:
        return "Error: browser-harness not found on PATH"
    except asyncio.TimeoutError:
        return _format_browser_harness_result(
            returncode=None,
            stdout="",
            stderr=f"browser-harness timed out after {timeout}s",
            timed_out=True,
        )
    return _format_browser_harness_result(
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        timed_out=False,
    )


class _BrowserHarnessResult:
    def __init__(self, returncode: int, stdout: str, stderr: str):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


async def _run_browser_harness(*, script: str, cwd: str, timeout: int) -> _BrowserHarnessResult:
    popen_kwargs: dict[str, object] = {
        "stdin": subprocess.PIPE,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "cwd": cwd,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "env": _browser_harness_env(),
        "shell": False,
    }
    if subprocess._mswindows:
        popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        popen_kwargs["start_new_session"] = True

    process = subprocess.Popen(["browser-harness"], **popen_kwargs)
    communicate_task = asyncio.create_task(asyncio.to_thread(process.communicate, script))

    try:
        stdout, stderr = await asyncio.wait_for(communicate_task, timeout=timeout)
    except asyncio.TimeoutError:
        await terminate_process_tree(process)
        communicate_task.cancel()
        raise
    except asyncio.CancelledError:
        await terminate_process_tree(process)
        communicate_task.cancel()
        raise

    return _BrowserHarnessResult(
        returncode=process.returncode or 0,
        stdout=stdout or "",
        stderr=stderr or "",
    )


def _format_browser_harness_result(
    *,
    returncode: int | None,
    stdout: str,
    stderr: str,
    timed_out: bool,
) -> str:
    payload = {
        "ok": returncode == 0 and not timed_out,
        "returncode": returncode,
        "stdout": _replace_surrogates(stdout),
        "stderr": _replace_surrogates(stderr),
        "timed_out": timed_out,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _replace_surrogates(value: str) -> str:
    return value.encode("utf-8", "replace").decode("utf-8", "replace")


def _browser_harness_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8:replace"
    env["PYTHONUTF8"] = "1"
    return env
