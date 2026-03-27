from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys

import pytest

from tools.bash import bash
from tools.sandbox import SandboxContext, get_sandbox_context


def _python_command(source: str) -> str:
    return f'"{sys.executable}" -c "{source}"'


@pytest.fixture
def workspace_root() -> Path:
    return Path.cwd().resolve()


@pytest.mark.asyncio
async def test_bash_returns_structured_output_for_success(workspace_root: Path) -> None:
    ctx = SandboxContext.create(workspace_root)

    result = await bash.execute(
        command=_python_command("print('hello')"),
        timeout=5,
        _overrides={get_sandbox_context: lambda: ctx},
    )

    payload = json.loads(result)
    assert payload["ok"] is True
    assert payload["returncode"] == 0
    assert payload["stdout"].strip() == "hello"
    assert payload["stderr"] == ""


@pytest.mark.asyncio
async def test_bash_returns_timeout_payload(workspace_root: Path) -> None:
    ctx = SandboxContext.create(workspace_root)

    result = await bash.execute(
        command=_python_command("import time; time.sleep(2)"),
        timeout=1,
        _overrides={get_sandbox_context: lambda: ctx},
    )

    payload = json.loads(result)
    assert payload["ok"] is False
    assert payload["timed_out"] is True
    assert payload["returncode"] is None
    assert "timed out" in payload["stderr"].lower()


@pytest.mark.asyncio
async def test_bash_cancellation_stops_long_running_command(workspace_root: Path) -> None:
    ctx = SandboxContext.create(workspace_root)

    task = asyncio.create_task(
        bash.execute(
            command=_python_command("import time; time.sleep(10)"),
            timeout=30,
            _overrides={get_sandbox_context: lambda: ctx},
        )
    )
    await asyncio.sleep(0.1)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=2.5)
