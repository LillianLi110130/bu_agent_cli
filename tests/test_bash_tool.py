from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

from tools.bash import bash
from tools.sandbox import SandboxContext, get_sandbox_context
from tools.task_cancel import task_cancel
from tools.task_output import task_output


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


@pytest.mark.asyncio
async def test_bash_can_run_command_in_background_and_read_output(workspace_root: Path) -> None:
    ctx = SandboxContext.create(workspace_root)

    result = await bash.execute(
        command=_python_command("import time; time.sleep(0.3); print('ready')"),
        timeout=5,
        run_in_background=True,
        _overrides={get_sandbox_context: lambda: ctx},
    )

    payload = json.loads(result)
    task_id = payload["backgroundTaskId"]
    assert payload["ok"] is True
    assert task_id
    assert payload["persistedOutputPath"]

    output_result = await task_output.execute(
        task_id=task_id,
        wait_for="ready",
        timeout=5,
        _overrides={get_sandbox_context: lambda: ctx},
    )

    output_payload = json.loads(output_result)
    assert output_payload["task_id"] == task_id
    assert output_payload["matched"] is True
    assert "ready" in output_payload["output"]


@pytest.mark.asyncio
async def test_bash_normalizes_false_string_run_in_background(workspace_root: Path) -> None:
    ctx = SandboxContext.create(workspace_root)

    result = await bash.execute(
        command=_python_command("print('hello')"),
        timeout=5,
        run_in_background="false",
        _overrides={get_sandbox_context: lambda: ctx},
    )

    payload = json.loads(result)
    assert payload["ok"] is True
    assert payload["stdout"].strip() == "hello"
    assert payload["backgroundTaskId"] is None
    assert payload["persistedOutputPath"] is None
    assert ctx.shell_task_manager is not None
    assert ctx.shell_task_manager.list_tasks() == []


@pytest.mark.asyncio
async def test_bash_normalizes_true_string_run_in_background(workspace_root: Path) -> None:
    ctx = SandboxContext.create(workspace_root)

    result = await bash.execute(
        command=_python_command("import time; time.sleep(0.1); print('ready')"),
        timeout=5,
        run_in_background="true",
        _overrides={get_sandbox_context: lambda: ctx},
    )

    payload = json.loads(result)
    assert payload["ok"] is True
    assert payload["backgroundTaskId"]
    assert payload["persistedOutputPath"]


@pytest.mark.asyncio
async def test_bash_rejects_invalid_string_run_in_background(workspace_root: Path) -> None:
    ctx = SandboxContext.create(workspace_root)

    result = await bash.execute(
        command=_python_command("print('hello')"),
        timeout=5,
        run_in_background="no",
        _overrides={get_sandbox_context: lambda: ctx},
    )

    assert "run_in_background must be a JSON boolean" in result
    assert ctx.shell_task_manager is not None
    assert ctx.shell_task_manager.list_tasks() == []


@pytest.mark.asyncio
async def test_task_cancel_stops_background_command(workspace_root: Path) -> None:
    ctx = SandboxContext.create(workspace_root)

    result = await bash.execute(
        command=_python_command("import time; time.sleep(30)"),
        timeout=5,
        run_in_background=True,
        _overrides={get_sandbox_context: lambda: ctx},
    )

    payload = json.loads(result)
    task_id = payload["backgroundTaskId"]
    assert task_id

    cancel_result = await task_cancel.execute(
        task_id=task_id,
        _overrides={get_sandbox_context: lambda: ctx},
    )

    assert "Cancelled shell task" in cancel_result
    assert ctx.shell_task_manager is not None
    task = ctx.shell_task_manager.get_task(task_id)
    assert task is not None
    assert task.status == "cancelled"


@pytest.mark.asyncio
async def test_shell_task_manager_shutdown_cancels_running_tasks(workspace_root: Path) -> None:
    ctx = SandboxContext.create(workspace_root)

    result = await bash.execute(
        command=_python_command("import time; time.sleep(30)"),
        timeout=5,
        run_in_background=True,
        _overrides={get_sandbox_context: lambda: ctx},
    )

    payload = json.loads(result)
    task_id = payload["backgroundTaskId"]
    assert task_id
    assert ctx.shell_task_manager is not None

    await asyncio.wait_for(ctx.shell_task_manager.shutdown(cancel_running=True), timeout=3)

    task = ctx.shell_task_manager.get_task(task_id)
    assert task is not None
    assert task.status == "cancelled"
