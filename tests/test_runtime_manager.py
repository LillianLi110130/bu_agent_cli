from __future__ import annotations

import importlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tools.sandbox import SandboxContext


class DummyAgent:
    async def query(self, message: str) -> str:
        return message


def _load_module(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        pytest.fail(f"Expected module '{module_name}' to exist: {exc}")


@pytest.mark.asyncio
async def test_runtime_manager_reuses_runtime_for_same_session_key(tmp_path: Path) -> None:
    manager_module = _load_module("bu_agent_sdk.runtime.manager")

    created = 0

    def runtime_factory():
        nonlocal created
        created += 1
        return DummyAgent(), SandboxContext.create(tmp_path)

    manager = manager_module.RuntimeManager(runtime_factory=runtime_factory)

    runtime_a = await manager.get_or_create_runtime("telegram:chat-1")
    runtime_b = await manager.get_or_create_runtime("telegram:chat-1")

    assert runtime_a is runtime_b
    assert created == 1
    assert manager.runtime_count == 1


@pytest.mark.asyncio
async def test_runtime_manager_creates_new_runtime_after_clear(tmp_path: Path) -> None:
    manager_module = _load_module("bu_agent_sdk.runtime.manager")

    created = 0

    def runtime_factory():
        nonlocal created
        created += 1
        return DummyAgent(), SandboxContext.create(tmp_path)

    manager = manager_module.RuntimeManager(runtime_factory=runtime_factory)

    first_runtime = await manager.get_or_create_runtime("telegram:chat-1")
    cleared = await manager.clear_runtime("telegram:chat-1")
    second_runtime = await manager.get_or_create_runtime("telegram:chat-1")

    assert cleared is True
    assert first_runtime is not second_runtime
    assert first_runtime.context.session_id != second_runtime.context.session_id
    assert created == 2


@pytest.mark.asyncio
async def test_runtime_manager_cleanup_removes_expired_runtimes(tmp_path: Path) -> None:
    manager_module = _load_module("bu_agent_sdk.runtime.manager")

    def runtime_factory():
        return DummyAgent(), SandboxContext.create(tmp_path)

    manager = manager_module.RuntimeManager(runtime_factory=runtime_factory)

    fresh_runtime = await manager.get_or_create_runtime("telegram:fresh")
    stale_runtime = await manager.get_or_create_runtime("telegram:stale")
    stale_runtime.last_used_at = datetime.now(UTC) - timedelta(minutes=61)
    fresh_runtime.last_used_at = datetime.now(UTC)

    removed = await manager.cleanup_expired_runtimes(timeout_minutes=60)

    assert removed == 1
    assert manager.runtime_count == 1
    assert await manager.get_runtime("telegram:stale") is None
    assert await manager.get_runtime("telegram:fresh") is fresh_runtime
