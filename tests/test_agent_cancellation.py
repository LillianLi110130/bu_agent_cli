from __future__ import annotations

import asyncio

import pytest

from agent_core import Agent
from cli.worker.runtime_factory import EchoLLM


def _create_agent() -> Agent:
    return Agent(
        llm=EchoLLM(prefix="echo:"),
        tools=[],
        system_prompt="test",
    )


@pytest.mark.asyncio
async def test_run_cancellable_returns_result_without_cancellation() -> None:
    agent = _create_agent()

    result = await agent._run_cancellable(asyncio.sleep(0, result="done"))

    assert result == "done"


@pytest.mark.asyncio
async def test_run_cancellable_exits_quickly_when_cancelled() -> None:
    agent = _create_agent()
    agent._cancel_event = asyncio.Event()

    task = asyncio.create_task(agent._run_cancellable(asyncio.sleep(10)))
    await asyncio.sleep(0.05)
    agent._cancel_event.set()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=0.5)


@pytest.mark.asyncio
async def test_sleep_with_cancel_returns_after_timeout_when_not_cancelled() -> None:
    agent = _create_agent()
    agent._cancel_event = asyncio.Event()

    await asyncio.wait_for(agent._sleep_with_cancel(0.05), timeout=0.3)


@pytest.mark.asyncio
async def test_sleep_with_cancel_exits_quickly_when_cancelled() -> None:
    agent = _create_agent()
    agent._cancel_event = asyncio.Event()

    task = asyncio.create_task(agent._sleep_with_cancel(10))
    await asyncio.sleep(0.05)
    agent._cancel_event.set()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=0.5)
