from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from agent_core.agent.config import AgentConfig
from agent_core.agent.subagent_manager import SubagentManager
from agent_core.agent.events import FinalResponseEvent


class _FakeRegistry:
    def __init__(self, config: AgentConfig) -> None:
        self._config = config

    def get_config(self, name: str) -> AgentConfig | None:
        if name == self._config.name:
            return self._config
        return None


class _FakeSubagent:
    async def query_stream(self, prompt: str) -> AsyncIterator[FinalResponseEvent]:
        yield FinalResponseEvent(content=f"handled:{prompt}")


class _FakeMainAgent:
    pass


@pytest.mark.asyncio
async def test_run_and_wait_returns_result_without_main_agent_injection_support(
    tmp_path: Path,
) -> None:
    config = AgentConfig(
        name="doc_reader",
        description="Read docs",
        mode="subagent",
        system_prompt="subagent prompt",
    )
    registry = _FakeRegistry(config)
    manager = SubagentManager(
        agent_factory=lambda config, parent_ctx, all_tools: _FakeSubagent(),
        registry=registry,
        all_tools=[],
        workspace=tmp_path,
        context=object(),
    )

    result = await manager.run_and_wait(
        subagent_name="doc_reader",
        prompt="read the document",
        timeout=1.0,
    )

    assert result.status == "completed"
    assert result.final_response == "handled:read the document"


def test_set_main_agent_is_supported_for_bootstrap_compatibility(tmp_path: Path) -> None:
    config = AgentConfig(
        name="doc_reader",
        description="Read docs",
        mode="subagent",
        system_prompt="subagent prompt",
    )
    registry = _FakeRegistry(config)
    manager = SubagentManager(
        agent_factory=lambda config, parent_ctx, all_tools: _FakeSubagent(),
        registry=registry,
        all_tools=[],
        workspace=tmp_path,
        context=object(),
    )

    main_agent = _FakeMainAgent()
    manager.set_main_agent(main_agent)  # type: ignore[arg-type]

    assert manager._main_agent is main_agent
