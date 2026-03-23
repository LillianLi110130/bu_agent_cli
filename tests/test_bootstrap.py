import importlib
from pathlib import Path

import pytest


class DummyLLM:
    def __init__(self, model: str = "dummy-model") -> None:
        self.model = model

    @property
    def provider(self) -> str:
        return "dummy"

    @property
    def name(self) -> str:
        return self.model

    async def ainvoke(self, messages, tools=None, tool_choice=None, **kwargs):
        raise NotImplementedError

    async def astream(self, messages, tools=None, tool_choice=None, **kwargs):
        if False:
            yield None


def _load_module(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        pytest.fail(f"Expected module '{module_name}' to exist: {exc}")


def test_build_system_prompt_uses_packaged_skills_outside_workspace(tmp_path: Path) -> None:
    module = _load_module("bu_agent_sdk.bootstrap.agent_factory")

    prompt = module.build_system_prompt(tmp_path)

    assert "brainstorming" in prompt
    assert str(tmp_path) in prompt


def test_sync_workspace_agents_md_deduplicates_and_replaces_content(tmp_path: Path) -> None:
    module = _load_module("bu_agent_sdk.bootstrap.session_bootstrap")
    sdk_module = _load_module("bu_agent_sdk")
    state = module.WorkspaceInstructionState()
    agent = sdk_module.Agent(llm=DummyLLM(), tools=[], system_prompt="system prompt")

    agents_md_path = tmp_path / "AGENTS.md"
    agents_md_path.write_text("first rule", encoding="utf-8")

    state = module.sync_workspace_agents_md(agent, tmp_path, state)
    state = module.sync_workspace_agents_md(agent, tmp_path, state)

    first_dev_messages = [
        message
        for message in agent.messages
        if message.role == "developer" and getattr(message, "content", "") == "first rule"
    ]
    system_messages = [message for message in agent.messages if message.role == "system"]

    assert len(first_dev_messages) == 1
    assert len(system_messages) == 1
    assert state.injected_content == "first rule"

    agents_md_path.write_text("second rule", encoding="utf-8")
    state = module.sync_workspace_agents_md(agent, tmp_path, state)

    developer_contents = [
        getattr(message, "content", "") for message in agent.messages if message.role == "developer"
    ]

    assert "first rule" not in developer_contents
    assert "second rule" in developer_contents
    assert state.injected_content == "second rule"
