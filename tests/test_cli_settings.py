from __future__ import annotations

import io
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from rich.console import Console

import cli.app as app_module
from agent_core import Agent
from agent_core.llm.messages import BaseMessage
from agent_core.llm.views import ChatInvokeCompletion, ChatInvokeCompletionChunk
from agent_core.runtime_paths import get_default_workspace
from cli.app import TGAgentCLI
from cli.slash_commands import SlashCommandRegistry
from tools import SandboxContext


class _DummyPrompter:
    def __init__(self, console):
        self.console = console


class FakeLLM:
    model = "fake-model"

    @property
    def provider(self) -> str:
        return "fake"

    @property
    def name(self) -> str:
        return self.model

    async def ainvoke(
        self,
        messages: list[BaseMessage],
        tools=None,
        tool_choice=None,
        **kwargs,
    ) -> ChatInvokeCompletion:
        del messages, tools, tool_choice, kwargs
        return ChatInvokeCompletion(content="ok")

    async def astream(
        self,
        messages: list[BaseMessage],
        tools=None,
        tool_choice=None,
        **kwargs,
    ) -> AsyncIterator[ChatInvokeCompletionChunk]:
        del messages, tools, tool_choice, kwargs
        if False:
            yield ChatInvokeCompletionChunk()


def _make_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[TGAgentCLI, io.StringIO]:
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(app_module, "InteractivePrompter", _DummyPrompter)
    agent = Agent(llm=FakeLLM(), tools=[], system_prompt="system prompt")
    context = SandboxContext.create(tmp_path / "workspace")
    cli = TGAgentCLI(
        agent=agent,
        context=context,
        slash_registry=SlashCommandRegistry(),
    )
    output = io.StringIO()
    cli._console = Console(file=output, force_terminal=False, color_system=None, width=160)
    return cli, output


@pytest.mark.asyncio
async def test_settings_command_can_set_default_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli, output = _make_cli(tmp_path, monkeypatch)
    target_workspace = tmp_path / "target-workspace"
    target_workspace.mkdir()

    handled = await cli._handle_slash_command("/settings")
    assert handled is True

    await cli._execute_input_text("1")
    await cli._execute_input_text("1")
    await cli._execute_input_text(str(target_workspace))

    assert get_default_workspace() == target_workspace.resolve()
    assert "默认工作区已保存：" in output.getvalue()
    assert cli._settings_handler.active is True


@pytest.mark.asyncio
async def test_settings_command_can_clear_default_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli, output = _make_cli(tmp_path, monkeypatch)
    home_settings_workspace = tmp_path / "preconfigured-workspace"
    home_settings_workspace.mkdir()

    settings_dir = tmp_path / "home" / ".tg_agent"
    settings_dir.mkdir(parents=True, exist_ok=True)
    (settings_dir / "settings.json").write_text(
        '{\n  "default_workspace": "%s"\n}\n' % home_settings_workspace.resolve().as_posix(),
        encoding="utf-8",
    )

    handled = await cli._handle_slash_command("/settings")
    assert handled is True

    await cli._execute_input_text("1")
    await cli._execute_input_text("2")
    await cli._execute_input_text("y")

    assert get_default_workspace() is None
    assert "默认工作区已清除。" in output.getvalue()
