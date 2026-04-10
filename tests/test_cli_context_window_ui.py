from __future__ import annotations

import io
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import prompt_toolkit
import pytest
from prompt_toolkit.formatted_text import to_formatted_text
from rich.console import Console

import cli.app as app_module
from agent_core import Agent
from agent_core.agent import FinalResponseEvent, TextEvent, ToolCallEvent, ToolResultEvent
from agent_core.llm.messages import BaseMessage, UserMessage
from agent_core.llm.views import ChatInvokeCompletion, ChatInvokeCompletionChunk
from cli.app import TGAgentCLI
from cli.session_runtime import CLISessionRuntime
from cli.slash_commands import SlashCommandRegistry
from cli.worker.runtime_factory import EchoLLM
from tools import SandboxContext


class _DummyPrompter:
    def __init__(self, console):
        self.console = console


class _FakeCompactionLLM:
    def __init__(self, response_text: str) -> None:
        self.model = "fake-compaction-model"
        self._response_text = response_text

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
        **kwargs: Any,
    ) -> ChatInvokeCompletion:
        del messages, tools, tool_choice, kwargs
        return ChatInvokeCompletion(content=self._response_text)

    async def astream(
        self,
        messages: list[BaseMessage],
        tools=None,
        tool_choice=None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatInvokeCompletionChunk]:
        del messages, tools, tool_choice, kwargs
        if False:
            yield ChatInvokeCompletionChunk()


def _render_plain_text(renderable: Any) -> str:
    console = Console(file=io.StringIO(), force_terminal=False, color_system=None, width=120)
    console.print(renderable)
    return console.file.getvalue()


@pytest.mark.asyncio
async def test_cli_run_wires_bottom_toolbar_and_persists_context_window_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TG_AGENT_HOME", str(tmp_path / ".tg_agent"))
    monkeypatch.setattr(app_module, "InteractivePrompter", _DummyPrompter)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    ctx = SandboxContext.create(workspace)
    runtime = CLISessionRuntime.create_for_context(ctx)
    cli = TGAgentCLI(
        agent=Agent(llm=EchoLLM(prefix="echo:"), tools=[], system_prompt="test"),
        context=ctx,
        slash_registry=SlashCommandRegistry(),
        session_runtime=runtime,
    )

    class _FakePromptSession:
        async def prompt_async(self):
            raise EOFError()

    captured_kwargs: dict[str, Any] = {}

    def _fake_prompt_session(**kwargs: Any):
        captured_kwargs.update(kwargs)
        return _FakePromptSession()

    monkeypatch.setattr(prompt_toolkit, "PromptSession", _fake_prompt_session)

    await cli.run()

    assert callable(captured_kwargs["bottom_toolbar"])
    toolbar_fragments = to_formatted_text(captured_kwargs["bottom_toolbar"]())
    toolbar_text = "".join(fragment[1] for fragment in toolbar_fragments)
    assert "模型: echo-worker" in toolbar_text
    assert "上下文:" in toolbar_text
    assert "状态:" in toolbar_text

    snapshot = json.loads(runtime.context_window_status_path.read_text(encoding="utf-8"))
    assert snapshot["session_id"] == runtime.session_id
    assert snapshot["model"] == "echo-worker"
    assert snapshot["runtime_state"] == "idle"
    assert snapshot["status"] == "正常"
    assert snapshot["context_limit"] > 0
    assert snapshot["estimated_tokens"] == 0
    assert snapshot["last_updated_at"] is not None


@pytest.mark.asyncio
async def test_compaction_callback_updates_status_snapshot_and_console(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TG_AGENT_HOME", str(tmp_path / ".tg_agent"))
    monkeypatch.setattr(app_module, "InteractivePrompter", _DummyPrompter)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    ctx = SandboxContext.create(workspace)
    runtime = CLISessionRuntime.create_for_context(ctx)
    llm = _FakeCompactionLLM("<summary>Compacted summary</summary>")
    cli = TGAgentCLI(
        agent=Agent(llm=llm, tools=[], system_prompt="test"),
        context=ctx,
        slash_registry=SlashCommandRegistry(),
        session_runtime=runtime,
    )

    await cli._refresh_context_window_status(trigger="test_setup")

    with cli._console.capture() as capture:
        result = await cli._agent._context.compact_messages(
            [UserMessage(content="Please compact this history")],
            cli._agent.llm,
        )

    console_text = capture.get()
    snapshot = json.loads(runtime.context_window_status_path.read_text(encoding="utf-8"))

    assert "上下文压缩中..." in console_text
    assert result.summary == "Compacted summary"
    assert cli._context_window_status.runtime_state == "idle"
    assert cli._context_window_status.last_compaction_started_at is not None
    assert cli._context_window_status.last_compaction_finished_at is not None
    assert snapshot["runtime_state"] == "idle"
    assert snapshot["last_compaction_started_at"] is not None
    assert snapshot["last_compaction_finished_at"] is not None
    assert snapshot["last_compaction_original_tokens"] == 0
    assert snapshot["last_compaction_new_tokens"] == 0


def test_execution_live_renderable_contains_fixed_status_footer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TG_AGENT_HOME", str(tmp_path / ".tg_agent"))
    monkeypatch.setattr(app_module, "InteractivePrompter", _DummyPrompter)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    ctx = SandboxContext.create(workspace)
    runtime = CLISessionRuntime.create_for_context(ctx)
    cli = TGAgentCLI(
        agent=Agent(llm=EchoLLM(prefix="echo:"), tools=[], system_prompt="test"),
        context=ctx,
        slash_registry=SlashCommandRegistry(),
        session_runtime=runtime,
    )
    cli._context_window_status.model = "echo-worker"
    cli._context_window_status.context_limit = 128_000
    cli._context_window_status.estimated_tokens = 82_000
    cli._context_window_status.context_utilization = 82_000 / 128_000
    cli._context_window_status.status = "正常"
    cli._execution_live_message = "思考中"

    rendered = _render_plain_text(cli._build_execution_live_renderable())

    assert "echo-worker" in rendered
    assert "82.0k/128.0k" in rendered
    assert "64%" in rendered
    assert "思考中" in rendered
    assert "运行状态" in rendered


@pytest.mark.asyncio
async def test_run_agent_renders_execution_state_through_live(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TG_AGENT_HOME", str(tmp_path / ".tg_agent"))
    monkeypatch.setattr(app_module, "InteractivePrompter", _DummyPrompter)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    ctx = SandboxContext.create(workspace)
    runtime = CLISessionRuntime.create_for_context(ctx)
    cli = TGAgentCLI(
        agent=Agent(llm=EchoLLM(prefix="echo:"), tools=[], system_prompt="test"),
        context=ctx,
        slash_registry=SlashCommandRegistry(),
        session_runtime=runtime,
    )

    live_instances: list["_FakeLive"] = []

    class _FakeLive:
        def __init__(self, renderable: Any, **kwargs: Any) -> None:
            self.renderables = [renderable]
            self.kwargs = kwargs
            self.started = False
            self.stopped = False
            live_instances.append(self)

        def start(self) -> None:
            self.started = True

        def update(self, renderable: Any, refresh: bool = False) -> None:
            del refresh
            self.renderables.append(renderable)

        def stop(self) -> None:
            self.stopped = True

    async def _fake_query_stream(user_input: Any, cancel_event=None):
        del user_input, cancel_event
        yield TextEvent(content="hello ")
        yield FinalResponseEvent(content="hello world")

    async def _fake_spinner() -> None:
        return None

    monkeypatch.setattr(app_module, "Live", _FakeLive)
    monkeypatch.setattr(cli, "_run_execution_live_spinner", _fake_spinner)
    monkeypatch.setattr(cli._agent, "query_stream", _fake_query_stream)

    with cli._console.capture() as capture:
        result = await cli._run_agent("hello")

    assert result == "hello world"
    assert len(live_instances) == 1
    assert live_instances[0].started is True
    assert live_instances[0].stopped is True
    assert "hello " in capture.get()

    rendered = "\n".join(_render_plain_text(renderable) for renderable in live_instances[0].renderables)
    assert "echo-worker" in rendered
    assert "运行状态" in rendered


@pytest.mark.asyncio
async def test_run_agent_refreshes_context_window_status_during_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TG_AGENT_HOME", str(tmp_path / ".tg_agent"))
    monkeypatch.setattr(app_module, "InteractivePrompter", _DummyPrompter)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    ctx = SandboxContext.create(workspace)
    runtime = CLISessionRuntime.create_for_context(ctx)
    cli = TGAgentCLI(
        agent=Agent(llm=EchoLLM(prefix="echo:"), tools=[], system_prompt="test"),
        context=ctx,
        slash_registry=SlashCommandRegistry(),
        session_runtime=runtime,
    )

    async def _fake_query_stream(user_input: Any, cancel_event=None):
        del user_input, cancel_event
        yield ToolCallEvent(tool="read_file", args={"path": "a.txt"}, tool_call_id="call-1")
        yield ToolResultEvent(tool="read_file", result="ok", tool_call_id="call-1")
        yield FinalResponseEvent(content="done")

    async def _fake_spinner() -> None:
        return None

    async def _fake_budget_refresh() -> None:
        return None

    refresh_triggers: list[str | None] = []
    original_refresh = cli._refresh_context_window_status

    async def _spy_refresh_context_window_status(*, trigger: str | None = None) -> None:
        refresh_triggers.append(trigger)
        await original_refresh(trigger=trigger)

    class _FakeLive:
        def __init__(self, renderable: Any, **kwargs: Any) -> None:
            del renderable, kwargs

        def start(self) -> None:
            return None

        def update(self, renderable: Any, refresh: bool = False) -> None:
            del renderable, refresh

        def stop(self) -> None:
            return None

    monkeypatch.setattr(app_module, "Live", _FakeLive)
    monkeypatch.setattr(cli, "_run_execution_live_spinner", _fake_spinner)
    monkeypatch.setattr(cli, "_run_execution_live_budget_refresh", _fake_budget_refresh)
    monkeypatch.setattr(cli._agent, "query_stream", _fake_query_stream)
    monkeypatch.setattr(cli, "_refresh_context_window_status", _spy_refresh_context_window_status)

    await cli._run_agent("hello")

    assert "cli_run_start" in refresh_triggers
    assert "cli_tool_call" in refresh_triggers
    assert "cli_tool_result" in refresh_triggers
    assert "cli_final_response" in refresh_triggers
    assert "cli_run_end" in refresh_triggers


@pytest.mark.asyncio
async def test_run_agent_prints_tool_steps_to_console_while_live_handles_preview(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TG_AGENT_HOME", str(tmp_path / ".tg_agent"))
    monkeypatch.setattr(app_module, "InteractivePrompter", _DummyPrompter)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    ctx = SandboxContext.create(workspace)
    runtime = CLISessionRuntime.create_for_context(ctx)
    cli = TGAgentCLI(
        agent=Agent(llm=EchoLLM(prefix="echo:"), tools=[], system_prompt="test"),
        context=ctx,
        slash_registry=SlashCommandRegistry(),
        session_runtime=runtime,
    )

    async def _fake_query_stream(user_input: Any, cancel_event=None):
        del user_input, cancel_event
        yield ToolCallEvent(tool="read_file", args={"path": "a.txt"}, tool_call_id="call-1")
        yield ToolResultEvent(tool="read_file", result="ok", tool_call_id="call-1")
        yield FinalResponseEvent(content="done")

    async def _fake_spinner() -> None:
        return None

    async def _fake_budget_refresh() -> None:
        return None

    class _FakeLive:
        def __init__(self, renderable: Any, **kwargs: Any) -> None:
            del renderable, kwargs

        def start(self) -> None:
            return None

        def update(self, renderable: Any, refresh: bool = False) -> None:
            del renderable, refresh

        def stop(self) -> None:
            return None

    monkeypatch.setattr(app_module, "Live", _FakeLive)
    monkeypatch.setattr(cli, "_run_execution_live_spinner", _fake_spinner)
    monkeypatch.setattr(cli, "_run_execution_live_budget_refresh", _fake_budget_refresh)
    monkeypatch.setattr(cli._agent, "query_stream", _fake_query_stream)

    with cli._console.capture() as capture:
        await cli._run_agent("hello")

    console_text = capture.get()
    assert "read_file" in console_text
    assert "a.txt" in console_text
    assert "ok" in console_text
