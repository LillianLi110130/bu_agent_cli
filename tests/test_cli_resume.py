from __future__ import annotations

import io
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from rich.console import Console

import cli.app as app_module
from agent_core import Agent
from agent_core.llm.messages import (
    AssistantMessage,
    BaseMessage,
    Function,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from agent_core.llm.views import ChatInvokeCompletion, ChatInvokeCompletionChunk
from cli.app import TGAgentCLI
from cli.resume_handler import ResumeSlashHandler
from cli.session_store import CLISessionStore, workspace_identity
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


def _make_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    workspace: Path | None = None,
) -> TGAgentCLI:
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(app_module, "InteractivePrompter", _DummyPrompter)
    agent = Agent(llm=FakeLLM(), tools=[], system_prompt="system prompt")
    context = SandboxContext.create(workspace or (tmp_path / "workspace"))
    cli = TGAgentCLI(
        agent=agent,
        context=context,
        slash_registry=SlashCommandRegistry(),
    )
    cli._console = Console(file=io.StringIO(), force_terminal=False, color_system=None, width=160)
    return cli


def _seed_session(
    store: CLISessionStore,
    *,
    session_id: str,
    workspace: Path,
    messages: list[BaseMessage],
    snapshot: list[BaseMessage] | None = None,
    compacted: bool = False,
    now: float = 1000.0,
) -> None:
    workspace_root, workspace_key = workspace_identity(workspace)
    store.create_session(
        session_id=session_id,
        workspace_root=workspace_root,
        workspace_key=workspace_key,
        model="fake-model",
        system_prompt="stored system",
        now=now,
    )
    store.append_messages(session_id, messages)
    store.upsert_context_snapshot(
        session_id=session_id,
        messages=snapshot if snapshot is not None else messages,
        compacted=compacted,
        now=now,
    )


def test_cli_session_db_is_created_under_user_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    cli = _make_cli(tmp_path, monkeypatch, workspace=workspace)

    assert cli._session_store is not None
    assert cli._session_store.db_path == tmp_path / "home" / ".tg_agent" / "sessions.db"
    assert cli._session_store.db_path.exists()
    assert not (workspace / ".tg_agent" / "sessions.db").exists()


def test_cli_startup_does_not_create_empty_session_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    cli = _make_cli(tmp_path, monkeypatch, workspace=workspace)

    assert cli._session_store is not None
    _root, workspace_key = workspace_identity(workspace)
    assert cli._session_store.list_sessions(workspace_key=workspace_key) == []
    assert cli._conversation_session_created is False


def test_session_store_filters_workspace_and_excludes_current(tmp_path: Path) -> None:
    store = CLISessionStore(tmp_path / "home" / ".tg_agent" / "sessions.db")
    workspace_a = tmp_path / "workspace-a"
    workspace_b = tmp_path / "workspace-b"

    _seed_session(store, session_id="current", workspace=workspace_a, messages=[])
    _seed_session(store, session_id="old-a", workspace=workspace_a, messages=[])
    _seed_session(store, session_id="old-b", workspace=workspace_b, messages=[])

    _root, workspace_key = workspace_identity(workspace_a)
    sessions = store.list_sessions(workspace_key=workspace_key, exclude_session_id="current")

    assert [session.id for session in sessions] == ["old-a"]


def test_session_store_round_trips_tool_aware_messages(tmp_path: Path) -> None:
    store = CLISessionStore(tmp_path / "home" / ".tg_agent" / "sessions.db")
    workspace = tmp_path / "workspace"
    tool_call = ToolCall(
        id="call-1",
        function=Function(name="read", arguments='{"file_path":"README.md"}'),
    )
    messages: list[BaseMessage] = [
        UserMessage(content="inspect"),
        AssistantMessage(content=None, tool_calls=[tool_call]),
        ToolMessage(tool_call_id="call-1", tool_name="read", content="file content"),
        AssistantMessage(content="done"),
    ]

    _seed_session(store, session_id="tool-session", workspace=workspace, messages=messages)

    snapshot = store.load_context_snapshot("tool-session")
    assert snapshot is not None
    loaded_assistant = snapshot.messages[1]
    loaded_tool = snapshot.messages[2]
    assert isinstance(loaded_assistant, AssistantMessage)
    assert loaded_assistant.tool_calls is not None
    assert loaded_assistant.tool_calls[0].function.name == "read"
    assert isinstance(loaded_tool, ToolMessage)
    assert loaded_tool.tool_call_id == "call-1"
    assert loaded_tool.tool_name == "read"


@pytest.mark.asyncio
async def test_resume_uncompressed_session_loads_snapshot_and_prints_recent_10_rounds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    cli = _make_cli(tmp_path, monkeypatch, workspace=workspace)
    assert cli._session_store is not None
    transcript: list[BaseMessage] = []
    for index in range(12):
        transcript.append(UserMessage(content=f"turn-{index:02d}-user"))
        transcript.append(AssistantMessage(content=f"turn-{index:02d}-assistant"))
    snapshot = [
        UserMessage(content="compressed user"),
        AssistantMessage(content="compressed assistant"),
    ]
    _seed_session(
        cli._session_store,
        session_id="old-session",
        workspace=workspace,
        messages=transcript,
        snapshot=snapshot,
        compacted=False,
    )

    output_buffer = io.StringIO()
    cli._console = Console(file=output_buffer, force_terminal=False, color_system=None, width=160)
    switched = await cli._switch_resume_session("old-session")

    output = output_buffer.getvalue()
    assert switched is True
    assert cli._conversation_session_id == "old-session"
    assert [message.content for message in cli._agent.messages] == [
        "compressed user",
        "compressed assistant",
    ]
    assert "最近 10 轮对话" in output
    assert "turn-01-user" not in output
    assert "turn-01-assistant" not in output
    assert "turn-02-user" in output
    assert "turn-11-assistant" in output


@pytest.mark.asyncio
async def test_resume_compacted_session_only_prints_compaction_hint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    cli = _make_cli(tmp_path, monkeypatch, workspace=workspace)
    assert cli._session_store is not None
    messages: list[BaseMessage] = [
        UserMessage(content="hidden user"),
        AssistantMessage(content="hidden assistant"),
    ]
    _seed_session(
        cli._session_store,
        session_id="compacted-session",
        workspace=workspace,
        messages=messages,
        snapshot=[UserMessage(content="summary context")],
        compacted=True,
    )

    output_buffer = io.StringIO()
    cli._console = Console(file=output_buffer, force_terminal=False, color_system=None, width=160)
    switched = await cli._switch_resume_session("compacted-session")

    output = output_buffer.getvalue()
    assert switched is True
    assert "该会话曾触发上下文压缩" in output
    assert "最近 10 轮对话" not in output
    assert "hidden assistant" not in output
    assert [message.content for message in cli._agent.messages] == ["summary context"]


@pytest.mark.asyncio
async def test_resume_then_persist_does_not_duplicate_loaded_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    cli = _make_cli(tmp_path, monkeypatch, workspace=workspace)
    assert cli._session_store is not None
    original_messages: list[BaseMessage] = [
        UserMessage(content="old user"),
        AssistantMessage(content="old assistant"),
    ]
    _seed_session(
        cli._session_store,
        session_id="old-session",
        workspace=workspace,
        messages=original_messages,
    )

    await cli._switch_resume_session("old-session")
    cli._agent._context.add_message(UserMessage(content="new user"))
    cli._agent._context.add_message(AssistantMessage(content="new assistant"))
    cli._persist_current_session_state()

    assert cli._session_store.count_messages("old-session") == 4
    rounds = cli._session_store.recent_user_assistant_rounds("old-session", limit=10)
    assert [round_item.user for round_item in rounds] == ["old user", "new user"]


@pytest.mark.asyncio
async def test_resume_picker_cancel_inputs_match_model_picker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _make_cli(tmp_path, monkeypatch)

    for cancel_input in ("q", "quit", "cancel", "exit"):
        cli._resume_handler = ResumeSlashHandler(
            store=cli._session_store,
            console=cli._console,
            workspace_dir=cli._ctx.working_dir,
        )
        assert cli._session_store is not None
        _seed_session(
            cli._session_store,
            session_id=f"session-{cancel_input}",
            workspace=cli._ctx.working_dir,
            messages=[UserMessage(content=f"user {cancel_input}")],
        )
        cli._resume_handler.start_pick_mode(current_session_id=cli._conversation_session_id)

        result = cli._resume_handler.handle_pick_input(cancel_input)

        assert result.handled is True
        assert result.selected_session_id is None
        assert cli._resume_handler.pick_active is False
