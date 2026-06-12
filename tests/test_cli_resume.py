from __future__ import annotations

import io
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from rich.console import Console

import cli.app as app_module
from agent_core import Agent
from agent_core.agent.registry import AgentRegistry, default_agent_sources
from agent_core.llm.messages import (
    AssistantMessage,
    BaseMessage,
    Function,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from agent_core.llm.views import ChatInvokeCompletion, ChatInvokeCompletionChunk
from agent_core.memory.store import MemoryStore
from cli.app import TGAgentCLI
from cli.at_commands import AtCommandRegistry
from cli.resume_handler import ResumeSlashHandler
from cli.session_store import CLISessionStore, workspace_identity
from cli.slash_commands import SlashCommandRegistry
from tools import SandboxContext
from tools.todos import get_todo_store


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


class FakeGatewayLLM(FakeLLM):
    session_id: str | None = None
    session_no: str | None = None
    worker_no: str | None = None
    user_id: str | None = None
    session_callback = None

    @property
    def provider(self) -> str:
        return "gateway"


def _make_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    workspace: Path | None = None,
    llm=None,
) -> TGAgentCLI:
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(app_module, "InteractivePrompter", _DummyPrompter)
    agent = Agent(llm=llm or FakeLLM(), tools=[], system_prompt="system prompt")
    context = SandboxContext.create(workspace or (tmp_path / "workspace"))
    cli = TGAgentCLI(
        agent=agent,
        context=context,
        slash_registry=SlashCommandRegistry(),
    )
    cli._console = Console(file=io.StringIO(), force_terminal=False, color_system=None, width=160)
    return cli


def _write_skill(
    skills_root: Path,
    directory_name: str,
    *,
    name: str,
    description: str,
) -> Path:
    skill_dir = skills_root / directory_name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_path = skill_dir / "SKILL.md"
    skill_path.write_text(
        "\n".join(
            [
                "---",
                f"name: {name}",
                f"description: {description}",
                "---",
                "",
                "# Test Skill",
            ]
        ),
        encoding="utf-8",
    )
    return skill_path


def _write_agent_config(agents_root: Path, *, name: str, description: str) -> Path:
    agents_root.mkdir(parents=True, exist_ok=True)
    agent_path = agents_root / f"{name}.md"
    agent_path.write_text(
        "\n".join(
            [
                "---",
                f"name: {name}",
                f"description: {description}",
                "---",
                "",
                "Agent instructions.",
            ]
        ),
        encoding="utf-8",
    )
    return agent_path


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


@pytest.mark.asyncio
async def test_new_refreshes_prompt_snapshot_without_replacing_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(app_module, "InteractivePrompter", _DummyPrompter)
    monkeypatch.setattr(app_module.os, "system", lambda _command: 0)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    skills_root = workspace / "skills"
    agents_root = workspace / ".tg_agent" / "agents"
    builtin_agents_root = tmp_path / "builtin-agents"
    builtin_agents_root.mkdir()
    _write_skill(skills_root, "demo", name="demo", description="old skill")
    _write_agent_config(agents_root, name="worker", description="old agent")

    memory_store = MemoryStore(base_dir=home / ".tg_agent" / "memories")
    memory_store.add("user", "old user memory")
    skill_registry = AtCommandRegistry(skill_dirs=[skills_root])
    agent_registry = AgentRegistry(
        agent_sources=default_agent_sources(
            workspace,
            builtin_agents_dir=builtin_agents_root,
        )
    )

    def build_prompt() -> str:
        memory_context = memory_store.render_context(memory_store.load_from_disk())
        skills = ", ".join(
            f"{skill.name}:{skill.description}"
            for skill in sorted(skill_registry.get_all(), key=lambda item: item.name)
        )
        agents = ", ".join(
            f"{name}:{agent_registry.get_config(name).description}"
            for name in sorted(agent_registry.list_agents())
            if agent_registry.get_config(name) is not None
        )
        return f"prompt\n{memory_context}\nskills={skills}\nagents={agents}"

    agent = Agent(llm=FakeLLM(), tools=[], system_prompt=build_prompt())
    context = SandboxContext.create(workspace)
    cli = TGAgentCLI(
        agent=agent,
        context=context,
        slash_registry=SlashCommandRegistry(),
        at_registry=skill_registry,
        agent_registry=agent_registry,
        system_prompt_builder=build_prompt,
    )
    cli._console = Console(file=io.StringIO(), force_terminal=False, color_system=None, width=160)

    async def noop_refresh_budget() -> None:
        return None

    monkeypatch.setattr(cli, "_print_welcome", lambda: None)
    monkeypatch.setattr(cli, "_refresh_empty_context_budget_display", noop_refresh_budget)

    original_agent_id = id(cli._agent)
    memory_store.replace("user", "old user memory", "new user memory")
    _write_skill(skills_root, "demo", name="demo", description="new skill")
    _write_agent_config(agents_root, name="worker", description="new agent")

    await cli._handle_new_command()

    assert id(cli._agent) == original_agent_id
    assert "new user memory" in cli._agent.system_prompt
    assert "old user memory" not in cli._agent.system_prompt
    assert "demo:new skill" in cli._agent.system_prompt
    assert "worker:new agent" in cli._agent.system_prompt
    assert cli._agent.messages == []


@pytest.mark.asyncio
async def test_reset_keeps_existing_prompt_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _make_cli(tmp_path, monkeypatch)

    async def noop_refresh_budget() -> None:
        return None

    monkeypatch.setattr(cli, "_refresh_empty_context_budget_display", noop_refresh_budget)
    cli._agent.system_prompt = "existing prompt"

    await cli._handle_reset_command()

    assert cli._agent.system_prompt == "existing prompt"


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


def test_session_store_list_sessions_supports_offset_pagination(tmp_path: Path) -> None:
    store = CLISessionStore(tmp_path / "home" / ".tg_agent" / "sessions.db")
    workspace = tmp_path / "workspace"
    for index in range(5):
        _seed_session(
            store,
            session_id=f"session-{index}",
            workspace=workspace,
            messages=[UserMessage(content=f"user {index}")],
            now=1000.0 + index,
        )

    _root, workspace_key = workspace_identity(workspace)
    sessions = store.list_sessions(workspace_key=workspace_key, limit=2, offset=2)

    assert [session.id for session in sessions] == ["session-2", "session-1"]


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


def test_session_store_renames_session_and_references(tmp_path: Path) -> None:
    db_path = tmp_path / "sessions.db"
    workspace = tmp_path / "workspace"
    store = CLISessionStore(db_path)
    messages = [
        UserMessage(content="old user"),
        AssistantMessage(content="old assistant"),
    ]
    _seed_session(store, session_id="local_old", workspace=workspace, messages=messages)

    renamed = store.rename_session("local_old", "server-session-1")

    assert renamed is True
    assert store.get_session("local_old") is None
    assert store.get_session("server-session-1") is not None
    assert store.count_messages("server-session-1") == 2
    snapshot = store.load_context_snapshot("server-session-1")
    assert snapshot is not None
    assert snapshot.session_id == "server-session-1"
    assert [message.content for message in snapshot.messages] == ["old user", "old assistant"]


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
async def test_resume_hydrates_todo_store_from_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    cli = _make_cli(tmp_path, monkeypatch, workspace=workspace)
    assert cli._session_store is not None
    get_todo_store(cli._ctx.session_id).write(
        [{"id": "stale", "content": "Stale", "status": "pending"}],
        merge=False,
    )
    snapshot: list[BaseMessage] = [
        AssistantMessage(
            content=None,
            tool_calls=[
                ToolCall(
                    id="call-todo",
                    function=Function(name="todo", arguments="{}"),
                )
            ],
        ),
        ToolMessage(
            tool_call_id="call-todo",
            tool_name="todo",
            content=(
                '{"todos":[{"id":"1","content":"Resume task","status":"in_progress"}],'
                '"summary":{},"warnings":[]}'
            ),
        ),
    ]
    _seed_session(
        cli._session_store,
        session_id="todo-session",
        workspace=workspace,
        messages=snapshot,
        snapshot=snapshot,
    )

    switched = await cli._switch_resume_session("todo-session")

    assert switched is True
    assert get_todo_store(cli._ctx.session_id).read() == [
        {"id": "1", "content": "Resume task", "status": "in_progress"}
    ]


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


def test_resume_picker_paginates_sessions_and_selects_current_page(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _make_cli(tmp_path, monkeypatch)
    assert cli._session_store is not None
    output_buffer = io.StringIO()
    console = Console(file=output_buffer, force_terminal=False, color_system=None, width=160)
    handler = ResumeSlashHandler(
        store=cli._session_store,
        console=console,
        workspace_dir=cli._ctx.working_dir,
    )
    for index in range(25):
        _seed_session(
            cli._session_store,
            session_id=f"session-{index:02d}",
            workspace=cli._ctx.working_dir,
            messages=[UserMessage(content=f"session-{index:02d} user")],
            now=1000.0 + index,
        )

    handler.start_pick_mode(current_session_id=cli._conversation_session_id)
    first_page_output = output_buffer.getvalue()

    assert handler.pick_active is True
    assert "第 1 页" in first_page_output
    assert "  1. session-24 user" in first_page_output
    assert "  20. session-05 user" in first_page_output
    assert "session-24 user" in first_page_output
    assert "session-05 user" in first_page_output
    assert "session-04 user" not in first_page_output

    next_result = handler.handle_pick_input("n")
    second_page_output = output_buffer.getvalue()

    assert next_result.handled is True
    assert next_result.selected_session_id is None
    assert "第 2 页" in second_page_output
    assert "  21. session-04 user" in second_page_output
    assert "  25. session-00 user" in second_page_output
    assert "session-04 user" in second_page_output
    assert "session-00 user" in second_page_output

    last_page_result = handler.handle_pick_input("n")
    assert last_page_result.handled is True
    assert "已经是最后一页" in output_buffer.getvalue()

    previous_page_number = handler.handle_pick_input("1")
    assert previous_page_number.handled is True
    assert previous_page_number.selected_session_id is None
    assert "选择超出范围，请输入 21-25" in output_buffer.getvalue()

    selected = handler.handle_pick_input("21")
    assert selected.handled is True
    assert selected.selected_session_id == "session-04"


def test_resume_picker_previous_page_navigation_and_first_page_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _make_cli(tmp_path, monkeypatch)
    assert cli._session_store is not None
    output_buffer = io.StringIO()
    console = Console(file=output_buffer, force_terminal=False, color_system=None, width=160)
    handler = ResumeSlashHandler(
        store=cli._session_store,
        console=console,
        workspace_dir=cli._ctx.working_dir,
    )
    for index in range(21):
        _seed_session(
            cli._session_store,
            session_id=f"session-{index:02d}",
            workspace=cli._ctx.working_dir,
            messages=[UserMessage(content=f"session-{index:02d} user")],
            now=1000.0 + index,
        )

    handler.start_pick_mode(current_session_id=cli._conversation_session_id)
    handler.handle_pick_input("n")
    previous_result = handler.handle_pick_input("p")
    first_page_output = output_buffer.getvalue()

    assert previous_result.handled is True
    assert previous_result.selected_session_id is None
    assert "第 1 页" in first_page_output
    assert "session-20 user" in first_page_output

    first_page_result = handler.handle_pick_input("p")
    assert first_page_result.handled is True
    assert "已经是第一页" in output_buffer.getvalue()


@pytest.mark.asyncio
async def test_new_command_does_not_create_empty_session_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    cli = _make_cli(tmp_path, monkeypatch, workspace=workspace)
    assert cli._session_store is not None
    old_conversation_id = cli._conversation_session_id
    sandbox_session_id = cli._ctx.session_id
    get_todo_store(sandbox_session_id).write(
        [{"id": "1", "content": "Will be cleared", "status": "pending"}],
        merge=False,
    )
    monkeypatch.setattr(app_module.os, "system", lambda command: 0)

    handled = await cli._handle_slash_command("/new")

    _root, workspace_key = workspace_identity(workspace)
    assert handled is True
    assert cli._conversation_session_id != old_conversation_id
    assert str(cli._conversation_session_id).startswith("local_")
    assert cli._conversation_session_created is False
    assert cli._ctx.session_id == sandbox_session_id
    assert cli._agent.messages == []
    assert get_todo_store(sandbox_session_id).read() == []
    assert cli._session_store.list_sessions(workspace_key=workspace_key) == []


@pytest.mark.asyncio
async def test_new_command_clears_session_until_gateway_session_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    cli = _make_cli(tmp_path, monkeypatch, workspace=workspace, llm=FakeGatewayLLM())
    cli._im_worker_id = "user@example.com"
    monkeypatch.setattr(app_module.os, "system", lambda command: 0)

    await cli._handle_slash_command("/new")

    assert cli._conversation_session_id is None
    assert cli._conversation_session_created is False

    cli._handle_gateway_session_event("server-session-1", is_new=True)

    assert cli._conversation_session_id == "server-session-1"


def test_gateway_cli_renames_persisted_local_session_on_session_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    cli = _make_cli(tmp_path, monkeypatch, workspace=workspace, llm=FakeGatewayLLM())
    assert cli._session_store is not None
    cli._conversation_session_id = "local_existing"
    cli._conversation_session_created = False
    cli._agent._context.add_message(UserMessage(content="local user"))
    cli._agent._context.add_message(AssistantMessage(content="local assistant"))
    cli._persist_current_session_state()

    cli._handle_gateway_session_event("server-session-1", is_new=True)

    assert cli._conversation_session_id == "server-session-1"
    assert cli._session_store.get_session("local_existing") is None
    assert cli._session_store.get_session("server-session-1") is not None
    assert cli._session_store.count_messages("server-session-1") == 2


@pytest.mark.asyncio
async def test_new_command_keeps_previous_persisted_session_resumable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    cli = _make_cli(tmp_path, monkeypatch, workspace=workspace)
    assert cli._session_store is not None
    old_conversation_id = cli._conversation_session_id
    cli._agent._context.add_message(UserMessage(content="old user"))
    cli._agent._context.add_message(AssistantMessage(content="old assistant"))
    monkeypatch.setattr(app_module.os, "system", lambda command: 0)

    await cli._handle_slash_command("/new")

    _root, workspace_key = workspace_identity(workspace)
    sessions = cli._session_store.list_sessions(
        workspace_key=workspace_key,
        exclude_session_id=cli._conversation_session_id,
    )
    assert [session.id for session in sessions] == [old_conversation_id]
    assert str(cli._conversation_session_id).startswith("local_")
    assert cli._agent.messages == []


@pytest.mark.asyncio
async def test_new_command_followup_messages_write_to_new_session_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    cli = _make_cli(tmp_path, monkeypatch, workspace=workspace)
    assert cli._session_store is not None
    old_conversation_id = cli._conversation_session_id
    cli._agent._context.add_message(UserMessage(content="old user"))
    cli._agent._context.add_message(AssistantMessage(content="old assistant"))
    cli._persist_current_session_state()
    monkeypatch.setattr(app_module.os, "system", lambda command: 0)

    await cli._handle_slash_command("/new")
    new_conversation_id = cli._conversation_session_id
    cli._agent._context.add_message(UserMessage(content="new user"))
    cli._agent._context.add_message(AssistantMessage(content="new assistant"))
    cli._persist_current_session_state()

    assert new_conversation_id != old_conversation_id
    assert str(new_conversation_id).startswith("local_")
    assert cli._session_store.count_messages(old_conversation_id) == 2
    assert cli._session_store.count_messages(new_conversation_id) == 2
    old_rounds = cli._session_store.recent_user_assistant_rounds(
        old_conversation_id,
        limit=10,
    )
    new_rounds = cli._session_store.recent_user_assistant_rounds(
        new_conversation_id,
        limit=10,
    )
    assert [round_item.user for round_item in old_rounds] == ["old user"]
    assert [round_item.user for round_item in new_rounds] == ["new user"]
