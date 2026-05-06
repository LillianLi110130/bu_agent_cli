from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from agent_core.agent import Agent
from agent_core.agent.registry import AgentRegistry
from agent_core.llm.views import ChatInvokeCompletion
from agent_core.team import Mailbox, TaskBoard, TeamMessage, TeamRuntime
from agent_core.team.experiment import TEAM_EXPERIMENT_ENV
from agent_core.team.runtime import TEAM_WORKER_INTERNAL_FLAG, build_team_worker_command
from cli.team.factory import create_team_member_agent
from cli.team.handler import TeamSlashHandler
from cli.team.inbox import TeamInboxAttachmentHook, TeamInboxBuffer
from cli.team.worker import TeamWorker
from rich.console import Console
import tg_crab_main
from tools import SandboxContext
from tools.team_tool import team_create, team_update_task


class _FakeProcess:
    pid = 4242


def _fake_popen(*args, **kwargs):
    _fake_popen.calls.append((args, kwargs))
    return _FakeProcess()


_fake_popen.calls = []


class _FakeLLM:
    model = "fake"

    @property
    def provider(self) -> str:
        return "fake"

    @property
    def name(self) -> str:
        return "fake"

    async def ainvoke(self, *args, **kwargs):
        return ChatInvokeCompletion(content="ok")


class _CapturingLLM(_FakeLLM):
    def __init__(self) -> None:
        self.calls = []

    async def ainvoke(self, *args, **kwargs):
        self.calls.append(kwargs.get("messages") or [])
        return ChatInvokeCompletion(content="ok")


def test_team_runtime_create_spawn_task_and_mailbox(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    teams_root = tmp_path / "teams"
    workspace.mkdir()
    runtime = TeamRuntime(
        teams_root=teams_root,
        workspace_root=workspace,
        popen_factory=_fake_popen,
    )

    team = runtime.create_team(name="demo", goal="Analyze the repo")
    member = runtime.spawn_member(
        team_id=team.team_id,
        member_id="explorer-1",
        agent_type="explore",
    )
    task = runtime.create_task(
        team_id=team.team_id,
        title="Inspect subagent runtime",
        description="Read the task manager and summarize it",
        assigned_to="explorer-1",
    )

    assert team.team_id.startswith("demo-")
    assert member.pid == 4242
    assert task.assigned_to == "explorer-1"
    assert (teams_root / team.team_id / "config.json").exists()
    assert (teams_root / team.team_id / "mailboxes" / "explorer-1" / "inbox").exists()
    assert _fake_popen.calls
    command = _fake_popen.calls[-1][0][0]
    assert TEAM_WORKER_INTERNAL_FLAG in command
    assert "tg_crab_main.py" in command[1]

    inbox_messages = Mailbox(teams_root / team.team_id).receive("explorer-1")
    assert [message.type for message in inbox_messages] == ["task_assigned"]
    assert inbox_messages[0].metadata["task_id"] == task.task_id

    runtime.send_message(team_id=team.team_id, recipient="lead", body="done", type="task_completed")
    lead_messages = runtime.read_lead_inbox(team.team_id)
    assert len(lead_messages) == 1
    assert lead_messages[0].body == "done"


def test_task_board_claim_complete_and_file_lock(tmp_path: Path) -> None:
    team_dir = tmp_path / "team"
    team_dir.mkdir()
    board = TaskBoard(team_dir)
    task = board.create_task(
        title="Edit file",
        description="Update one file",
        assigned_to="worker-1",
        write_scope=["src/app.py"],
    )

    claimed = board.claim_next("worker-1")

    assert claimed is not None
    assert claimed.task_id == task.task_id
    assert claimed.status == "in_progress"
    assert board.claim_next("worker-2") is None

    completed = board.complete_task(task.task_id, "worker-1", "finished")

    assert completed.status == "completed"
    assert completed.result == "finished"
    assert not any((team_dir / "locks" / "files").glob("*.lock"))


def test_team_runtime_send_message_accepts_any_sender(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    teams_root = tmp_path / "teams"
    workspace.mkdir()
    runtime = TeamRuntime(teams_root=teams_root, workspace_root=workspace)
    team = runtime.create_team(name="demo", goal="Coordinate")

    message = runtime.send_message(
        team_id=team.team_id,
        sender="explorer-1",
        recipient="coder-1",
        body="Please inspect the failing test.",
        type="handoff",
    )

    assert message.sender == "explorer-1"
    assert message.recipient == "coder-1"
    inbox_messages = Mailbox(teams_root / team.team_id).receive("coder-1")
    assert [item.message_id for item in inbox_messages] == [message.message_id]
    assert inbox_messages[0].type == "handoff"


def test_team_slash_handler_create_and_list(tmp_path: Path) -> None:
    runtime = TeamRuntime(
        teams_root=tmp_path / "teams",
        workspace_root=tmp_path,
        popen_factory=_fake_popen,
    )
    console = Console(record=True, width=100)
    handler = TeamSlashHandler(runtime=runtime, console=console)

    asyncio.run(handler.handle(["create", "demo", "Analyze", "repo"]))
    asyncio.run(handler.handle(["list"]))

    output = console.export_text()
    assert "已创建 team" in output
    assert "demo-" in output
    assert "Analyze repo" in output


def test_tg_crab_main_create_agent_initializes_team_runtime(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class _FakeLLM:
        model = "fake"

        @property
        def provider(self) -> str:
            return "fake"

        @property
        def name(self) -> str:
            return "fake"

    class _FakeRegistries:
        slash_registry = None
        skill_registry = type(
            "SkillRegistry",
            (),
            {"get_all": lambda self: []},
        )()
        agent_registry = type(
            "AgentRegistry",
            (),
            {
                "list_callable_agents": lambda self: [],
                "get_config": lambda self, _name: None,
            },
        )()
        plugin_manager = None

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv(TEAM_EXPERIMENT_ENV, "1")
    monkeypatch.setattr(tg_crab_main, "create_llm", lambda _model: _FakeLLM())
    monkeypatch.setattr(tg_crab_main, "build_agent_hooks", lambda: [])

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    _agent, ctx, _runtime = tg_crab_main.create_agent(
        model=None,
        root_dir=workspace,
        runtime_registries=_FakeRegistries(),
    )

    assert isinstance(ctx.team_runtime, TeamRuntime)


def test_tg_crab_main_create_agent_leaves_team_runtime_disabled_by_default(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class _FakeLLM:
        model = "fake"

        @property
        def provider(self) -> str:
            return "fake"

        @property
        def name(self) -> str:
            return "fake"

    class _FakeRegistries:
        slash_registry = None
        skill_registry = type(
            "SkillRegistry",
            (),
            {"get_all": lambda self: []},
        )()
        agent_registry = type(
            "AgentRegistry",
            (),
            {
                "list_callable_agents": lambda self: [],
                "get_config": lambda self, _name: None,
            },
        )()
        plugin_manager = None

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv(TEAM_EXPERIMENT_ENV, raising=False)
    monkeypatch.setattr(tg_crab_main, "create_llm", lambda _model: _FakeLLM())
    monkeypatch.setattr(tg_crab_main, "build_agent_hooks", lambda: [])

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    _agent, ctx, _runtime = tg_crab_main.create_agent(
        model=None,
        root_dir=workspace,
        runtime_registries=_FakeRegistries(),
    )

    assert ctx.team_runtime is None


def test_build_team_worker_command_uses_unified_main_entrypoint(tmp_path: Path) -> None:
    command = build_team_worker_command(
        teams_root=tmp_path / "teams",
        team_id="team-1",
        member_id="explorer-1",
        agent_type="explore",
        workspace_root=tmp_path / "workspace",
    )

    assert command[0] == tg_crab_main.sys.executable
    assert command[2] == TEAM_WORKER_INTERNAL_FLAG
    assert "--team-id" in command
    assert "team-1" in command


def test_tg_crab_main_team_worker_internal_flag_helpers() -> None:
    argv = [TEAM_WORKER_INTERNAL_FLAG, "--team-id", "team-1"]

    assert tg_crab_main._should_run_internal_team_worker(argv) is True
    assert tg_crab_main._strip_internal_team_worker_flag(argv) == ["--team-id", "team-1"]


def test_team_worker_member_session_runtime_lives_under_team_dir(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    teams_root = tmp_path / "teams"
    workspace.mkdir()
    store_runtime = TeamRuntime(teams_root=teams_root, workspace_root=workspace)
    team = store_runtime.create_team(name="demo", goal="Analyze")
    ctx = SandboxContext.create(workspace)
    worker = TeamWorker(
        teams_root=teams_root,
        team_id=team.team_id,
        member_id="explorer-1",
        agent_type="explore",
        workspace=workspace,
    )

    runtime = worker._create_member_session_runtime(ctx)  # noqa: SLF001

    expected_root = teams_root / team.team_id / "sessions" / "explorer-1"
    assert runtime.rollout_dir == expected_root.resolve()
    assert runtime.meta_path == expected_root.resolve() / "meta.json"
    assert runtime.checkpoints_dir == expected_root.resolve() / "checkpoints"
    assert runtime.artifacts_dir == expected_root.resolve() / "artifacts"
    assert runtime.meta_path.exists()


def test_team_slash_handler_reports_disabled_experiment() -> None:
    console = Console(record=True, width=100)
    handler = TeamSlashHandler(runtime=None, console=console)

    asyncio.run(handler.handle(["create", "demo"]))

    output = console.export_text()
    assert TEAM_EXPERIMENT_ENV in output


def test_team_tool_reports_disabled_experiment() -> None:
    result = asyncio.run(
        team_create.func(
            ctx=SimpleNamespace(team_runtime=None),
            name="demo",
            goal="Analyze",
        )
    )

    assert result.startswith("Error:")
    assert TEAM_EXPERIMENT_ENV in result


def test_team_update_task_tool_updates_existing_task(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    teams_root = tmp_path / "teams"
    workspace.mkdir()
    runtime = TeamRuntime(teams_root=teams_root, workspace_root=workspace)
    team = runtime.create_team(name="demo", goal="Coordinate")
    task = runtime.create_task(
        team_id=team.team_id,
        title="Old title",
        description="Old description",
        assigned_to="explorer-1",
        write_scope=["old.py"],
    )

    result = asyncio.run(
        team_update_task.func(
            ctx=SimpleNamespace(team_runtime=runtime),
            team_id=team.team_id,
            task_id=task.task_id,
            title="New title",
            description="New description",
            status="blocked",
            assigned_to="coder-1",
            depends_on=["task_dep"],
            write_scope=["new.py"],
            error="Waiting for dependency",
        )
    )

    payload = json.loads(result)
    assert payload["title"] == "New title"
    assert payload["description"] == "New description"
    assert payload["status"] == "blocked"
    assert payload["assigned_to"] == "coder-1"
    assert payload["depends_on"] == ["task_dep"]
    assert payload["write_scope"] == ["new.py"]
    assert payload["error"] == "Waiting for dependency"
    messages = Mailbox(teams_root / team.team_id).receive("coder-1")
    assert [message.type for message in messages] == ["task_updated"]


def test_team_update_task_tool_rejects_invalid_status(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    teams_root = tmp_path / "teams"
    workspace.mkdir()
    runtime = TeamRuntime(teams_root=teams_root, workspace_root=workspace)
    team = runtime.create_team(name="demo", goal="Coordinate")
    task = runtime.create_task(
        team_id=team.team_id,
        title="Task",
        description="Description",
    )

    result = asyncio.run(
        team_update_task.func(
            ctx=SimpleNamespace(team_runtime=runtime),
            team_id=team.team_id,
            task_id=task.task_id,
            status="cancelled",
        )
    )

    assert result.startswith("Error:")
    assert "Invalid task status" in result


def test_team_agent_factory_creates_team_member_agent_directly(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "workspace"
    agents_dir = tmp_path / "agents"
    workspace.mkdir()
    agents_dir.mkdir()
    (agents_dir / "explore.md").write_text(
        """---
name: explore
description: explore
model: inherit
---
Explore.
""",
        encoding="utf-8",
    )
    registry = AgentRegistry(agent_sources=[("test", agents_dir, 1)])
    runtime = SimpleNamespace(
        slash_registry=None,
        skill_registry=type(
            "SkillRegistry",
            (),
            {
                "get_all": lambda self: [],
                "get": lambda self, _name: None,
            },
        )(),
        agent_registry=registry,
        plugin_manager=None,
    )
    monkeypatch.setattr(tg_crab_main, "create_llm", lambda _model: _FakeLLM())
    monkeypatch.setattr(tg_crab_main, "build_agent_hooks", lambda: [])

    agent, ctx, returned_runtime = create_team_member_agent(
        model=None,
        root_dir=workspace,
        team_prompt="## Agent Team Runtime\n\nTeam-specific instructions.",
        agent_type="explore",
        runtime_registries=runtime,
    )

    assert ctx.working_dir == workspace.resolve()
    assert returned_runtime is runtime
    assert agent.runtime_role == "team_member"
    assert agent.agent_config is registry.get_config("explore")
    assert agent.system_prompt is not None
    assert agent.system_prompt.index("## Agent Team Runtime") < agent.system_prompt.index(
        "## Agent Type Instructions"
    )
    assert "Explore." in agent.system_prompt
    assert all(
        getattr(tool, "name", None) not in {"delegate", "delegate_parallel"}
        for tool in agent.tools
    )


def test_team_worker_uses_main_create_agent_factory(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    teams_root = tmp_path / "teams"
    agents_dir = tmp_path / "agents"
    workspace.mkdir()
    agents_dir.mkdir()
    (agents_dir / "explore.md").write_text(
        """---
name: explore
description: explore
model: inherit
---
Explore.
""",
        encoding="utf-8",
    )
    store_runtime = TeamRuntime(teams_root=teams_root, workspace_root=workspace)
    team = store_runtime.create_team(name="demo", goal="Analyze")
    factory_calls: list[Path] = []

    def fake_create_agent(_model, root_dir, *, agent_type, team_prompt):
        factory_calls.append(Path(root_dir))
        registry = AgentRegistry(agent_sources=[("test", agents_dir, 1)])
        config = registry.get_config(agent_type)
        assert config is not None
        system_prompt = "\n\n".join(
            [
                "base system prompt",
                team_prompt,
                f"## Agent Type Instructions\n\n{config.system_prompt}",
            ]
        )
        agent = Agent(
            llm=_FakeLLM(),
            tools=[],
            system_prompt=system_prompt,
            agent_config=config,
            runtime_role="team_member",
        )
        ctx = SandboxContext.create(root_dir)
        runtime = SimpleNamespace(
            agent_registry=registry,
            skill_registry=None,
        )
        return agent, ctx, runtime

    worker = TeamWorker(
        teams_root=teams_root,
        team_id=team.team_id,
        member_id="explorer-1",
        agent_type="explore",
        workspace=workspace,
        create_agent_factory=fake_create_agent,
    )

    agent, ctx = worker._build_member_agent()  # noqa: SLF001
    agent.bind_session_runtime(worker._create_member_session_runtime(ctx))  # noqa: SLF001

    assert factory_calls == [workspace]
    assert agent.runtime_role == "team_member"
    assert agent.system_prompt is not None
    assert agent.system_prompt.index("base system prompt") < agent.system_prompt.index(
        "## Agent Team Runtime"
    )
    assert agent.system_prompt.index("## Agent Team Runtime") < agent.system_prompt.index(
        "## Agent Type Instructions"
    )
    assert "Explore." in agent.system_prompt
    assert f"Team ID: {team.team_id}" in agent.system_prompt
    assert "Member ID: explorer-1" in agent.system_prompt
    expected_session = teams_root / team.team_id / "sessions" / "explorer-1"
    assert (expected_session / "meta.json").exists()


def test_team_worker_send_message_uses_shared_messenger(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    teams_root = tmp_path / "teams"
    workspace.mkdir()
    store_runtime = TeamRuntime(teams_root=teams_root, workspace_root=workspace)
    team = store_runtime.create_team(name="demo", goal="Coordinate")
    worker = TeamWorker(
        teams_root=teams_root,
        team_id=team.team_id,
        member_id="explorer-1",
        agent_type="explore",
        workspace=workspace,
    )

    worker.send_message(
        recipient="coder-1",
        body="Can you take the implementation task?",
        type="handoff",
    )

    messages = Mailbox(teams_root / team.team_id).receive("coder-1")
    assert len(messages) == 1
    assert messages[0].sender == "explorer-1"
    assert messages[0].recipient == "coder-1"
    assert messages[0].type == "handoff"


def test_team_inbox_attachment_hook_adds_messages_before_llm_call() -> None:
    llm = _CapturingLLM()
    agent = Agent(
        llm=llm,
        tools=[],
        system_prompt="base",
        runtime_role="team_member",
        use_streaming=False,
    )
    buffer = TeamInboxBuffer()
    agent.register_hook(TeamInboxAttachmentHook(buffer=buffer, member_id="explorer-1"))

    asyncio.run(
        buffer.push(
            TeamMessage(
                message_id="msg_1",
                team_id="team-1",
                sender="lead",
                recipient="explorer-1",
                type="note",
                body="Please incorporate this before the next model call.",
            )
        )
    )

    result = asyncio.run(agent.query("Continue the task."))

    assert result == "ok"
    assert llm.calls
    contents = [getattr(message, "content", "") for message in llm.calls[0]]
    assert any("## Team Messages" in str(content) for content in contents)
    assert any("Please incorporate this before the next model call." in str(content) for content in contents)


def test_team_worker_submits_pending_messages_when_idle(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    teams_root = tmp_path / "teams"
    workspace.mkdir()
    store_runtime = TeamRuntime(teams_root=teams_root, workspace_root=workspace)
    team = store_runtime.create_team(name="demo", goal="Coordinate")
    worker = TeamWorker(
        teams_root=teams_root,
        team_id=team.team_id,
        member_id="explorer-1",
        agent_type="explore",
        workspace=workspace,
    )
    store_runtime.send_message(
        team_id=team.team_id,
        sender="lead",
        recipient="explorer-1",
        body="Summarize the latest handoff.",
        type="note",
    )

    class _IdleAgent:
        def __init__(self) -> None:
            self.queries: list[str] = []

        async def query(self, prompt: str) -> str:
            self.queries.append(prompt)
            return "message processed"

    idle_agent = _IdleAgent()

    asyncio.run(worker._receive_messages_once())  # noqa: SLF001
    asyncio.run(worker._submit_pending_messages(idle_agent))  # noqa: SLF001

    assert idle_agent.queries
    assert "## Team Messages" in idle_agent.queries[0]
    assert "Summarize the latest handoff." in idle_agent.queries[0]
    lead_messages = Mailbox(teams_root / team.team_id).receive("lead")
    assert {message.type for message in lead_messages} == {"message_ack", "messages_processed"}
