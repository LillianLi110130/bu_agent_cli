from __future__ import annotations

import asyncio
import json
import signal
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from agent_core.agent import Agent
from agent_core.agent.events import FinalResponseEvent
from agent_core.agent.registry import AgentRegistry
from agent_core.agent.runtime_events import (
    LLMCallRequested,
    LLMResponseReceived,
    RunFinished,
    ToolCallRequested,
    ToolResultReceived,
)
from agent_core.skill.review import SkillReviewHook
from agent_core.llm.messages import Function, ToolCall, ToolMessage
from agent_core.llm.views import ChatInvokeCompletion
from agent_core.team import Mailbox, TaskBoard, TeamMessage, TeamRuntime
from agent_core.team.experiment import TEAM_EXPERIMENT_ENV
from agent_core.team.models import utc_now_iso
from agent_core.team.runtime import TEAM_WORKER_INTERNAL_FLAG, build_team_worker_command
from cli.team.auto_prompt import (
    TeamAutoParseError,
    build_team_auto_prompt,
    parse_team_auto_request,
)
from cli.team.factory import create_team_member_agent
from cli.team.handler import TeamSlashHandler
from cli.team.event_log import TeamAgentEventLogHook
from cli.team.heartbeat import TeamHeartbeatHook
from cli.team.inbox import TeamInboxAttachmentHook, TeamInboxBuffer
from cli.team.worker import TeamWorker
from rich.console import Console
import tg_crab_main
from tools import SandboxContext
from tools.team_tool import team_create, team_send_message, team_snapshot, team_update_task


class _FakeProcess:
    pid = 4242

    def __init__(self) -> None:
        self.returncode: int | None = None
        self.wait_calls = 0

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.wait_calls += 1
        self.returncode = 0
        return self.returncode


def _fake_popen(*args, **kwargs):
    process = _FakeProcess()
    _fake_popen.calls.append((args, kwargs, process))
    return process


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
    assert [message.type for message in inbox_messages] == ["task_assignment"]
    assert inbox_messages[0].metadata["task_id"] == task.task_id

    runtime.send_message(
        team_id=team.team_id,
        recipient="lead",
        body="done",
        type="task_done_notification",
    )
    lead_messages = runtime.read_lead_inbox(team.team_id)
    assert len(lead_messages) == 1
    assert lead_messages[0].type == "task_done_notification"
    assert lead_messages[0].body == "done"


def test_team_runtime_registers_member_before_process_start(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    teams_root = tmp_path / "teams"
    workspace.mkdir()
    seen_members: list[tuple[str, int | None]] = []

    def popen_factory(*args, **kwargs):
        del args, kwargs
        members = runtime.store.list_members(team.team_id)
        seen_members.extend((member.status, member.pid) for member in members)
        return _FakeProcess()

    runtime = TeamRuntime(
        teams_root=teams_root,
        workspace_root=workspace,
        popen_factory=popen_factory,
    )
    team = runtime.create_team(name="demo", goal="Coordinate")

    member = runtime.spawn_member(
        team_id=team.team_id,
        member_id="worker-1",
        agent_type="general-purpose",
    )

    assert seen_members == [("starting", None)]
    assert member.status == "running"
    assert member.pid == 4242


def test_team_runtime_start_team_sets_active_and_state(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    teams_root = tmp_path / "teams"
    workspace.mkdir()
    runtime = TeamRuntime(teams_root=teams_root, workspace_root=workspace)

    team = runtime.start_team(goal="Implement the team cockpit")

    assert runtime.get_active_team() == team.team_id
    state = runtime.read_state(team.team_id)
    assert state.team_id == team.team_id
    assert state.goal == "Implement the team cockpit"
    assert state.active is True
    status = runtime.status(team.team_id)
    assert status["state"]["active"] is True


def test_team_runtime_rejects_second_active_running_team(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    teams_root = tmp_path / "teams"
    workspace.mkdir()
    runtime = TeamRuntime(teams_root=teams_root, workspace_root=workspace)

    first = runtime.start_team(goal="First team")

    try:
        runtime.start_team(goal="Second team")
    except ValueError as exc:
        assert first.team_id in str(exc)
        assert "/team shutdown" in str(exc)
    else:
        raise AssertionError("expected active team guard")


def test_team_runtime_allows_new_team_after_active_shutdown(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    teams_root = tmp_path / "teams"
    workspace.mkdir()
    runtime = TeamRuntime(teams_root=teams_root, workspace_root=workspace)

    first = runtime.start_team(goal="First team")
    runtime.shutdown_team(first.team_id)

    assert runtime.get_active_team() is None

    second = runtime.start_team(goal="Second team")

    assert runtime.get_active_team() == second.team_id


def test_team_runtime_rejects_shutdown_for_terminal_team(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    teams_root = tmp_path / "teams"
    workspace.mkdir()
    runtime = TeamRuntime(teams_root=teams_root, workspace_root=workspace)
    team = runtime.start_team(goal="First team")
    runtime.shutdown_team(team.team_id)

    try:
        runtime.shutdown_team(team.team_id)
    except ValueError as exc:
        assert "已经关闭" in str(exc)
        assert team.team_id in str(exc)
    else:
        raise AssertionError("expected terminal team shutdown rejection")


def test_team_runtime_request_stop_member_only_requests_shutdown(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "workspace"
    teams_root = tmp_path / "teams"
    workspace.mkdir()
    runtime = TeamRuntime(
        teams_root=teams_root,
        workspace_root=workspace,
        popen_factory=_fake_popen,
    )
    team = runtime.create_team(name="demo", goal="Coordinate")
    runtime.spawn_member(
        team_id=team.team_id,
        member_id="worker-1",
        agent_type="general-purpose",
    )
    kill_calls: list[tuple[int, int]] = []
    monkeypatch.setattr("agent_core.team.runtime.os.kill", lambda pid, sig: kill_calls.append((pid, sig)))

    runtime.request_stop_member(team.team_id, "worker-1")

    assert kill_calls == []
    messages = Mailbox(teams_root / team.team_id).receive("worker-1", ack=False)
    assert [message.type for message in messages] == ["shutdown_request"]
    members = runtime.store.list_members(team.team_id)
    assert members[0].status == "stopping"


def test_team_runtime_stop_member_forces_after_grace_timeout(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "workspace"
    teams_root = tmp_path / "teams"
    workspace.mkdir()
    runtime = TeamRuntime(
        teams_root=teams_root,
        workspace_root=workspace,
        popen_factory=_fake_popen,
    )
    team = runtime.create_team(name="demo", goal="Coordinate")
    runtime.spawn_member(
        team_id=team.team_id,
        member_id="worker-1",
        agent_type="general-purpose",
    )
    kill_calls: list[tuple[int, int]] = []
    monkeypatch.setattr("agent_core.team.runtime.os.kill", lambda pid, sig: kill_calls.append((pid, sig)))

    runtime.stop_member(team.team_id, "worker-1", grace_seconds=0)

    assert kill_calls == [(4242, signal.SIGTERM)]
    messages = Mailbox(teams_root / team.team_id).receive("worker-1", ack=False)
    assert [message.type for message in messages] == ["shutdown_request"]
    assert not runtime._member_processes  # noqa: SLF001


def test_team_runtime_shutdown_forces_member_after_grace_timeout(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "workspace"
    teams_root = tmp_path / "teams"
    workspace.mkdir()
    runtime = TeamRuntime(
        teams_root=teams_root,
        workspace_root=workspace,
        popen_factory=_fake_popen,
    )
    team = runtime.start_team(goal="Coordinate")
    runtime.spawn_member(
        team_id=team.team_id,
        member_id="worker-1",
        agent_type="general-purpose",
    )
    kill_calls: list[tuple[int, int]] = []
    monkeypatch.setattr("agent_core.team.runtime.os.kill", lambda pid, sig: kill_calls.append((pid, sig)))

    runtime.shutdown_team(team.team_id, grace_seconds=0)

    assert kill_calls == [(4242, signal.SIGTERM)]
    assert runtime.get_active_team() is None
    assert not runtime._member_processes  # noqa: SLF001


def test_team_runtime_reaps_exited_member_process(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    teams_root = tmp_path / "teams"
    workspace.mkdir()
    runtime = TeamRuntime(
        teams_root=teams_root,
        workspace_root=workspace,
        popen_factory=_fake_popen,
    )
    team = runtime.create_team(name="demo", goal="Coordinate")
    runtime.spawn_member(
        team_id=team.team_id,
        member_id="worker-1",
        agent_type="general-purpose",
    )
    process = _fake_popen.calls[-1][2]
    process.returncode = 0

    assert runtime.reap_member_process(team.team_id, "worker-1") == 0
    assert process.wait_calls == 1
    assert not runtime._member_processes  # noqa: SLF001


def test_team_auto_prompt_parses_goal_and_keeps_orchestration_model_driven() -> None:
    request = parse_team_auto_request(
        ["Refactor", "auth", "module", "--name", "auth-refactor"]
    )

    assert request.goal == "Refactor auth module"
    assert request.name == "auth-refactor"

    prompt = build_team_auto_prompt(request)
    assert "Team goal:\nRefactor auth module" in prompt
    assert "Requested team name:\nauth-refactor" in prompt
    assert "Language rules:" in prompt
    assert "prefer Chinese" in prompt
    assert "Do not use a hardcoded fixed team shape" in prompt
    assert "team_snapshot" in prompt


def test_team_auto_prompt_requires_goal() -> None:
    try:
        parse_team_auto_request(["--name", "demo"])
    except TeamAutoParseError as exc:
        assert "goal" in str(exc)
    else:
        raise AssertionError("expected TeamAutoParseError")


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


def test_task_board_does_not_claim_unassigned_tasks(tmp_path: Path) -> None:
    team_dir = tmp_path / "team"
    team_dir.mkdir()
    board = TaskBoard(team_dir)
    board.create_task(
        title="Unassigned",
        description="Lead must assign this explicitly",
    )

    assert board.claim_next("worker-1") is None


def test_task_board_rejects_missing_self_and_cyclic_dependencies(tmp_path: Path) -> None:
    team_dir = tmp_path / "team"
    team_dir.mkdir()
    board = TaskBoard(team_dir)
    task_a = board.create_task(title="Task A", description="A")
    task_b = board.create_task(title="Task B", description="B", depends_on=[task_a.task_id])

    try:
        board.create_task(title="Missing", description="Missing", depends_on=["task_missing"])
    except ValueError as exc:
        assert "Create dependency tasks first" in str(exc)
    else:
        raise AssertionError("expected missing dependency rejection")

    try:
        board.update_task(task_a.task_id, depends_on=[task_a.task_id])
    except ValueError as exc:
        assert "cannot depend on itself" in str(exc)
    else:
        raise AssertionError("expected self dependency rejection")

    try:
        board.update_task(task_a.task_id, depends_on=[task_b.task_id])
    except ValueError as exc:
        assert "cycle" in str(exc)
    else:
        raise AssertionError("expected cyclic dependency rejection")


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
        type="message",
    )

    assert message.sender == "explorer-1"
    assert message.recipient == "coder-1"
    inbox_messages = Mailbox(teams_root / team.team_id).receive("coder-1")
    assert [item.message_id for item in inbox_messages] == [message.message_id]
    assert inbox_messages[0].type == "message"


def test_team_snapshot_groups_tasks_members_and_peeks_inbox(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    teams_root = tmp_path / "teams"
    workspace.mkdir()
    runtime = TeamRuntime(
        teams_root=teams_root,
        workspace_root=workspace,
        popen_factory=_fake_popen,
    )
    team = runtime.create_team(name="demo", goal="Coordinate")
    runtime.spawn_member(
        team_id=team.team_id,
        member_id="explorer-1",
        agent_type="explore",
    )
    pending = runtime.create_task(
        team_id=team.team_id,
        title="Pending task",
        description="Wait for worker",
    )
    blocked = runtime.create_task(
        team_id=team.team_id,
        title="Blocked task",
        description="Needs guidance",
    )
    runtime.update_task(
        team_id=team.team_id,
        task_id=blocked.task_id,
        status="blocked",
        error="Missing context",
    )
    runtime.send_message(
        team_id=team.team_id,
        sender="explorer-1",
        recipient="lead",
        body="Need more context.",
        type="message",
    )

    snapshot = runtime.snapshot(team.team_id)

    assert snapshot["summary"]["tasks"]["pending"] == 1
    assert snapshot["summary"]["tasks"]["blocked"] == 1
    assert snapshot["summary"]["lead_inbox_unread"] == 1
    assert snapshot["tasks_by_status"]["pending"][0]["task_id"] == pending.task_id
    assert snapshot["tasks_by_status"]["blocked"][0]["task_id"] == blocked.task_id
    assert snapshot["lead_inbox"][0]["body"] == "Need more context."
    assert any("blocked" in warning for warning in snapshot["warnings"])
    assert "inspect blocked tasks" in snapshot["suggested_next"]
    still_unread = runtime.read_lead_inbox(team.team_id)
    assert len(still_unread) == 1


def test_team_snapshot_groups_idle_member_by_heartbeat(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    teams_root = tmp_path / "teams"
    workspace.mkdir()
    runtime = TeamRuntime(
        teams_root=teams_root,
        workspace_root=workspace,
        popen_factory=_fake_popen,
    )
    team = runtime.create_team(name="demo", goal="Coordinate")
    runtime.spawn_member(
        team_id=team.team_id,
        member_id="frontend-1",
        agent_type="frontend",
    )
    runtime.store.write_heartbeat(
        team.team_id,
        "frontend-1",
        {
            "team_id": team.team_id,
            "member_id": "frontend-1",
            "status": "idle",
            "updated_at": utc_now_iso(),
        },
    )

    snapshot = runtime.snapshot(team.team_id)

    assert snapshot["summary"]["members"]["idle"] == 1
    assert "running" not in snapshot["summary"]["members"]
    assert "running" not in snapshot["members_by_state"]
    assert snapshot["members_by_state"]["idle"][0]["member_id"] == "frontend-1"


def test_team_snapshot_default_stale_timeout_is_300_seconds(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    teams_root = tmp_path / "teams"
    workspace.mkdir()
    runtime = TeamRuntime(
        teams_root=teams_root,
        workspace_root=workspace,
        popen_factory=_fake_popen,
    )
    team = runtime.create_team(name="demo", goal="Coordinate")
    runtime.spawn_member(
        team_id=team.team_id,
        member_id="frontend-1",
        agent_type="frontend",
    )
    runtime.store.write_heartbeat(
        team.team_id,
        "frontend-1",
        {
            "team_id": team.team_id,
            "member_id": "frontend-1",
            "status": "idle",
            "updated_at": (datetime.now(timezone.utc) - timedelta(seconds=240)).isoformat(),
        },
    )

    snapshot = runtime.snapshot(team.team_id)

    assert snapshot["summary"]["members"]["idle"] == 1
    assert snapshot["summary"]["members"]["stale"] == 0


def test_team_snapshot_treats_legacy_running_member_as_idle(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    teams_root = tmp_path / "teams"
    workspace.mkdir()
    runtime = TeamRuntime(
        teams_root=teams_root,
        workspace_root=workspace,
        popen_factory=_fake_popen,
    )
    team = runtime.create_team(name="demo", goal="Coordinate")
    runtime.spawn_member(
        team_id=team.team_id,
        member_id="legacy-1",
        agent_type="general-purpose",
    )
    runtime.store.write_heartbeat(
        team.team_id,
        "legacy-1",
        {
            "team_id": team.team_id,
            "member_id": "legacy-1",
            "status": "running",
            "updated_at": utc_now_iso(),
        },
    )

    snapshot = runtime.snapshot(team.team_id)

    assert snapshot["summary"]["members"]["idle"] == 1
    assert "running" not in snapshot["summary"]["members"]
    assert snapshot["members_by_state"]["idle"][0]["member_id"] == "legacy-1"


def test_team_snapshot_groups_failed_member(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    teams_root = tmp_path / "teams"
    workspace.mkdir()
    runtime = TeamRuntime(
        teams_root=teams_root,
        workspace_root=workspace,
        popen_factory=_fake_popen,
    )
    team = runtime.create_team(name="demo", goal="Coordinate")
    runtime.spawn_member(
        team_id=team.team_id,
        member_id="backend-1",
        agent_type="backend",
    )
    runtime.store.mark_member_status(team.team_id, "backend-1", "failed")
    runtime.store.write_heartbeat(
        team.team_id,
        "backend-1",
        {
            "team_id": team.team_id,
            "member_id": "backend-1",
            "status": "failed",
            "updated_at": utc_now_iso(),
            "error": "boom",
        },
    )

    snapshot = runtime.snapshot(team.team_id)

    assert snapshot["summary"]["members"]["failed"] == 1
    assert snapshot["members_by_state"]["failed"][0]["member_id"] == "backend-1"
    assert any("backend-1 failed" in warning for warning in snapshot["warnings"])


def test_team_snapshot_tool_returns_snapshot(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    teams_root = tmp_path / "teams"
    workspace.mkdir()
    runtime = TeamRuntime(teams_root=teams_root, workspace_root=workspace)
    team = runtime.create_team(name="demo", goal="Coordinate")
    runtime.create_task(
        team_id=team.team_id,
        title="Task",
        description="Description",
    )

    result = asyncio.run(
        team_snapshot.func(
            ctx=SimpleNamespace(team_runtime=runtime),
            team_id=team.team_id,
        )
    )

    payload = json.loads(result)
    assert payload["team"]["team_id"] == team.team_id
    assert payload["summary"]["tasks"]["pending"] == 1
    assert "tasks_by_status" in payload


def test_team_slash_handler_create_and_list(tmp_path: Path) -> None:
    runtime = TeamRuntime(
        teams_root=tmp_path / "teams",
        workspace_root=tmp_path,
        popen_factory=_fake_popen,
    )
    console = Console(record=True, width=100)
    handler = TeamSlashHandler(runtime=runtime, console=console)

    asyncio.run(handler.handle(["create", "Analyze", "repo", "--name", "demo"]))
    asyncio.run(handler.handle(["list"]))

    output = console.export_text()
    assert "已创建并切换到 team" in output
    assert "demo-" in output
    assert "Analyze repo" in output


def test_team_slash_handler_rejects_start_alias(tmp_path: Path) -> None:
    runtime = TeamRuntime(
        teams_root=tmp_path / "teams",
        workspace_root=tmp_path,
        popen_factory=_fake_popen,
    )
    console = Console(record=True, width=100)
    handler = TeamSlashHandler(runtime=runtime, console=console)

    asyncio.run(handler.handle(["start", "Analyze", "repo", "--name", "demo"]))

    output = console.export_text()
    assert "未知 /team 子命令：start" in output
    assert runtime.list_teams() == []


def test_team_slash_handler_ui_prints_static_dashboard_paths(tmp_path: Path) -> None:
    runtime = TeamRuntime(
        teams_root=tmp_path / "teams",
        workspace_root=tmp_path,
        popen_factory=_fake_popen,
    )
    team = runtime.start_team(goal="Observe team")
    console = Console(record=True, width=140)
    handler = TeamSlashHandler(runtime=runtime, console=console)

    asyncio.run(handler.handle(["ui"]))

    output = console.export_text()
    assert "Team dashboard 是纯 HTML" in output
    assert "team_dashboard.html" in output
    assert str(runtime.store.team_dir(team.team_id)).replace("\n", "") in output.replace("\n", "")


def test_team_slash_handler_create_and_active_cockpit(tmp_path: Path) -> None:
    runtime = TeamRuntime(
        teams_root=tmp_path / "teams",
        workspace_root=tmp_path,
        popen_factory=_fake_popen,
    )
    console = Console(record=True, width=120)
    handler = TeamSlashHandler(runtime=runtime, console=console)

    asyncio.run(handler.handle(["create", "Build", "a", "team", "cockpit"]))
    active = runtime.get_active_team()
    assert active is not None

    asyncio.run(handler.handle([]))
    asyncio.run(handler.handle(["task", "Check", "status", "--to", "explorer-1"]))
    asyncio.run(handler.handle(["tasks"]))

    output = console.export_text()
    assert "已创建并切换到 team" in output
    assert active in output
    assert "Build a team cockpit" in output
    assert "Check status" in output


def test_team_slash_handler_rejects_use_command(tmp_path: Path) -> None:
    runtime = TeamRuntime(
        teams_root=tmp_path / "teams",
        workspace_root=tmp_path,
        popen_factory=_fake_popen,
    )
    console = Console(record=True, width=100)
    handler = TeamSlashHandler(runtime=runtime, console=console)

    asyncio.run(handler.handle(["use", "some-team"]))

    output = console.export_text()
    assert "未知 /team 子命令：use" in output


def test_team_slash_handler_rejects_second_active_team(tmp_path: Path) -> None:
    runtime = TeamRuntime(
        teams_root=tmp_path / "teams",
        workspace_root=tmp_path,
        popen_factory=_fake_popen,
    )
    console = Console(record=True, width=120)
    handler = TeamSlashHandler(runtime=runtime, console=console)

    asyncio.run(handler.handle(["create", "First", "team"]))
    asyncio.run(handler.handle(["create", "Second", "team"]))

    output = console.export_text()
    assert "already has an active running team" in output
    assert "/team shutdown" in output


def test_team_slash_handler_rejects_shutdown_for_terminal_team(tmp_path: Path) -> None:
    runtime = TeamRuntime(
        teams_root=tmp_path / "teams",
        workspace_root=tmp_path,
        popen_factory=_fake_popen,
    )
    team = runtime.start_team(goal="Done team")
    runtime.shutdown_team(team.team_id)
    console = Console(record=True, width=120)
    handler = TeamSlashHandler(runtime=runtime, console=console)

    asyncio.run(handler.handle(["shutdown", team.team_id]))

    output = console.export_text()
    assert "已经关闭" in output
    assert "已请求关闭 team" not in output


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


def test_tg_crab_main_shutdowns_active_team_on_exit() -> None:
    class _Runtime:
        def __init__(self) -> None:
            self.shutdowns: list[str] = []

        def get_active_team(self) -> str:
            return "team-1"

        def shutdown_team(self, team_id: str) -> None:
            self.shutdowns.append(team_id)

    runtime = _Runtime()
    ctx = SimpleNamespace(team_runtime=runtime)

    asyncio.run(tg_crab_main._shutdown_active_team_on_exit(ctx))

    assert runtime.shutdowns == ["team-1"]


def test_tg_crab_main_shutdown_active_team_noops_without_active_team() -> None:
    class _Runtime:
        def __init__(self) -> None:
            self.shutdowns: list[str] = []

        def get_active_team(self) -> None:
            return None

        def shutdown_team(self, team_id: str) -> None:
            self.shutdowns.append(team_id)

    runtime = _Runtime()
    ctx = SimpleNamespace(team_runtime=runtime)

    asyncio.run(tg_crab_main._shutdown_active_team_on_exit(ctx))

    assert runtime.shutdowns == []


def test_tg_crab_main_shutdown_active_team_silently_skips_terminal_team(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    teams_root = tmp_path / "teams"
    workspace.mkdir()
    runtime = TeamRuntime(teams_root=teams_root, workspace_root=workspace)
    team = runtime.start_team(goal="Terminal team")
    runtime.shutdown_team(team.team_id)
    runtime.set_active_team(team.team_id)
    ctx = SimpleNamespace(team_runtime=runtime)

    asyncio.run(tg_crab_main._shutdown_active_team_on_exit(ctx))

    state = runtime.read_state(team.team_id)
    assert state.active is False
    assert runtime.get_active_team() == team.team_id


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
    assert "Agent team 实验功能未启用" in output
    assert TEAM_EXPERIMENT_ENV in output


def test_team_slash_handler_empty_cockpit_suggests_auto(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    teams_root = tmp_path / "teams"
    workspace.mkdir()
    runtime = TeamRuntime(teams_root=teams_root, workspace_root=workspace)
    console = Console(record=True, width=100)
    handler = TeamSlashHandler(runtime=runtime, console=console)

    asyncio.run(handler.handle([]))

    output = console.export_text()
    assert "当前 workspace 没有 active team" in output
    assert "/team create <goal>" in output
    assert "/team auto <goal>" in output
    assert "[--name <name>]" in output


def test_team_slash_handler_usage_preserves_optional_brackets(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    teams_root = tmp_path / "teams"
    workspace.mkdir()
    runtime = TeamRuntime(teams_root=teams_root, workspace_root=workspace)
    console = Console(record=True, width=100)
    handler = TeamSlashHandler(runtime=runtime, console=console)

    asyncio.run(handler.handle(["status"]))

    output = console.export_text()
    assert "用法：/team status [team_id]" in output


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


def test_team_create_tool_sets_active_team_by_default(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    teams_root = tmp_path / "teams"
    workspace.mkdir()
    runtime = TeamRuntime(teams_root=teams_root, workspace_root=workspace)

    result = asyncio.run(
        team_create.func(
            ctx=SimpleNamespace(team_runtime=runtime),
            name="demo",
            goal="Coordinate",
        )
    )

    payload = json.loads(result)
    assert runtime.get_active_team() == payload["team_id"]


def test_team_create_tool_rejects_second_active_team(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    teams_root = tmp_path / "teams"
    workspace.mkdir()
    runtime = TeamRuntime(teams_root=teams_root, workspace_root=workspace)

    asyncio.run(
        team_create.func(
            ctx=SimpleNamespace(team_runtime=runtime),
            name="demo",
            goal="Coordinate",
        )
    )
    result = asyncio.run(
        team_create.func(
            ctx=SimpleNamespace(team_runtime=runtime),
            name="demo-two",
            goal="Coordinate again",
        )
    )

    assert result.startswith("Error:")
    assert "already has an active running team" in result


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
    dependency = runtime.create_task(
        team_id=team.team_id,
        title="Dependency",
        description="Dependency",
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
            depends_on=[dependency.task_id],
            write_scope=["new.py"],
            error="Waiting for dependency",
        )
    )

    payload = json.loads(result)
    assert payload["title"] == "New title"
    assert payload["description"] == "New description"
    assert payload["status"] == "blocked"
    assert payload["assigned_to"] == "coder-1"
    assert payload["depends_on"] == [dependency.task_id]
    assert payload["write_scope"] == ["new.py"]
    assert payload["error"] == "Waiting for dependency"
    messages = Mailbox(teams_root / team.team_id).receive("coder-1")
    assert [message.type for message in messages] == ["task_assignment"]


def test_team_update_task_sends_update_when_assignment_does_not_change(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    teams_root = tmp_path / "teams"
    workspace.mkdir()
    runtime = TeamRuntime(teams_root=teams_root, workspace_root=workspace)
    team = runtime.create_team(name="demo", goal="Coordinate")
    task = runtime.create_task(
        team_id=team.team_id,
        title="Old title",
        description="Old description",
        assigned_to="coder-1",
    )
    Mailbox(teams_root / team.team_id).receive("coder-1")

    runtime.update_task(
        team_id=team.team_id,
        task_id=task.task_id,
        title="New title",
        assigned_to="coder-1",
    )

    messages = Mailbox(teams_root / team.team_id).receive("coder-1")
    assert [message.type for message in messages] == ["task_update"]


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
        getattr(tool, "name", None)
        not in {
            "delegate",
            "delegate_parallel",
            "skill_manage",
            "team_create",
            "team_spawn_member",
            "team_create_task",
            "team_update_task",
            "team_shutdown",
        }
        for tool in agent.tools
    )
    assert not any(isinstance(hook, SkillReviewHook) for hook in agent.hooks)


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
    assert isinstance(ctx.team_runtime, TeamRuntime)
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
    assert "Language rules:" in agent.system_prompt
    assert "prefer Chinese" in agent.system_prompt
    assert "Team context:" in agent.system_prompt
    assert "Lead: lead" in agent.system_prompt
    assert "You: explorer-1 (agent_type=explore)" in agent.system_prompt
    assert str(teams_root / team.team_id / "members.json") in agent.system_prompt
    assert "Read the members registry" in agent.system_prompt
    assert "coordinate with other teammates" in agent.system_prompt
    assert "Skills are read-only for teammates" in agent.system_prompt
    assert "send a message to the team lead with the exact question" in agent.system_prompt
    assert "clarification_request" in agent.system_prompt
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
        type="message",
    )

    messages = Mailbox(teams_root / team.team_id).receive("coder-1")
    assert len(messages) == 1
    assert messages[0].sender == "explorer-1"
    assert messages[0].recipient == "coder-1"
    assert messages[0].type == "message"


def test_team_send_message_tool_preserves_metadata_and_reply_to(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    teams_root = tmp_path / "teams"
    workspace.mkdir()
    runtime = TeamRuntime(teams_root=teams_root, workspace_root=workspace)
    team = runtime.create_team(name="demo", goal="Coordinate")

    result = asyncio.run(
        team_send_message.func(
            ctx=SimpleNamespace(team_runtime=runtime),
            team_id=team.team_id,
            sender="explorer-1",
            recipient="lead",
            type="clarification_request",
            body="Which API shape should I use?",
            metadata={
                "task_id": "task_1",
                "question": "Which API shape should I use?",
                "options": ["REST", "GraphQL"],
                "recommended": "REST",
                "blocking": True,
            },
            reply_to="msg-parent",
        )
    )

    payload = json.loads(result)
    assert payload["type"] == "clarification_request"
    assert payload["metadata"]["blocking"] is True
    assert payload["metadata"]["recommended"] == "REST"
    assert payload["reply_to"] == "msg-parent"


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
                type="message",
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


def test_team_agent_event_log_hook_writes_compact_jsonl(tmp_path: Path) -> None:
    event_log = tmp_path / "events.jsonl"
    hook = TeamAgentEventLogHook(
        team_id="team-1",
        member_id="explorer-1",
        event_log_path=event_log,
    )
    tool_call = ToolCall(
        id="call_1",
        function=Function(name="bash", arguments='{"cmd":"pytest tests/test_team_runtime.py"}'),
    )

    async def run_case() -> None:
        await hook.before_event(
            LLMCallRequested(messages=[], tools=[], tool_choice=None, iteration=1),
            SimpleNamespace(),
        )
        await hook.before_event(
            LLMResponseReceived(
                response=ChatInvokeCompletion(content="I will run tests.", tool_calls=[tool_call]),
                iteration=1,
            ),
            SimpleNamespace(),
        )
        await hook.before_event(
            ToolCallRequested(tool_call=tool_call, iteration=1),
            SimpleNamespace(),
        )
        await hook.before_event(
            ToolResultReceived(
                tool_call=tool_call,
                tool_result=ToolMessage(
                    tool_call_id="call_1",
                    tool_name="bash",
                    content="38 passed",
                    is_error=False,
                ),
                iteration=1,
            ),
            SimpleNamespace(),
        )
        await hook.before_event(
            RunFinished(final_response="done", iterations=1),
            SimpleNamespace(),
        )

    asyncio.run(run_case())

    records = [json.loads(line) for line in event_log.read_text(encoding="utf-8").splitlines()]
    assert [record["type"] for record in records] == [
        "llm_call_requested",
        "llm_response_received",
        "tool_call_requested",
        "tool_result_received",
        "run_finished",
    ]
    assert all(record["team_id"] == "team-1" for record in records)
    assert all(record["member_id"] == "explorer-1" for record in records)
    assert records[0]["payload"]["message_count"] == 0
    assert records[1]["payload"]["tool_calls"] == [{"id": "call_1", "name": "bash"}]
    assert records[2]["payload"]["args_preview"] == '{"cmd":"pytest tests/test_team_runtime.py"}'
    assert records[3]["payload"]["result_preview"] == "38 passed"


def test_team_heartbeat_hook_refreshes_at_agent_loop_boundaries() -> None:
    writes: list[tuple[str, dict]] = []
    state = {"status": "working", "extra": {"task_id": "task-1"}}
    hook = TeamHeartbeatHook(
        write_heartbeat=lambda status, **extra: writes.append((status, extra)),
        get_state=lambda: (state["status"], dict(state["extra"])),
    )
    tool_call = ToolCall(
        id="call_1",
        function=Function(name="bash", arguments='{"cmd":"pytest"}'),
    )

    async def run_case() -> None:
        await hook.before_event(
            LLMCallRequested(messages=[], tools=[], tool_choice=None, iteration=1),
            SimpleNamespace(),
        )
        await hook.before_event(
            ToolCallRequested(tool_call=tool_call, iteration=1),
            SimpleNamespace(),
        )
        await hook.after_event(
            ToolResultReceived(
                tool_call=tool_call,
                tool_result=ToolMessage(
                    tool_call_id="call_1",
                    tool_name="bash",
                    content="ok",
                    is_error=False,
                ),
                iteration=1,
            ),
            SimpleNamespace(),
            [],
        )

    asyncio.run(run_case())

    assert writes == [
        (
            "working",
            {
                "task_id": "task-1",
                "heartbeat_reason": "llm_call_requested",
                "iteration": 1,
            },
        ),
        (
            "working",
            {
                "task_id": "task-1",
                "heartbeat_reason": "tool_call_requested",
                "iteration": 1,
                "current_tool": "bash",
                "current_tool_call_id": "call_1",
            },
        ),
        (
            "working",
            {
                "task_id": "task-1",
                "heartbeat_reason": "tool_result_received",
                "iteration": 1,
                "last_tool": "bash",
                "last_tool_call_id": "call_1",
                "last_tool_error": False,
            },
        ),
    ]


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
        type="message",
    )

    class _IdleAgent:
        def __init__(self) -> None:
            self.queries: list[str] = []

        async def query_stream(self, prompt: str):
            self.queries.append(prompt)
            yield FinalResponseEvent(content="message processed")

    idle_agent = _IdleAgent()

    asyncio.run(worker._receive_messages_once())  # noqa: SLF001
    asyncio.run(worker._submit_pending_messages(idle_agent))  # noqa: SLF001

    assert idle_agent.queries
    assert "## Team Messages" in idle_agent.queries[0]
    assert "Summarize the latest handoff." in idle_agent.queries[0]
    lead_messages = Mailbox(teams_root / team.team_id).receive("lead")
    assert {message.type for message in lead_messages} == {"message_ack", "message"}


def test_team_worker_sends_idle_notification_once_per_idle_stretch(tmp_path: Path) -> None:
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

    worker._mark_idle(reason="no_claimable_task")  # noqa: SLF001
    worker._mark_idle(reason="no_claimable_task")  # noqa: SLF001

    messages = Mailbox(teams_root / team.team_id).receive("lead")
    idle_messages = [message for message in messages if message.type == "idle_notification"]
    assert len(idle_messages) == 1
    assert idle_messages[0].metadata["reason"] == "no_claimable_task"
    assert idle_messages[0].metadata["previous_status"] == "started"

    worker._set_heartbeat_state("working", task_id="task-1")  # noqa: SLF001
    worker._clear_heartbeat_state()  # noqa: SLF001
    worker._reset_idle_notification()  # noqa: SLF001
    worker._mark_idle(reason="no_claimable_task")  # noqa: SLF001

    messages = Mailbox(teams_root / team.team_id).receive("lead")
    idle_messages = [message for message in messages if message.type == "idle_notification"]
    assert len(idle_messages) == 1
    assert idle_messages[0].metadata["previous_status"] == "working"


def test_team_worker_completes_task_from_query_stream_final_response(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    teams_root = tmp_path / "teams"
    workspace.mkdir()
    store_runtime = TeamRuntime(teams_root=teams_root, workspace_root=workspace)
    team = store_runtime.create_team(name="demo", goal="Coordinate")
    store_runtime.create_task(
        team_id=team.team_id,
        title="Implement feature",
        description="Do the work",
        assigned_to="explorer-1",
    )
    worker = TeamWorker(
        teams_root=teams_root,
        team_id=team.team_id,
        member_id="explorer-1",
        agent_type="explore",
        workspace=workspace,
    )

    class _StreamAgent:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        async def query_stream(self, prompt: str):
            self.prompts.append(prompt)
            yield FinalResponseEvent(content="streamed task result")

    agent = _StreamAgent()
    claimed_task = worker.task_board.claim_next("explorer-1")
    assert claimed_task is not None

    asyncio.run(worker._run_task(agent, claimed_task))  # noqa: SLF001

    assert agent.prompts
    tasks = store_runtime.list_tasks(team.team_id)
    assert tasks[0].status == "completed"
    assert tasks[0].result == "streamed task result"
    lead_messages = Mailbox(teams_root / team.team_id).receive("lead")
    assert "task_done_notification" in {message.type for message in lead_messages}


def test_team_worker_blocks_task_when_query_stream_returns_cancelled(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    teams_root = tmp_path / "teams"
    workspace.mkdir()
    store_runtime = TeamRuntime(teams_root=teams_root, workspace_root=workspace)
    team = store_runtime.create_team(name="demo", goal="Coordinate")
    store_runtime.create_task(
        team_id=team.team_id,
        title="Implement feature",
        description="Do the work",
        assigned_to="explorer-1",
    )
    worker = TeamWorker(
        teams_root=teams_root,
        team_id=team.team_id,
        member_id="explorer-1",
        agent_type="explore",
        workspace=workspace,
    )

    class _CancelledAgent:
        async def query_stream(self, prompt: str):
            del prompt
            yield FinalResponseEvent(content="[Cancelled by user]")

    claimed_task = worker.task_board.claim_next("explorer-1")
    assert claimed_task is not None

    asyncio.run(worker._run_task(_CancelledAgent(), claimed_task))  # noqa: SLF001

    tasks = store_runtime.list_tasks(team.team_id)
    assert tasks[0].status == "blocked"
    assert tasks[0].result is None
    assert tasks[0].error == "Task was cancelled by user"
    lead_messages = Mailbox(teams_root / team.team_id).receive("lead")
    blocked_messages = [
        message for message in lead_messages if message.type == "task_blocked_notification"
    ]
    assert len(blocked_messages) == 1
    assert blocked_messages[0].metadata["cancelled"] is True


def test_team_worker_marks_member_failed_on_unhandled_exception(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    teams_root = tmp_path / "teams"
    workspace.mkdir()
    store_runtime = TeamRuntime(
        teams_root=teams_root,
        workspace_root=workspace,
        popen_factory=_fake_popen,
    )
    team = store_runtime.create_team(name="demo", goal="Coordinate")
    store_runtime.spawn_member(
        team_id=team.team_id,
        member_id="explorer-1",
        agent_type="explore",
    )

    class _RuntimeFailAgent:
        def register_hook(self, hook) -> None:
            self.hook = hook

        def bind_session_runtime(self, runtime) -> None:
            self.runtime = runtime

    def fake_create_agent(*args, **kwargs):
        del args, kwargs
        return _RuntimeFailAgent(), SandboxContext.create(workspace), SimpleNamespace()

    worker = TeamWorker(
        teams_root=teams_root,
        team_id=team.team_id,
        member_id="explorer-1",
        agent_type="explore",
        workspace=workspace,
        create_agent_factory=fake_create_agent,
    )

    async def quiet_message_pump() -> None:
        await asyncio.sleep(10)

    async def failing_work_loop(agent) -> None:
        del agent
        raise RuntimeError("worker boom")

    worker._message_pump = quiet_message_pump  # type: ignore[method-assign]  # noqa: SLF001
    worker._work_loop = failing_work_loop  # type: ignore[method-assign]  # noqa: SLF001

    try:
        asyncio.run(worker.run())
    except RuntimeError as exc:
        assert str(exc) == "worker boom"
    else:
        raise AssertionError("expected worker failure")

    members = store_runtime.store.list_members(team.team_id)
    assert members[0].status == "failed"
    heartbeat = store_runtime.store.read_heartbeat(team.team_id, "explorer-1")
    assert heartbeat is not None
    assert heartbeat["status"] == "failed"
    assert heartbeat["error"] == "worker boom"
    lead_messages = Mailbox(teams_root / team.team_id).receive("lead")
    assert "worker_failed" in {message.type for message in lead_messages}
