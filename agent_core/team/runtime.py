"""Primary-CLI-led multi-process team runtime."""

from __future__ import annotations

import os
import signal
import sys
from pathlib import Path
from typing import Any, Callable

import subprocess

from agent_core.runtime_paths import application_root, is_frozen_app, tg_agent_home
from agent_core.team.messaging import TeamMessenger
from agent_core.team.models import TeamConfig, TeamMember, TeamMessage, TeamTask, utc_now_iso
from agent_core.team.store import TeamStore
from agent_core.team.task_board import TaskBoard


PopenFactory = Callable[..., subprocess.Popen]
TEAM_WORKER_INTERNAL_FLAG = "--run-team-worker-internal"


def build_team_worker_command(
    *,
    teams_root: Path,
    team_id: str,
    member_id: str,
    agent_type: str,
    workspace_root: Path,
) -> list[str]:
    """Build the teammate subprocess command through the unified main entrypoint."""
    if is_frozen_app():
        command = [sys.executable, TEAM_WORKER_INTERNAL_FLAG]
    else:
        command = [
            sys.executable,
            str(application_root() / "tg_crab_main.py"),
            TEAM_WORKER_INTERNAL_FLAG,
        ]

    command.extend(
        [
            "--teams-root",
            str(teams_root),
            "--team-id",
            team_id,
            "--member-id",
            member_id,
            "--agent-type",
            agent_type,
            "--workspace",
            str(workspace_root),
        ]
    )
    return command


class TeamRuntime:
    """Lead-side runtime. The primary CLI process owns this object."""

    def __init__(
        self,
        *,
        teams_root: Path | None = None,
        workspace_root: Path,
        popen_factory: PopenFactory | None = None,
    ) -> None:
        self.teams_root = (teams_root or (tg_agent_home() / "teams")).expanduser().resolve()
        self.workspace_root = workspace_root.resolve()
        self.store = TeamStore(self.teams_root)
        self._popen_factory = popen_factory or subprocess.Popen

    def create_team(self, *, name: str, goal: str) -> TeamConfig:
        return self.store.create_team(
            name=name,
            goal=goal,
            workspace_root=self.workspace_root,
        )

    def list_teams(self) -> list[TeamConfig]:
        return self.store.list_teams()

    def spawn_member(
        self,
        *,
        team_id: str,
        member_id: str,
        agent_type: str,
        role: str = "member",
    ) -> TeamMember:
        team_dir = self.store.team_dir(team_id)
        self.store.load_config(team_id)
        self.store.ensure_mailbox(team_id, member_id)
        log_path = team_dir / "logs" / f"{member_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = build_team_worker_command(
            teams_root=self.teams_root,
            team_id=team_id,
            member_id=member_id,
            agent_type=agent_type,
            workspace_root=self.workspace_root,
        )
        log_file = log_path.open("ab")
        try:
            process = self._popen_factory(
                command,
                stdout=log_file,
                stderr=log_file,
                cwd=str(self.workspace_root),
            )
        finally:
            log_file.close()
        member = TeamMember(
            member_id=member_id,
            role=role,
            agent_type=agent_type,
            pid=int(process.pid),
            status="running",
        )
        return self.store.upsert_member(team_id, member)

    def stop_member(self, team_id: str, member_id: str) -> TeamMessage:
        message = self.send_message(
            team_id=team_id,
            recipient=member_id,
            body="shutdown",
            type="shutdown",
        )
        for member in self.store.list_members(team_id):
            if member.member_id == member_id and member.pid:
                try:
                    os.kill(member.pid, signal.SIGTERM)
                except OSError:
                    pass
                break
        self.store.mark_member_status(team_id, member_id, "stopping")
        return message

    def shutdown_team(self, team_id: str) -> None:
        for member in self.store.list_members(team_id):
            if member.status not in {"stopped", "completed"}:
                self.stop_member(team_id, member.member_id)
        self.store.update_config_status(team_id, "shutdown")

    def create_task(
        self,
        *,
        team_id: str,
        title: str,
        description: str,
        assigned_to: str | None = None,
        depends_on: list[str] | None = None,
        write_scope: list[str] | None = None,
    ) -> TeamTask:
        board = self._task_board(team_id)
        task = board.create_task(
            title=title,
            description=description,
            assigned_to=assigned_to,
            depends_on=depends_on,
            write_scope=write_scope,
        )
        if assigned_to:
            self.send_message(
                team_id=team_id,
                recipient=assigned_to,
                type="task_assigned",
                body=f"Task assigned: {task.title}",
                metadata={"task_id": task.task_id},
            )
        self.store.append_event(team_id, "task_created", actor="lead", payload=task.to_dict())
        return task

    def list_tasks(self, team_id: str) -> list[TeamTask]:
        return self._task_board(team_id).list_tasks()

    def update_task(
        self,
        *,
        team_id: str,
        task_id: str,
        title: str | None = None,
        description: str | None = None,
        status: str | None = None,
        assigned_to: str | None = None,
        depends_on: list[str] | None = None,
        write_scope: list[str] | None = None,
        result: str | None = None,
        error: str | None = None,
    ) -> TeamTask:
        task = self._task_board(team_id).update_task(
            task_id,
            title=title,
            description=description,
            status=status,
            assigned_to=assigned_to,
            depends_on=depends_on,
            write_scope=write_scope,
            result=result,
            error=error,
        )
        self.store.append_event(team_id, "task_updated", actor="lead", payload=task.to_dict())
        if assigned_to:
            self.send_message(
                team_id=team_id,
                recipient=assigned_to,
                type="task_updated",
                body=f"Task updated: {task.title}",
                metadata={"task_id": task.task_id},
            )
        return task

    def send_message(
        self,
        *,
        team_id: str,
        recipient: str,
        body: str,
        sender: str = "lead",
        type: str = "note",
        metadata: dict[str, Any] | None = None,
        reply_to: str | None = None,
    ) -> TeamMessage:
        return self._messenger(team_id).send_message(
            sender=sender,
            recipient=recipient,
            body=body,
            type=type,
            metadata=metadata,
            reply_to=reply_to,
        )

    def read_lead_inbox(self, team_id: str, *, ack: bool = True) -> list[TeamMessage]:
        return self._messenger(team_id).receive("lead", ack=ack)

    def status(self, team_id: str) -> dict[str, Any]:
        config = self.store.load_config(team_id)
        members = []
        for member in self.store.list_members(team_id):
            heartbeat = self.store.read_heartbeat(team_id, member.member_id)
            item = member.to_dict()
            if heartbeat:
                item["heartbeat"] = heartbeat
                item["last_heartbeat_at"] = heartbeat.get("updated_at")
            members.append(item)
        tasks = [task.to_dict() for task in self.list_tasks(team_id)]
        return {
            "team": config.to_dict(),
            "members": members,
            "tasks": tasks,
            "updated_at": utc_now_iso(),
        }

    def _messenger(self, team_id: str) -> TeamMessenger:
        return TeamMessenger(
            team_id=team_id,
            team_dir=self.store.team_dir(team_id),
            append_event=lambda event_type, actor, payload: self.store.append_event(
                team_id,
                event_type,
                actor=actor,
                payload=payload,
            ),
        )

    def _task_board(self, team_id: str) -> TaskBoard:
        return TaskBoard(self.store.team_dir(team_id))
