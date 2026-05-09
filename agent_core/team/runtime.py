"""Primary-CLI-led multi-process team runtime."""

from __future__ import annotations

from datetime import datetime, timezone
import os
import signal
import sys
from pathlib import Path
from typing import Any, Callable

import subprocess

from agent_core.runtime_paths import application_root, is_frozen_app, tg_agent_home
from agent_core.team.messaging import TeamMessenger
from agent_core.team.models import (
    TeamConfig,
    TeamMember,
    TeamMessage,
    TeamState,
    TeamTask,
    utc_now_iso,
)
from agent_core.team.store import TeamStore
from agent_core.team.task_board import TaskBoard


PopenFactory = Callable[..., subprocess.Popen]
TEAM_WORKER_INTERNAL_FLAG = "--run-team-worker-internal"
TERMINAL_TEAM_STATUSES = {"shutdown", "completed", "failed"}
TERMINAL_TEAM_PHASES = {"shutdown", "completed", "failed"}


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

    def start_team(self, *, goal: str, name: str | None = None) -> TeamConfig:
        self.ensure_can_start_team()
        team = self.create_team(name=name or self._slug_from_goal(goal), goal=goal)
        self.set_active_team(team.team_id)
        return team

    def list_teams(self) -> list[TeamConfig]:
        return self.store.list_teams()

    def set_active_team(self, team_id: str) -> None:
        self.store.set_active_team(workspace_root=self.workspace_root, team_id=team_id)

    def clear_active_team(self, team_id: str | None = None) -> None:
        self.store.clear_active_team(workspace_root=self.workspace_root, team_id=team_id)

    def get_active_team(self) -> str | None:
        return self.store.get_active_team(workspace_root=self.workspace_root)

    def get_active_running_team(self) -> TeamConfig | None:
        team_id = self.get_active_team()
        if team_id is None:
            return None
        try:
            config = self.store.load_config(team_id)
            state = self.store.read_state(team_id)
        except FileNotFoundError:
            return None
        if config.status in TERMINAL_TEAM_STATUSES:
            return None
        if not state.active:
            return None
        if state.phase in TERMINAL_TEAM_PHASES:
            return None
        return config

    def ensure_can_start_team(self) -> None:
        active = self.get_active_running_team()
        if active is None:
            return
        raise ValueError(
            "Current workspace already has an active running team: "
            f"{active.team_id}. Use /team shutdown before creating a new team, "
            f"or continue with /team use {active.team_id}."
        )

    def read_state(self, team_id: str) -> TeamState:
        return self.store.read_state(team_id)

    def update_phase(self, team_id: str, phase: str) -> TeamState:
        return self.store.update_phase(team_id, phase)

    def spawn_member(
        self,
        *,
        team_id: str,
        member_id: str,
        agent_type: str,
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
            agent_type=agent_type,
            pid=int(process.pid),
            status="running",
        )
        return self.store.upsert_member(team_id, member)

    def stop_member(self, team_id: str, member_id: str) -> TeamMessage:
        message = self.send_message(
            team_id=team_id,
            recipient=member_id,
            body="shutdown requested",
            type="shutdown_request",
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
            if member.status not in {"stopped", "completed", "failed"}:
                self.stop_member(team_id, member.member_id)
        self.store.update_config_status(team_id, "shutdown")
        self.store.write_state(team_id, active=False, phase="shutdown")
        self.clear_active_team(team_id)

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
                type="task_assignment",
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
        self.store.append_event(team_id, "task_update", actor="lead", payload=task.to_dict())
        if assigned_to:
            self.send_message(
                team_id=team_id,
                recipient=assigned_to,
                type="task_update",
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
        type: str = "message",
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

    def ack_lead_messages(self, team_id: str, message_ids: set[str]) -> list[TeamMessage]:
        return self._messenger(team_id).ack("lead", message_ids)

    def status(self, team_id: str) -> dict[str, Any]:
        config = self.store.load_config(team_id)
        state = self.store.read_state(team_id)
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
            "state": state.to_dict(),
            "members": members,
            "tasks": tasks,
            "updated_at": utc_now_iso(),
        }

    def snapshot(
        self,
        team_id: str,
        *,
        include_inbox: bool = True,
        ack_inbox: bool = False,
        stale_after_seconds: int = 120,
    ) -> dict[str, Any]:
        status = self.status(team_id)
        tasks = status["tasks"]
        members = status["members"]
        tasks_by_status: dict[str, list[dict[str, Any]]] = {
            "pending": [],
            "in_progress": [],
            "blocked": [],
            "completed": [],
        }
        for task in tasks:
            tasks_by_status.setdefault(task["status"], []).append(task)

        members_by_state: dict[str, list[dict[str, Any]]] = {
            "idle": [],
            "working": [],
            "stale": [],
            "stopped": [],
            "failed": [],
        }
        warnings: list[str] = []
        now = datetime.now(timezone.utc)
        for member in members:
            heartbeat = member.get("heartbeat") or {}
            heartbeat_status = str(heartbeat.get("status") or "")
            status_name = heartbeat_status or str(member.get("status") or "running")
            updated_at = heartbeat.get("updated_at") or member.get("last_heartbeat_at")
            is_stale = self._is_stale(updated_at, now=now, stale_after_seconds=stale_after_seconds)
            if status_name in {"failed", "crashed"}:
                members_by_state["failed"].append(member)
                warnings.append(f"{member['member_id']} failed")
            elif is_stale:
                members_by_state["stale"].append(member)
                warnings.append(f"{member['member_id']} heartbeat is stale")
            elif status_name in {"working", "processing_messages", "message_pending"}:
                members_by_state["working"].append(member)
            elif status_name in {"stopped", "stopping"}:
                members_by_state["stopped"].append(member)
            else:
                members_by_state["idle"].append(member)

        lead_inbox = (
            [message.to_dict() for message in self.read_lead_inbox(team_id, ack=ack_inbox)]
            if include_inbox
            else []
        )
        for task in tasks_by_status.get("blocked", []):
            reason = task.get("error") or "no error recorded"
            warnings.append(f"{task['task_id']} blocked: {reason}")

        task_counts = {status_name: len(items) for status_name, items in tasks_by_status.items()}
        member_counts = {state_name: len(items) for state_name, items in members_by_state.items()}
        terminal = self._snapshot_terminal(task_counts=task_counts, team_status=status["team"]["status"])
        suggested_next = self._suggest_next(task_counts=task_counts, warnings=warnings, terminal=terminal)
        return {
            "team": status["team"],
            "state": status["state"],
            "summary": {
                "tasks": task_counts,
                "members": member_counts,
                "lead_inbox_unread": len(lead_inbox),
            },
            "tasks_by_status": tasks_by_status,
            "members_by_state": members_by_state,
            "lead_inbox": lead_inbox,
            "warnings": warnings,
            "terminal": terminal,
            "suggested_next": suggested_next,
            "updated_at": utc_now_iso(),
        }

    @staticmethod
    def _is_stale(
        timestamp: str | None,
        *,
        now: datetime,
        stale_after_seconds: int,
    ) -> bool:
        if not timestamp:
            return False
        try:
            parsed = datetime.fromisoformat(timestamp)
        except ValueError:
            return False
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return (now - parsed).total_seconds() > stale_after_seconds

    @staticmethod
    def _snapshot_terminal(*, task_counts: dict[str, int], team_status: str) -> bool:
        if team_status in {"shutdown", "completed", "failed"}:
            return True
        total = sum(task_counts.values())
        return total > 0 and task_counts.get("completed", 0) == total

    @staticmethod
    def _suggest_next(
        *,
        task_counts: dict[str, int],
        warnings: list[str],
        terminal: bool,
    ) -> list[str]:
        if terminal:
            return ["summarize results", "shutdown team when appropriate"]
        if task_counts.get("blocked", 0):
            return ["inspect blocked tasks", "update task or send a message"]
        if warnings:
            return ["inspect warnings", "send status check if needed"]
        if task_counts.get("in_progress", 0):
            return ["wait for teammate progress", "read lead inbox"]
        if task_counts.get("pending", 0):
            return ["wait for teammates to claim pending tasks"]
        return ["create tasks or plan the team work"]

    @staticmethod
    def _slug_from_goal(goal: str) -> str:
        words = [
            "".join(ch if ch.isalnum() else "-" for ch in word.lower()).strip("-")
            for word in goal.split()
        ]
        slug = "-".join(word for word in words if word)[:24].strip("-")
        return slug or "team"

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
