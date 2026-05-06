"""Persistent team metadata store."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from agent_core.team.atomic_io import atomic_write_json, read_json
from agent_core.team.file_lock import FileLock
from agent_core.team.models import TeamConfig, TeamMember, utc_now_iso


class TeamStore:
    """Filesystem store for team configs, members, heartbeats, and events."""

    def __init__(self, teams_root: Path):
        self.teams_root = teams_root.expanduser().resolve()
        self.teams_root.mkdir(parents=True, exist_ok=True)

    def new_team_id(self, name: str) -> str:
        normalized = "".join(ch if ch.isalnum() else "-" for ch in name.lower()).strip("-")
        prefix = normalized[:24] or "team"
        return f"{prefix}-{uuid.uuid4().hex[:8]}"

    def team_dir(self, team_id: str) -> Path:
        return self.teams_root / team_id

    def create_team(self, *, name: str, goal: str, workspace_root: Path) -> TeamConfig:
        team_id = self.new_team_id(name or goal)
        team_dir = self.team_dir(team_id)
        for path in (
            team_dir / "mailboxes" / "lead" / "inbox",
            team_dir / "mailboxes" / "lead" / "read",
            team_dir / "locks",
            team_dir / "heartbeats",
            team_dir / "logs",
            team_dir / "sessions",
        ):
            path.mkdir(parents=True, exist_ok=True)
        config = TeamConfig(
            team_id=team_id,
            name=name or team_id,
            goal=goal,
            workspace_root=str(workspace_root.resolve()),
        )
        atomic_write_json(team_dir / "config.json", config.to_dict())
        atomic_write_json(team_dir / "members.json", {"members": []})
        atomic_write_json(team_dir / "tasks.json", {"version": 1, "tasks": []})
        self.append_event(team_id, "team_created", actor="lead", payload=config.to_dict())
        return config

    def load_config(self, team_id: str) -> TeamConfig:
        payload = read_json(self.team_dir(team_id) / "config.json", None)
        if payload is None:
            raise FileNotFoundError(f"Team not found: {team_id}")
        return TeamConfig.from_dict(payload)

    def list_teams(self) -> list[TeamConfig]:
        teams: list[TeamConfig] = []
        for config_path in sorted(self.teams_root.glob("*/config.json")):
            try:
                teams.append(TeamConfig.from_dict(read_json(config_path, {})))
            except Exception:
                continue
        return teams

    def update_config_status(self, team_id: str, status: str) -> TeamConfig:
        config = self.load_config(team_id)
        config.status = status
        config.updated_at = utc_now_iso()
        atomic_write_json(self.team_dir(team_id) / "config.json", config.to_dict())
        self.append_event(team_id, "team_status_updated", actor="lead", payload={"status": status})
        return config

    def list_members(self, team_id: str) -> list[TeamMember]:
        payload = read_json(self.team_dir(team_id) / "members.json", {"members": []})
        return [TeamMember.from_dict(item) for item in payload.get("members", [])]

    def upsert_member(self, team_id: str, member: TeamMember) -> TeamMember:
        team_dir = self.team_dir(team_id)
        with FileLock(team_dir / "locks" / "members.lock", owner="lead"):
            members = self.list_members(team_id)
            replaced = False
            for index, existing in enumerate(members):
                if existing.member_id == member.member_id:
                    members[index] = member
                    replaced = True
                    break
            if not replaced:
                members.append(member)
            atomic_write_json(
                team_dir / "members.json",
                {"members": [item.to_dict() for item in members]},
            )
        self.ensure_mailbox(team_id, member.member_id)
        self.append_event(
            team_id,
            "member_upserted",
            actor="lead",
            payload=member.to_dict(),
        )
        return member

    def mark_member_status(self, team_id: str, member_id: str, status: str) -> None:
        members = self.list_members(team_id)
        for member in members:
            if member.member_id == member_id:
                member.status = status
                member.updated_at = utc_now_iso()
                self.upsert_member(team_id, member)
                return

    def ensure_mailbox(self, team_id: str, member_id: str) -> None:
        base = self.team_dir(team_id) / "mailboxes" / member_id
        (base / "inbox").mkdir(parents=True, exist_ok=True)
        (base / "read").mkdir(parents=True, exist_ok=True)

    def write_heartbeat(self, team_id: str, member_id: str, payload: dict) -> None:
        atomic_write_json(self.team_dir(team_id) / "heartbeats" / f"{member_id}.json", payload)

    def read_heartbeat(self, team_id: str, member_id: str) -> dict | None:
        path = self.team_dir(team_id) / "heartbeats" / f"{member_id}.json"
        return read_json(path, None)

    def append_event(self, team_id: str, event_type: str, *, actor: str, payload: dict) -> None:
        event = {
            "type": event_type,
            "actor": actor,
            "created_at": utc_now_iso(),
            "payload": payload,
        }
        event_path = self.team_dir(team_id) / "events.jsonl"
        event_path.parent.mkdir(parents=True, exist_ok=True)
        with event_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
