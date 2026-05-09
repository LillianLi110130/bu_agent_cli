"""Dataclasses for filesystem-backed agent teams."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from agent_core.team.protocol import normalize_message_type


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _list(value: Any) -> list:
    return value if isinstance(value, list) else []


@dataclass(slots=True)
class TeamConfig:
    team_id: str
    name: str
    goal: str
    workspace_root: str
    lead_member_id: str = "lead"
    status: str = "running"
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    version: int = 1

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TeamConfig":
        return cls(
            team_id=str(payload["team_id"]),
            name=str(payload.get("name") or payload["team_id"]),
            goal=str(payload.get("goal") or ""),
            workspace_root=str(payload.get("workspace_root") or ""),
            lead_member_id=str(payload.get("lead_member_id") or "lead"),
            status=str(payload.get("status") or "running"),
            created_at=str(payload.get("created_at") or utc_now_iso()),
            updated_at=str(payload.get("updated_at") or utc_now_iso()),
            version=int(payload.get("version") or 1),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TeamState:
    team_id: str
    goal: str
    active: bool = True
    phase: str = "created"
    fix_loop_count: int = 0
    max_fix_loops: int = 3
    stage_history: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    version: int = 1

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TeamState":
        return cls(
            team_id=str(payload["team_id"]),
            goal=str(payload.get("goal") or ""),
            active=bool(payload.get("active", True)),
            phase=str(payload.get("phase") or "created"),
            fix_loop_count=int(payload.get("fix_loop_count") or 0),
            max_fix_loops=int(payload.get("max_fix_loops") or 3),
            stage_history=[str(item) for item in _list(payload.get("stage_history"))],
            created_at=str(payload.get("created_at") or utc_now_iso()),
            updated_at=str(payload.get("updated_at") or utc_now_iso()),
            version=int(payload.get("version") or 1),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TeamMember:
    member_id: str
    agent_type: str
    status: str = "running"
    pid: int | None = None
    started_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    last_heartbeat_at: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TeamMember":
        pid = payload.get("pid")
        return cls(
            member_id=str(payload["member_id"]),
            agent_type=str(payload.get("agent_type") or "general-purpose"),
            status=str(payload.get("status") or "running"),
            pid=int(pid) if pid is not None else None,
            started_at=str(payload.get("started_at") or utc_now_iso()),
            updated_at=str(payload.get("updated_at") or utc_now_iso()),
            last_heartbeat_at=payload.get("last_heartbeat_at"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TeamTask:
    task_id: str
    title: str
    description: str
    status: str = "pending"
    assigned_to: str | None = None
    claimed_by: str | None = None
    depends_on: list[str] = field(default_factory=list)
    write_scope: list[str] = field(default_factory=list)
    result: str | None = None
    error: str | None = None
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    claimed_at: str | None = None
    completed_at: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TeamTask":
        return cls(
            task_id=str(payload["task_id"]),
            title=str(payload.get("title") or payload["task_id"]),
            description=str(payload.get("description") or ""),
            status=str(payload.get("status") or "pending"),
            assigned_to=payload.get("assigned_to"),
            claimed_by=payload.get("claimed_by"),
            depends_on=[str(item) for item in _list(payload.get("depends_on"))],
            write_scope=[str(item) for item in _list(payload.get("write_scope"))],
            result=payload.get("result"),
            error=payload.get("error"),
            created_at=str(payload.get("created_at") or utc_now_iso()),
            updated_at=str(payload.get("updated_at") or utc_now_iso()),
            claimed_at=payload.get("claimed_at"),
            completed_at=payload.get("completed_at"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TeamMessage:
    message_id: str
    team_id: str
    sender: str
    recipient: str
    type: str
    body: str
    metadata: dict[str, Any] = field(default_factory=dict)
    reply_to: str | None = None
    created_at: str = field(default_factory=utc_now_iso)
    read_at: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TeamMessage":
        metadata = payload.get("metadata")
        return cls(
            message_id=str(payload["message_id"]),
            team_id=str(payload["team_id"]),
            sender=str(payload.get("sender") or "unknown"),
            recipient=str(payload.get("recipient") or "lead"),
            type=normalize_message_type(str(payload.get("type") or "message")),
            body=str(payload.get("body") or ""),
            metadata=metadata if isinstance(metadata, dict) else {},
            reply_to=payload.get("reply_to"),
            created_at=str(payload.get("created_at") or utc_now_iso()),
            read_at=payload.get("read_at"),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["type"] = normalize_message_type(str(payload.get("type") or "message"))
        return payload
