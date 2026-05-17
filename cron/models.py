"""Data models for scheduled cron jobs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

CronScheduleKind = Literal["once", "interval", "cron"]
CronSource = Literal["local", "remote"]
CronDelivery = Literal["local", "remote"]
CronExecutionMode = Literal["enqueue_current_session", "fresh_agent_background"]
CronJobState = Literal["scheduled", "completed"]
CronRunStatus = Literal[
    "claimed",
    "enqueued",
    "success",
    "failed",
    "missed",
    "delivery_failed",
]
CronDeliveryStatus = Literal["local_only", "queued", "delivered", "failed", "not_attempted"]


def datetime_to_iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        raise ValueError("datetime values must be timezone-aware")
    return value.isoformat()


def datetime_from_iso(value: str | None) -> datetime | None:
    if value is None:
        return None
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        raise ValueError(f"datetime value must include timezone offset: {value}")
    return parsed


@dataclass(slots=True)
class CronSchedule:
    kind: CronScheduleKind
    display: str
    timezone: str
    run_at: datetime | None = None
    seconds: int | None = None
    expr: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "kind": self.kind,
            "display": self.display,
            "timezone": self.timezone,
        }
        if self.run_at is not None:
            payload["run_at"] = datetime_to_iso(self.run_at)
        if self.seconds is not None:
            payload["seconds"] = self.seconds
        if self.expr is not None:
            payload["expr"] = self.expr
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CronSchedule":
        return cls(
            kind=payload["kind"],
            display=str(payload.get("display") or ""),
            timezone=str(payload.get("timezone") or ""),
            run_at=datetime_from_iso(payload.get("run_at")),
            seconds=int(payload["seconds"]) if payload.get("seconds") is not None else None,
            expr=str(payload["expr"]) if payload.get("expr") is not None else None,
        )


@dataclass(slots=True)
class CronRepeat:
    times: int | None = None
    completed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"times": self.times, "completed": self.completed}

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "CronRepeat":
        payload = payload or {}
        raw_times = payload.get("times")
        return cls(
            times=int(raw_times) if raw_times is not None else None,
            completed=int(payload.get("completed") or 0),
        )


@dataclass(slots=True)
class CronExecution:
    mode: CronExecutionMode
    workspace_root: str
    session_binding_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "workspace_root": self.workspace_root,
            "session_binding_id": self.session_binding_id,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CronExecution":
        return cls(
            mode=payload["mode"],
            workspace_root=str(payload["workspace_root"]),
            session_binding_id=str(payload["session_binding_id"]),
        )


@dataclass(slots=True)
class CronRun:
    run_id: str
    scheduled_at: datetime
    claimed_at: datetime
    execution_mode: CronExecutionMode
    started_at: datetime | None = None
    finished_at: datetime | None = None
    status: CronRunStatus = "claimed"
    archive_path: str | None = None
    bridge_request_id: str | None = None
    delivery_status: CronDeliveryStatus = "not_attempted"
    delivery_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "scheduled_at": datetime_to_iso(self.scheduled_at),
            "claimed_at": datetime_to_iso(self.claimed_at),
            "started_at": datetime_to_iso(self.started_at),
            "finished_at": datetime_to_iso(self.finished_at),
            "status": self.status,
            "execution_mode": self.execution_mode,
            "archive_path": self.archive_path,
            "bridge_request_id": self.bridge_request_id,
            "delivery_status": self.delivery_status,
            "delivery_error": self.delivery_error,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "CronRun | None":
        if not payload:
            return None
        return cls(
            run_id=str(payload["run_id"]),
            scheduled_at=datetime_from_iso(payload["scheduled_at"]),
            claimed_at=datetime_from_iso(payload["claimed_at"]),
            started_at=datetime_from_iso(payload.get("started_at")),
            finished_at=datetime_from_iso(payload.get("finished_at")),
            status=payload.get("status") or "claimed",
            execution_mode=payload.get("execution_mode") or "enqueue_current_session",
            archive_path=payload.get("archive_path"),
            bridge_request_id=payload.get("bridge_request_id"),
            delivery_status=payload.get("delivery_status") or "not_attempted",
            delivery_error=payload.get("delivery_error"),
        )


@dataclass(slots=True)
class CronJob:
    id: str
    name: str
    prompt: str
    schedule: CronSchedule
    schedule_display: str
    repeat: CronRepeat
    source: CronSource
    delivery: CronDelivery
    execution: CronExecution
    enabled: bool
    state: CronJobState
    next_run_at: datetime
    created_at: datetime
    updated_at: datetime
    remote_completion_context: dict[str, Any] | None = None
    model: str | None = None
    provider: str | None = None
    base_url: str | None = None
    enabled_toolsets: list[str] | None = None
    last_run_at: datetime | None = None
    last_status: CronRunStatus | None = None
    last_error: str | None = None
    last_delivery_error: str | None = None
    last_run: CronRun | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "prompt": self.prompt,
            "schedule": self.schedule.to_dict(),
            "schedule_display": self.schedule_display,
            "repeat": self.repeat.to_dict(),
            "source": self.source,
            "delivery": self.delivery,
            "execution": self.execution.to_dict(),
            "remote_completion_context": self.remote_completion_context,
            "model": self.model,
            "provider": self.provider,
            "base_url": self.base_url,
            "enabled_toolsets": self.enabled_toolsets,
            "enabled": self.enabled,
            "state": self.state,
            "next_run_at": datetime_to_iso(self.next_run_at),
            "last_run_at": datetime_to_iso(self.last_run_at),
            "last_status": self.last_status,
            "last_error": self.last_error,
            "last_delivery_error": self.last_delivery_error,
            "last_run": self.last_run.to_dict() if self.last_run is not None else None,
            "created_at": datetime_to_iso(self.created_at),
            "updated_at": datetime_to_iso(self.updated_at),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CronJob":
        return cls(
            id=str(payload["id"]),
            name=str(payload["name"]),
            prompt=str(payload["prompt"]),
            schedule=CronSchedule.from_dict(payload["schedule"]),
            schedule_display=str(payload.get("schedule_display") or payload["schedule"].get("display") or ""),
            repeat=CronRepeat.from_dict(payload.get("repeat")),
            source=payload.get("source") or "local",
            delivery=payload.get("delivery") or "local",
            execution=CronExecution.from_dict(payload["execution"]),
            remote_completion_context=payload.get("remote_completion_context"),
            model=payload.get("model"),
            provider=payload.get("provider"),
            base_url=payload.get("base_url"),
            enabled_toolsets=payload.get("enabled_toolsets"),
            enabled=bool(payload.get("enabled", True)),
            state=payload.get("state") or "scheduled",
            next_run_at=datetime_from_iso(payload["next_run_at"]),
            last_run_at=datetime_from_iso(payload.get("last_run_at")),
            last_status=payload.get("last_status"),
            last_error=payload.get("last_error"),
            last_delivery_error=payload.get("last_delivery_error"),
            last_run=CronRun.from_dict(payload.get("last_run")),
            created_at=datetime_from_iso(payload["created_at"]),
            updated_at=datetime_from_iso(payload["updated_at"]),
        )


@dataclass(slots=True)
class CronHostContext:
    source: CronSource
    workspace_root: Path
    session_binding_id: str
    default_delivery: CronDelivery = "local"
    worker_id: str | None = None
    gateway_client: Any | None = None
    fresh_agent_runner: Any | None = None


@dataclass(slots=True)
class CronDeliveryResult:
    ok: bool
    status: CronDeliveryStatus
    error: str | None = None


@dataclass(slots=True)
class CronTickResult:
    claimed: int = 0
    executed: int = 0
    missed: int = 0
    skipped_locked: bool = False
    errors: list[str] = field(default_factory=list)
