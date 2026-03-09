from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


@dataclass(slots=True)
class RalphPaths:
    workspace_root: Path
    spec_name: str
    spec_dir: Path
    requirement_dir: Path
    plan_dir: Path
    plan_file: Path
    implement_dir: Path
    log_dir: Path


@dataclass(slots=True)
class RalphCommandResult:
    success: bool
    message: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RalphRunRecord:
    run_id: str
    spec_name: str
    status: str
    pid: int | None
    command: list[str]
    cwd: str
    plan_file: str
    log_dir: str
    stdout_log: str
    stderr_log: str
    created_at: str
    started_at: str
    ended_at: str | None = None
    exit_code: int | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RalphRunRecord":
        return cls(
            run_id=str(data["run_id"]),
            spec_name=str(data.get("spec_name", "")),
            status=str(data.get("status", "unknown")),
            pid=data.get("pid"),
            command=list(data.get("command", [])),
            cwd=str(data.get("cwd", "")),
            plan_file=str(data.get("plan_file", "")),
            log_dir=str(data.get("log_dir", "")),
            stdout_log=str(data.get("stdout_log", "")),
            stderr_log=str(data.get("stderr_log", "")),
            created_at=str(data.get("created_at", now_iso())),
            started_at=str(data.get("started_at", now_iso())),
            ended_at=data.get("ended_at"),
            exit_code=data.get("exit_code"),
            error=data.get("error"),
        )
