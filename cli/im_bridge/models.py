"""Data models for the file-backed IM bridge."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal


InputKind = Literal["text", "slash", "skill", "image"]


def utc_now() -> datetime:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc)


def to_iso8601(value: datetime) -> str:
    """Serialize a UTC datetime to an ISO-8601 string."""
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def from_iso8601(value: str) -> datetime:
    """Parse an ISO-8601 string into a UTC datetime."""
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value).astimezone(timezone.utc)


def classify_input_kind(content: str) -> InputKind:
    """Infer the logical input kind from the raw text content."""
    stripped = content.lstrip()
    if stripped.startswith('/'):
        return "slash"
    if stripped.startswith('@\"') or stripped.startswith("@'"):
        return "image"
    if stripped.startswith('@'):
        return "skill"
    return "text"


@dataclass
class BridgeRequest:
    """One queued request in the local file bridge."""

    version: int
    request_id: str
    seq: int
    source: Literal["local", "remote"]
    source_meta: dict[str, Any] = field(default_factory=dict)
    content: str = ""
    input_kind: InputKind = "text"
    content_type: str = "text"
    enqueue_time: datetime = field(default_factory=utc_now)
    remote_response_required: bool = False
    status: Literal["pending", "running"] = "pending"
    started_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the request for JSON storage."""
        payload = asdict(self)
        payload["enqueue_time"] = to_iso8601(self.enqueue_time)
        payload["started_at"] = to_iso8601(self.started_at) if self.started_at else None
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BridgeRequest":
        """Deserialize a request from JSON payload."""
        return cls(
            version=int(payload["version"]),
            request_id=str(payload["request_id"]),
            seq=int(payload["seq"]),
            source=str(payload["source"]),
            source_meta=dict(payload.get("source_meta") or {}),
            content=str(payload.get("content", "")),
            input_kind=str(payload.get("input_kind", "text")),
            content_type=str(payload.get("content_type", "text")),
            enqueue_time=from_iso8601(str(payload["enqueue_time"])),
            remote_response_required=bool(payload.get("remote_response_required", False)),
            status=str(payload.get("status", "pending")),
            started_at=(
                from_iso8601(str(payload["started_at"])) if payload.get("started_at") else None
            ),
        )


@dataclass
class BridgeResult:
    """One completed or failed result in the local file bridge."""

    version: int
    request_id: str
    seq: int
    source: Literal["local", "remote"]
    final_status: Literal["completed", "failed"]
    input_content: str = ""
    input_kind: InputKind = "text"
    final_content: str = ""
    error_code: str | None = None
    error_message: str | None = None
    started_at: datetime | None = None
    finished_at: datetime = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result for JSON storage."""
        payload = asdict(self)
        payload["started_at"] = to_iso8601(self.started_at) if self.started_at else None
        payload["finished_at"] = to_iso8601(self.finished_at)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BridgeResult":
        """Deserialize a result from JSON payload."""
        return cls(
            version=int(payload["version"]),
            request_id=str(payload["request_id"]),
            seq=int(payload["seq"]),
            source=str(payload["source"]),
            final_status=str(payload["final_status"]),
            input_content=str(payload.get("input_content", "")),
            input_kind=str(payload.get("input_kind", "text")),
            final_content=str(payload.get("final_content", "")),
            error_code=(
                str(payload["error_code"]) if payload.get("error_code") is not None else None
            ),
            error_message=(
                str(payload["error_message"]) if payload.get("error_message") is not None else None
            ),
            started_at=(
                from_iso8601(str(payload["started_at"])) if payload.get("started_at") else None
            ),
            finished_at=from_iso8601(str(payload["finished_at"])),
        )
