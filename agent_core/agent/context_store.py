"""Filesystem-backed stores for rollout-local context state."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence
from uuid import uuid4

from agent_core.agent.compaction.models import CompactionResult, CompactionWorkingState
from agent_core.llm.messages import BaseMessage, ToolMessage


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _sanitize_segment(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    return cleaned.strip("-") or uuid4().hex[:8]


def _dedupe_preserve_order(values: Sequence[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = str(value).strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        deduped.append(cleaned)
    return deduped


def _serialize_message(message: BaseMessage) -> dict[str, object]:
    try:
        return message.model_dump(mode="json")
    except Exception:
        return {
            "role": getattr(message, "role", "unknown"),
            "content": str(getattr(message, "content", "")),
        }


def _serialize_tool_content(content: object) -> object:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        serialized_parts: list[object] = []
        for part in content:
            if hasattr(part, "model_dump"):
                serialized_parts.append(part.model_dump(mode="json"))
            else:
                serialized_parts.append(part)
        return serialized_parts
    return str(content)


def _estimate_content_size(content: object) -> int:
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        total = 0
        for part in content:
            if hasattr(part, "model_dump_json"):
                total += len(part.model_dump_json())
            else:
                total += len(str(part))
        return total
    return len(str(content))


@dataclass(slots=True)
class CheckpointRecord:
    reference: str
    path: Path
    message_count: int


class ArtifactStore:
    """Persist recoverable large artifacts under one rollout directory."""

    DEFAULT_TOOL_OUTPUT_CHAR_THRESHOLD = 1200

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def save_json(self, category: str, name: str, payload: dict[str, object]) -> Path:
        target_dir = self.root_dir / _sanitize_segment(category)
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{_sanitize_segment(name)}.json"
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return target

    def save_tool_message(
        self,
        message: ToolMessage,
        *,
        force: bool = False,
        char_threshold: int | None = None,
    ) -> Path | None:
        threshold = (
            self.DEFAULT_TOOL_OUTPUT_CHAR_THRESHOLD
            if char_threshold is None
            else max(0, int(char_threshold))
        )
        if not force and _estimate_content_size(message.content) < threshold:
            return None

        payload = {
            "tool_call_id": message.tool_call_id,
            "tool_name": message.tool_name,
            "is_error": message.is_error,
            "ephemeral": message.ephemeral,
            "created_at": _now_iso(),
            "content": _serialize_tool_content(message.content),
        }
        return self.save_json("tool", message.tool_call_id, payload)

    def save_image_detail(self, detail_text: str, *, source_hint: str = "") -> Path:
        artifact_id = f"image-{uuid4().hex[:12]}"
        payload = {
            "artifact_id": artifact_id,
            "source_hint": source_hint,
            "created_at": _now_iso(),
            "detail": detail_text,
        }
        return self.save_json("image", artifact_id, payload)


class CheckpointStore:
    """Persist compacted-away message segments before they are replaced."""

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def save_messages(self, messages: Sequence[BaseMessage]) -> CheckpointRecord:
        if not messages:
            raise ValueError("Cannot create a checkpoint for an empty message segment.")

        next_index = self._next_index()
        reference = f"checkpoint://{next_index:04d}"
        target = self.root_dir / f"{next_index:04d}.jsonl"
        lines = [
            json.dumps(
                {
                    "type": "meta",
                    "ref": reference,
                    "created_at": _now_iso(),
                    "message_count": len(messages),
                },
                ensure_ascii=False,
            )
        ]
        lines.extend(
            json.dumps(
                {
                    "type": "message",
                    "index": index,
                    "message": _serialize_message(message),
                },
                ensure_ascii=False,
            )
            for index, message in enumerate(messages)
        )
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return CheckpointRecord(reference=reference, path=target, message_count=len(messages))

    def _next_index(self) -> int:
        indices = []
        for path in self.root_dir.glob("*.jsonl"):
            try:
                indices.append(int(path.stem))
            except ValueError:
                continue
        return (max(indices) + 1) if indices else 1


class WorkingStateStore:
    """Persist the current structured task state for one rollout session."""

    def __init__(self, path: Path, *, session_id: str) -> None:
        self.path = path
        self.session_id = session_id
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def ensure_initialized(self) -> None:
        if self.path.exists():
            return
        self.path.write_text(
            json.dumps(self._empty_payload(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def write_result(
        self,
        result: CompactionResult,
        *,
        artifact_refs: Sequence[str] = (),
    ) -> dict[str, object]:
        existing = self._load_payload()
        working_state = result.working_state or CompactionWorkingState()

        payload: dict[str, object] = {
            "session_id": self.session_id,
            "goal": working_state.user_goal or existing.get("goal", ""),
            "constraints": _dedupe_preserve_order(
                [
                    *self._read_list(existing, "constraints"),
                    *working_state.user_constraints,
                ]
            ),
            "confirmed_facts": _dedupe_preserve_order(
                [
                    *self._read_list(existing, "confirmed_facts"),
                    *working_state.confirmed_conclusions,
                ]
            ),
            "decisions": self._read_list(existing, "decisions"),
            "failed_attempts": _dedupe_preserve_order(
                [
                    *self._read_list(existing, "failed_attempts"),
                    *working_state.failed_attempts,
                ]
            ),
            "files_checked": _dedupe_preserve_order(
                [
                    *self._read_list(existing, "files_checked"),
                    *working_state.files_reviewed,
                ]
            ),
            "files_modified": _dedupe_preserve_order(
                [
                    *self._read_list(existing, "files_modified"),
                    *working_state.files_modified,
                ]
            ),
            "open_questions": _dedupe_preserve_order(
                [
                    *self._read_list(existing, "open_questions"),
                    *working_state.recent_history_notes,
                ]
            ),
            "next_steps": _dedupe_preserve_order(
                [
                    *self._read_list(existing, "next_steps"),
                    *working_state.remaining_actions,
                ]
            ),
            "artifact_refs": _dedupe_preserve_order(
                [
                    *self._read_list(existing, "artifact_refs"),
                    *working_state.artifact_refs,
                    *artifact_refs,
                ]
            ),
            "checkpoint_refs": _dedupe_preserve_order(
                [
                    *self._read_list(existing, "checkpoint_refs"),
                    result.checkpoint_ref or "",
                ]
            ),
            "checkpoint_paths": _dedupe_preserve_order(
                [
                    *self._read_list(existing, "checkpoint_paths"),
                    result.checkpoint_path or "",
                ]
            ),
            "last_compacted_at": _now_iso() if result.compacted else existing.get("last_compacted_at"),
        }
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return payload

    def _load_payload(self) -> dict[str, object]:
        if not self.path.exists():
            return self._empty_payload()
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return self._empty_payload()
        if not isinstance(loaded, dict):
            return self._empty_payload()
        return loaded

    def _empty_payload(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "goal": "",
            "constraints": [],
            "confirmed_facts": [],
            "decisions": [],
            "failed_attempts": [],
            "files_checked": [],
            "files_modified": [],
            "open_questions": [],
            "next_steps": [],
            "artifact_refs": [],
            "checkpoint_refs": [],
            "checkpoint_paths": [],
            "last_compacted_at": None,
        }

    @staticmethod
    def _read_list(payload: dict[str, object], key: str) -> list[str]:
        raw = payload.get(key)
        if not isinstance(raw, list):
            return []
        return _dedupe_preserve_order([str(item) for item in raw])
