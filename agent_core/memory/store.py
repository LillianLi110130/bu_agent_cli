from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path

from agent_core.runtime_paths import tg_agent_home

MEMORY_SEPARATOR = "\n§\n"
USER_TARGET = "user"
MEMORY_TARGET = "memory"
VALID_TARGETS = {USER_TARGET, MEMORY_TARGET}

_UNSAFE_SUBSTRINGS = (
    "ignore previous instructions",
    "you are now",
    "system prompt",
    "developer message",
    "do not tell the user",
    ".env",
    ".netrc",
    ".pgpass",
    ".npmrc",
    ".pypirc",
    "credentials",
    "authorized_keys",
    "~/.ssh",
)


class MemoryStoreError(ValueError):
    """Raised when a memory file operation is invalid or unsafe."""


@dataclass(frozen=True, slots=True)
class MemorySnapshot:
    user_entries: list[str]
    memory_entries: list[str]


@dataclass(frozen=True, slots=True)
class MemoryStoreResult:
    action: str
    target: str
    text: str
    path: Path


class MemoryStore:
    """Small file-backed store for CLI persistent memory."""

    def __init__(
        self,
        base_dir: Path | None = None,
        *,
        user_char_limit: int = 1375,
        memory_char_limit: int = 2200,
    ) -> None:
        self.base_dir = base_dir or tg_agent_home() / "memories"
        self.user_char_limit = user_char_limit
        self.memory_char_limit = memory_char_limit

    @property
    def user_path(self) -> Path:
        return self.base_dir / "USER.md"

    @property
    def memory_path(self) -> Path:
        return self.base_dir / "MEMORY.md"

    def load_from_disk(self) -> MemorySnapshot:
        return MemorySnapshot(
            user_entries=self.list(USER_TARGET),
            memory_entries=self.list(MEMORY_TARGET),
        )

    def render_context(self, snapshot: MemorySnapshot) -> str:
        if not snapshot.user_entries and not snapshot.memory_entries:
            return ""

        lines = [
            "## Persistent Memory Snapshot",
            "The following memory was loaded from ~/.tg_agent/memories at CLI session start.",
            "Treat this as a frozen snapshot; new writes apply to future CLI sessions.",
        ]
        if snapshot.user_entries:
            lines.extend(["", "### User Memory"])
            lines.extend(f"- {entry}" for entry in snapshot.user_entries)
        if snapshot.memory_entries:
            lines.extend(["", "### Agent Memory"])
            lines.extend(f"- {entry}" for entry in snapshot.memory_entries)
        return "\n".join(lines)

    def list(self, target: str) -> list[str]:
        path = self._path_for(target)
        if not path.exists():
            return []
        return self._parse_entries(path.read_text(encoding="utf-8"))

    def add(self, target: str, text: str) -> MemoryStoreResult:
        target = self._normalize_target(target)
        text = self._normalize_text(text, field_name="text")
        self._validate_safe(text)

        entries = self.list(target)
        entries.append(text)
        self._write_entries(target, entries)
        return MemoryStoreResult(
            action="added",
            target=target,
            text=text,
            path=self._path_for(target),
        )

    def replace(self, target: str, old_text: str, new_text: str) -> MemoryStoreResult:
        target = self._normalize_target(target)
        old_text = self._normalize_text(old_text, field_name="old_text")
        new_text = self._normalize_text(new_text, field_name="text")
        self._validate_safe(new_text)

        entries = self.list(target)
        index = self._find_unique_entry(entries, old_text)
        entries[index] = new_text
        self._write_entries(target, entries)
        return MemoryStoreResult(
            action="replaced",
            target=target,
            text=new_text,
            path=self._path_for(target),
        )

    def remove(self, target: str, old_text: str) -> MemoryStoreResult:
        target = self._normalize_target(target)
        old_text = self._normalize_text(old_text, field_name="old_text")

        entries = self.list(target)
        index = self._find_unique_entry(entries, old_text)
        removed = entries.pop(index)
        self._write_entries(target, entries)
        return MemoryStoreResult(
            action="removed",
            target=target,
            text=removed,
            path=self._path_for(target),
        )

    def _path_for(self, target: str) -> Path:
        target = self._normalize_target(target)
        if target == USER_TARGET:
            return self.user_path
        return self.memory_path

    @staticmethod
    def _normalize_target(target: str) -> str:
        normalized = (target or "").strip().lower()
        if normalized not in VALID_TARGETS:
            raise MemoryStoreError("target must be 'user' or 'memory'.")
        return normalized

    @staticmethod
    def _normalize_text(text: str | None, *, field_name: str) -> str:
        normalized = (text or "").strip()
        if not normalized:
            raise MemoryStoreError(f"{field_name} is required.")
        if MEMORY_SEPARATOR.strip() in normalized:
            raise MemoryStoreError("memory text may not contain the entry separator.")
        return normalized

    @staticmethod
    def _parse_entries(content: str) -> list[str]:
        return [entry.strip() for entry in content.split(MEMORY_SEPARATOR) if entry.strip()]

    @staticmethod
    def _find_unique_entry(entries: list[str], needle: str) -> int:
        matches = [index for index, entry in enumerate(entries) if needle in entry]
        if not matches:
            raise MemoryStoreError("old_text did not match any memory entry.")
        if len(matches) > 1:
            raise MemoryStoreError("old_text matched multiple memory entries.")
        return matches[0]

    def _write_entries(self, target: str, entries: list[str]) -> None:
        path = self._path_for(target)
        content = MEMORY_SEPARATOR.join(entries)
        if content:
            content += "\n"
        self._validate_limit(target, content)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f"{path.name}.tmp")
        tmp_path.write_text(content, encoding="utf-8")
        tmp_path.replace(path)

    def _validate_limit(self, target: str, content: str) -> None:
        limit = self.user_char_limit if target == USER_TARGET else self.memory_char_limit
        if len(content) > limit:
            raise MemoryStoreError(
                f"{target} memory exceeds the {limit} character limit."
            )

    @staticmethod
    def _validate_safe(text: str) -> None:
        lowered = text.lower()
        for unsafe in _UNSAFE_SUBSTRINGS:
            if unsafe in lowered:
                raise MemoryStoreError(f"unsafe memory content rejected: {unsafe}")

        for char in text:
            category = unicodedata.category(char)
            if category == "Cf" or (category == "Cc" and char not in "\n\r\t"):
                raise MemoryStoreError(
                    "unsafe memory content rejected: invisible control character"
                )
