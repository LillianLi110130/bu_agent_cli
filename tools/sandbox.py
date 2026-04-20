"""Sandbox context for secure filesystem access."""

from __future__ import annotations

import asyncio
import fnmatch
import os
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tools.shell_tasks import ShellTaskManager


_IGNORE_FILE_NAME = ".tgagentignore"


class SecurityError(Exception):
    """Raised when a path escapes the sandbox."""


def _sanitize_path(path: str) -> str:
    """Fix Windows paths with accidental escape sequences like ``\b`` or ``\t``."""
    control_chars = {
        "\a": r"\a",
        "\b": r"\b",
        "\f": r"\f",
        "\n": r"\n",
        "\r": r"\r",
        "\t": r"\t",
        "\v": r"\v",
    }
    for control, escape in control_chars.items():
        if control in path:
            path = path.replace(control, escape)
    return path


def _normalize_pattern(pattern: str) -> str:
    """Normalize ignore patterns to POSIX-style separators."""
    normalized = pattern.strip().replace("\\", "/")
    normalized = re.sub(r"/+", "/", normalized)
    if normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _default_user_tgagent_dir() -> Path:
    """Return the user's fixed ~/.tg_agent directory."""
    home = os.environ.get("HOME") or os.environ.get("USERPROFILE")
    if home:
        return Path(home).expanduser() / ".tg_agent"
    return Path("~/.tg_agent").expanduser()


@dataclass(frozen=True)
class IgnoreRule:
    """One parsed ignore rule from ``.tgagentignore``."""

    source: str
    pattern: str
    negated: bool
    anchored: bool
    directory_only: bool
    basename_only: bool

    @classmethod
    def from_line(cls, line: str) -> "IgnoreRule | None":
        """Parse one non-empty line from ``.tgagentignore``."""
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            return None

        negated = stripped.startswith("!")
        if negated:
            stripped = stripped[1:].strip()
        if not stripped:
            return None

        directory_only = stripped.endswith("/")
        if directory_only:
            stripped = stripped[:-1]

        normalized = _normalize_pattern(stripped)
        if not normalized:
            return None

        anchored = normalized.startswith("/")
        if anchored:
            normalized = normalized[1:]
        if not normalized:
            return None

        basename_only = "/" not in normalized
        return cls(
            source=line.rstrip("\n"),
            pattern=normalized,
            negated=negated,
            anchored=anchored,
            directory_only=directory_only,
            basename_only=basename_only,
        )

    def matches(self, rel_path: str, name: str, *, is_dir: bool) -> bool:
        """Return True when this rule matches the given relative path context."""
        if self.directory_only and not is_dir:
            return False
        if self.basename_only:
            return fnmatch.fnmatchcase(name, self.pattern)
        if self.anchored:
            return fnmatch.fnmatchcase(rel_path, self.pattern)
        if fnmatch.fnmatchcase(rel_path, self.pattern):
            return True
        return any(
            fnmatch.fnmatchcase(partial, self.pattern) for partial in _iter_suffix_paths(rel_path)
        )


def _iter_suffix_paths(rel_path: str) -> list[str]:
    """Return suffix path variants for unanchored path rules."""
    parts = rel_path.split("/")
    return ["/".join(parts[index:]) for index in range(1, len(parts))]


@dataclass
class SandboxContext:
    """Sandboxed filesystem context with multiple allowed directories."""

    root_dir: Path
    working_dir: Path
    allowed_dirs: list[Path] = field(default_factory=list)
    ignore_rules: list[IgnoreRule] = field(default_factory=list)
    ignored_patterns: list[str] = field(default_factory=list)
    ignore_file_lines: list[str] = field(default_factory=list)
    runtime_managed_dirs: list[Path] = field(default_factory=list)
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    subagent_executor: Any | None = None
    current_agent: Any | None = None
    shell_task_manager: ShellTaskManager | None = None
    subagent_events: asyncio.Queue | None = field(default=None)

    def __post_init__(self) -> None:
        if self.subagent_events is None:
            self.subagent_events = asyncio.Queue()
        if self.shell_task_manager is None:
            self.shell_task_manager = ShellTaskManager(self.root_dir, self.session_id)

    @classmethod
    def create(cls, root_dir: Path | str | None = None) -> "SandboxContext":
        """Create a new sandbox context, defaulting to the current directory."""
        session_id = str(uuid.uuid4())[:8]
        root = Path.cwd().resolve() if root_dir is None else Path(root_dir).resolve()
        if not root.exists():
            root.mkdir(parents=True, exist_ok=True)
        allowed_dirs = [root]
        user_tgagent_dir = _default_user_tgagent_dir().resolve()
        allowed_dirs.append(user_tgagent_dir)
        ctx = cls(
            root_dir=root,
            working_dir=root,
            allowed_dirs=allowed_dirs,
            session_id=session_id,
        )
        ctx.load_ignore_rules()
        return ctx

    @property
    def ignore_file(self) -> Path:
        """Return the workspace ignore file."""
        return self.root_dir / _IGNORE_FILE_NAME

    def add_allowed_dir(self, path: Path | str) -> Path:
        """Add a directory to the allowed list."""
        resolved = Path(path).resolve()
        if not resolved.exists():
            raise SecurityError(f"Path does not exist: {resolved}")
        if not resolved.is_dir():
            raise SecurityError(f"Path is not a directory: {resolved}")

        resolved_str = str(resolved).lower()
        for allowed in list(self.allowed_dirs):
            allowed_str = str(allowed.resolve()).lower()
            if resolved_str == allowed_str:
                return resolved
            if self._is_same_or_parent(allowed, resolved):
                return resolved
            if self._is_same_or_parent(resolved, allowed):
                self.allowed_dirs.remove(allowed)

        self.allowed_dirs.append(resolved)
        self.allowed_dirs.sort(key=lambda item: str(item).lower())
        return resolved

    def add_ignore_pattern(self, pattern: str, *, persist: bool = True) -> str:
        """Add one ignore pattern line to ``.tgagentignore``."""
        cleaned = _normalize_pattern(pattern)
        if not cleaned:
            raise SecurityError("Ignore pattern cannot be empty.")
        if cleaned.startswith("#"):
            raise SecurityError("Ignore pattern cannot start with '#'.")

        if cleaned in self.ignored_patterns:
            return cleaned

        self.ignore_file_lines.append(cleaned)
        self._rebuild_ignore_rules()
        if persist:
            self.save_ignore_rules()
        return cleaned

    def add_runtime_managed_dir(self, path: Path | str) -> Path:
        """Register a runtime-managed directory such as rollout artifacts/checkpoints."""
        resolved = Path(path).resolve()
        if not resolved.exists():
            raise SecurityError(f"Path does not exist: {resolved}")
        if not resolved.is_dir():
            raise SecurityError(f"Path is not a directory: {resolved}")

        for existing in self.runtime_managed_dirs:
            if existing.resolve() == resolved:
                return resolved

        self.runtime_managed_dirs.append(resolved)
        self.runtime_managed_dirs.sort(key=lambda item: str(item).lower())
        return resolved

    def remove_ignore_pattern(self, pattern: str, *, persist: bool = True) -> str:
        """Remove one exact ignore pattern line from ``.tgagentignore``."""
        cleaned = _normalize_pattern(pattern)
        for index, raw_line in enumerate(list(self.ignore_file_lines)):
            rule = IgnoreRule.from_line(raw_line)
            if rule is None:
                continue
            if _normalize_pattern(rule.source) == cleaned:
                del self.ignore_file_lines[index]
                self._rebuild_ignore_rules()
                if persist:
                    self.save_ignore_rules()
                return cleaned
        raise SecurityError(f"Ignore pattern not found: {cleaned}")

    def load_ignore_rules(self) -> None:
        """Load ignore patterns from ``.tgagentignore`` when available."""
        if not self.ignore_file.exists():
            self.ignore_rules.clear()
            self.ignored_patterns.clear()
            self.ignore_file_lines.clear()
            return

        try:
            raw_lines = self.ignore_file.read_text(encoding="utf-8").splitlines()
        except OSError:
            self.ignore_rules.clear()
            self.ignored_patterns.clear()
            self.ignore_file_lines.clear()
            return

        self.ignore_file_lines = raw_lines
        self._rebuild_ignore_rules()

    def save_ignore_rules(self) -> None:
        """Persist ignore patterns to ``.tgagentignore``."""
        content = "\n".join(self.ignore_file_lines)
        if content:
            content += "\n"
        self.ignore_file.write_text(content, encoding="utf-8")
        self._rebuild_ignore_rules()

    def is_allowed(self, path: Path) -> bool:
        """Check whether a path is within any allowed directory."""
        resolved = path.resolve()
        return any(self._is_same_or_parent(allowed, resolved) for allowed in self.allowed_dirs)

    def is_ignored(self, path: Path) -> bool:
        """Check whether a path is blocked by ignore rules."""
        if not self.ignore_rules:
            return False

        relative_parts = self._relative_parts(path)
        if relative_parts is None:
            return False

        ignored = False
        path_exists = path.exists()
        path_is_dir = path_exists and path.is_dir()
        for index in range(1, len(relative_parts) + 1):
            rel_path = "/".join(relative_parts[:index])
            name = relative_parts[index - 1]
            is_dir = index < len(relative_parts) or path_is_dir
            for rule in self.ignore_rules:
                if rule.matches(rel_path, name, is_dir=is_dir):
                    ignored = not rule.negated
        return ignored

    def resolve_path(self, path: str | Path) -> Path:
        """Resolve and validate a path is within the sandbox and not ignored."""
        resolved = self._resolve_user_path(path)
        if not self.is_allowed(resolved):
            allowed_list = ", ".join(str(d) for d in self.allowed_dirs)
            raise SecurityError(
                f"Path escapes sandbox: {path} -> {resolved}\n"
                f"Allowed directories: {allowed_list}\n"
                f"Use /allow <path> to add a directory to the sandbox."
            )
        if self.is_ignored(resolved):
            raise SecurityError(
                f"Path is blocked by .tgagentignore: {path} -> {resolved}\n"
                f"Edit {self.ignore_file.name} to inspect or change active rules."
            )
        return resolved

    def is_runtime_artifact_path(self, path: Path | str) -> bool:
        """Return True when a resolved path is under one of the runtime-managed dirs."""
        resolved = Path(path).resolve()
        return any(self._is_same_or_parent(root, resolved) for root in self.runtime_managed_dirs)

    def _resolve_user_path(self, path: str | Path) -> Path:
        """Resolve a user path without applying allow/ignore checks."""
        path_str = _sanitize_path(str(path))
        normalized = os.path.normpath(path_str)
        path_obj = Path(normalized).expanduser()
        if path_obj.is_absolute():
            return path_obj.resolve()
        return (self.working_dir / path_obj).resolve()

    def _rebuild_ignore_rules(self) -> None:
        """Reparse ignore rules from the in-memory pattern list."""
        self.ignore_rules = []
        normalized_patterns: list[str] = []
        normalized_lines: list[str] = []
        for line in self.ignore_file_lines:
            normalized_lines.append(line.rstrip("\n"))
            rule = IgnoreRule.from_line(line)
            if rule is None:
                continue
            self.ignore_rules.append(rule)
            normalized_patterns.append(rule.source.strip())
        self.ignore_file_lines = normalized_lines
        self.ignored_patterns = normalized_patterns

    def _relative_parts(self, path: Path) -> tuple[str, ...] | None:
        """Return POSIX-style relative parts for a path under the workspace root."""
        try:
            relative = path.resolve().relative_to(self.root_dir.resolve())
        except ValueError:
            return None
        parts = tuple(part for part in relative.parts if part not in ("", "."))
        return parts or None

    @staticmethod
    def _is_same_or_parent(parent: Path, child: Path) -> bool:
        """Return True when ``child`` is the same path or is nested under ``parent``."""
        parent_resolved = parent.resolve()
        child_resolved = child.resolve()
        try:
            return child_resolved.is_relative_to(parent_resolved)
        except AttributeError:
            parent_str = str(parent_resolved).lower()
            child_str = str(child_resolved).lower()
            if child_str == parent_str:
                return True
            if child_str.startswith(parent_str):
                remainder = child_str[len(parent_str) :]
                return not remainder or remainder[0] in ("/", "\\")
            return False


def get_sandbox_context() -> SandboxContext:
    """Dependency injection marker. Override this in the agent."""
    raise RuntimeError("get_sandbox_context() must be overridden via dependency_overrides")


def get_current_agent() -> Any:
    """Dependency injection marker for the currently executing agent."""
    raise RuntimeError("get_current_agent() must be overridden via dependency_overrides")
