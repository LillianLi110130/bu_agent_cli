"""CLI-only session runtime bootstrap for context-engineering state."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from agent_core.runtime_paths import tg_agent_home
from tools.sandbox import SandboxContext

CLI_SESSION_RUNTIME_VERSION = 1


def default_cli_state_root() -> Path:
    """Return the CLI state root used for rollout-local context state."""
    return tg_agent_home()


def _resolve_now(now: datetime | None = None) -> datetime:
    if now is None:
        return datetime.now().astimezone()
    if now.tzinfo is None:
        return now.astimezone()
    return now


def _format_timestamp(value: datetime) -> str:
    return value.isoformat(timespec="seconds")


@dataclass(slots=True)
class CLISessionRuntime:
    """Metadata and filesystem layout for one CLI rollout session."""

    session_id: str
    root_dir: Path
    sessions_dir: Path
    rollout_dir: Path
    checkpoints_dir: Path
    artifacts_dir: Path
    working_state_path: Path
    meta_path: Path
    started_at: str
    last_active_at: str
    version: int = CLI_SESSION_RUNTIME_VERSION

    @classmethod
    def create_for_context(
        cls,
        ctx: SandboxContext,
        *,
        root_dir: Path | None = None,
        now: datetime | None = None,
    ) -> "CLISessionRuntime":
        """Create and bind a rollout directory for the current CLI process."""
        resolved_now = _resolve_now(now)
        return cls._create_runtime(
            session_id=ctx.session_id,
            root_dir=root_dir,
            now=resolved_now,
            rollout_dir_name=f"rollout-{resolved_now.strftime('%Y%m%d-%H%M%S')}-{ctx.session_id}",
            ctx=ctx,
        )

    @classmethod
    def create_helper_top_level_runtime(
        cls,
        ctx: SandboxContext,
        *,
        helper_name: str,
        root_dir: Path | None = None,
        now: datetime | None = None,
    ) -> "CLISessionRuntime":
        """Create a helper-specific top-level rollout for a dedicated agent run."""
        resolved_helper_name = str(helper_name).strip()
        if not resolved_helper_name:
            raise ValueError("helper_name must not be empty")

        resolved_now = _resolve_now(now)
        runtime_id = uuid4().hex[:8]
        rollout_dir_name = (
            f"rollout-{resolved_now.strftime('%Y%m%d-%H%M%S')}-{resolved_helper_name}-{runtime_id}"
        )
        return cls._create_runtime(
            session_id=runtime_id,
            root_dir=root_dir,
            now=resolved_now,
            rollout_dir_name=rollout_dir_name,
            ctx=ctx,
        )

    @classmethod
    def _create_runtime(
        cls,
        *,
        session_id: str,
        rollout_dir_name: str,
        ctx: SandboxContext,
        root_dir: Path | None = None,
        now: datetime | None = None,
    ) -> "CLISessionRuntime":
        """Create one rollout runtime with a precomputed session id and directory name."""
        resolved_now = _resolve_now(now)
        resolved_root = (root_dir or default_cli_state_root()).expanduser().resolve()
        resolved_root.mkdir(parents=True, exist_ok=True)

        sessions_dir = resolved_root / "sessions"
        rollout_dir = (
            sessions_dir
            / resolved_now.strftime("%Y")
            / resolved_now.strftime("%m")
            / resolved_now.strftime("%d")
            / rollout_dir_name
        )
        rollout_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir = rollout_dir / "checkpoints"
        artifacts_dir = rollout_dir / "artifacts"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        ctx.add_allowed_dir(resolved_root)

        runtime = cls(
            session_id=session_id,
            root_dir=resolved_root,
            sessions_dir=sessions_dir,
            rollout_dir=rollout_dir,
            checkpoints_dir=checkpoints_dir,
            artifacts_dir=artifacts_dir,
            working_state_path=rollout_dir / "working_state.json",
            meta_path=rollout_dir / "meta.json",
            started_at=_format_timestamp(resolved_now),
            last_active_at=_format_timestamp(resolved_now),
        )
        runtime.write_meta()
        return runtime

    def touch(self, *, now: datetime | None = None) -> None:
        """Refresh last-active timestamp for the current rollout session."""
        self.last_active_at = _format_timestamp(_resolve_now(now))
        self.write_meta()

    def to_meta(self) -> dict[str, object]:
        """Serialize the rollout metadata stored on disk."""
        return {
            "session_id": self.session_id,
            "rollout_dir_name": self.rollout_dir.name,
            "started_at": self.started_at,
            "last_active_at": self.last_active_at,
            "version": self.version,
        }

    def write_meta(self) -> None:
        """Persist rollout metadata to ``meta.json``."""
        self.meta_path.write_text(
            json.dumps(self.to_meta(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
