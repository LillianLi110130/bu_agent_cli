"""Sandbox context for secure filesystem access."""

import uuid
from dataclasses import dataclass, field
from pathlib import Path


class SecurityError(Exception):
    """Raised when a path escapes the sandbox."""

    pass


@dataclass
class SandboxContext:
    """Sandboxed filesystem context. All file operations are restricted to root_dir."""

    root_dir: Path  # 沙盒的根目录（边界）
    working_dir: Path  # 当前工作目录（在root_dir内）
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    @classmethod
    def create(cls, root_dir: Path | str | None = None) -> "SandboxContext":
        """Create a new sandbox context, defaulting to current directory."""
        session_id = str(uuid.uuid4())[:8]
        if root_dir is None:
            root = Path.cwd().resolve()
        else:
            root = Path(root_dir).resolve()
        if not root.exists():
            root.mkdir(parents=True, exist_ok=True)
        return cls(root_dir=root, working_dir=root, session_id=session_id)

    def resolve_path(self, path: str | Path) -> Path:
        """Resolve and validate a path is within the sandbox."""
        path_obj = Path(path)
        if path_obj.is_absolute():
            resolved = path_obj.resolve()
        else:
            resolved = (self.working_dir / path_obj).resolve()

        # Security check: ensure path is within sandbox
        resolved_str = str(resolved).lower()
        root_str = str(self.root_dir.resolve()).lower()

        if not resolved_str.startswith(root_str):
            raise SecurityError(
                f"Path escapes sandbox: {path} -> {resolved} (root: {self.root_dir})"
            )
        if resolved_str != root_str and not resolved_str[len(root_str) :].startswith(("\\", "/")):
            raise SecurityError(
                f"Path escapes sandbox: {path} -> {resolved} (root: {self.root_dir})"
            )
        return resolved


def get_sandbox_context() -> SandboxContext:
    """Dependency injection marker. Override this in the agent."""
    raise RuntimeError("get_sandbox_context() must be overridden via dependency_overrides")
