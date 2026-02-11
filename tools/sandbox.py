"""Sandbox context for secure filesystem access."""

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class SecurityError(Exception):
    """Raised when a path escapes the sandbox."""

    pass


@dataclass
class SandboxContext:
    """Sandboxed filesystem context with multiple allowed directories."""

    root_dir: Path  # 沙盒的根目录（主边界，向后兼容）
    working_dir: Path  # 当前工作目录（在允许的目录内）
    allowed_dirs: list[Path] = field(default_factory=list)  # 额外允许的目录列表
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    subagent_manager: Any | None = None

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
        # 初始化时 root_dir 也在允许列表中
        return cls(root_dir=root, working_dir=root, allowed_dirs=[root], session_id=session_id)

    def add_allowed_dir(self, path: Path | str) -> Path:
        """Add a directory to the allowed list.

        Args:
            path: Directory path to allow

        Returns:
            The resolved path that was added

        Raises:
            SecurityError: If the path doesn't exist or is not a directory
        """
        resolved = Path(path).resolve()
        if not resolved.exists():
            raise SecurityError(f"Path does not exist: {resolved}")
        if not resolved.is_dir():
            raise SecurityError(f"Path is not a directory: {resolved}")

        resolved_str = str(resolved).lower()
        # 检查是否已经在允许列表中
        for allowed in self.allowed_dirs:
            if resolved_str == str(allowed).lower():
                return resolved  # 已存在，直接返回
            # 检查是否是某个已允许目录的子目录
            allowed_str = str(allowed.resolve()).lower()
            if resolved_str.startswith(allowed_str) and resolved_str[len(allowed_str):].startswith(("\\", "/")):
                return resolved  # 已经是某个允许目录的子目录
            # 检查某个已允许目录是否是其子目录（避免循环）
            if allowed_str.startswith(resolved_str) and allowed_str[len(resolved_str):].startswith(("\\", "/")):
                # 用新路径替换旧的子目录
                self.allowed_dirs.remove(allowed)
                break

        self.allowed_dirs.append(resolved)
        return resolved

    def is_allowed(self, path: Path) -> bool:
        """Check if a path is within any allowed directory.

        Args:
            path: Path to check

        Returns:
            True if path is within allowed directories
        """
        resolved = path.resolve()
        for allowed in self.allowed_dirs:
            allowed_resolved = allowed.resolve()
            # 使用 is_relative_to 检查路径是否在允许目录内
            try:
                if resolved.is_relative_to(allowed_resolved):
                    return True
            except AttributeError:
                # Python < 3.9，回退到字符串比较
                resolved_str = str(resolved).lower()
                allowed_str = str(allowed_resolved).lower()
                if resolved_str == allowed_str:
                    return True
                if resolved_str.startswith(allowed_str):
                    remainder = resolved_str[len(allowed_str):]
                    if not remainder or remainder[0] in ("/", "\\"):
                        return True
        return False

    def resolve_path(self, path: str | Path) -> Path:
        """Resolve and validate a path is within the sandbox."""
        path_obj = Path(path)
        if path_obj.is_absolute():
            resolved = path_obj.resolve()
        else:
            resolved = (self.working_dir / path_obj).resolve()

        # Security check: ensure path is within any allowed directory
        if not self.is_allowed(resolved):
            allowed_list = ", ".join(str(d) for d in self.allowed_dirs)
            raise SecurityError(
                f"Path escapes sandbox: {path} -> {resolved}\n"
                f"Allowed directories: {allowed_list}\n"
                f"Use /allow <path> to add a directory to the sandbox."
            )
        return resolved


def get_sandbox_context() -> SandboxContext:
    """Dependency injection marker. Override this in the agent."""
    raise RuntimeError("get_sandbox_context() must be overridden via dependency_overrides")
