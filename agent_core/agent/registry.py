from __future__ import annotations

from pathlib import Path
from typing import Dict

from .config import AgentConfig, parse_agent_config


class AgentRegistry:
    """Agent注册表，管理所有agent配置"""

    def __init__(
        self,
        agents_dir: Path | None = None,
        *,
        agent_sources: list[tuple[str, Path, int]] | None = None,
    ):
        self._configs: Dict[str, AgentConfig] = {}
        self._config_sources: dict[str, tuple[str, int]] = {}

        if agent_sources is not None:
            self._load_sources(agent_sources)
        elif agents_dir is not None:
            self._load_configs(agents_dir, source_scope="builtin", source_priority=4)

    def _load_sources(self, agent_sources: list[tuple[str, Path, int]]) -> None:
        for source_scope, agents_dir, source_priority in sorted(
            agent_sources,
            key=lambda item: item[2],
            reverse=True,
        ):
            self._load_configs(
                agents_dir,
                source_scope=source_scope,
                source_priority=source_priority,
            )

    def _load_configs(
        self,
        agents_dir: Path,
        *,
        source_scope: str,
        source_priority: int,
    ) -> None:
        if not agents_dir.exists():
            return

        for md_file in agents_dir.glob("*.md"):
            config = parse_agent_config(
                md_file,
                source_scope=source_scope,
                source_priority=source_priority,
            )
            if config:
                self.register(config)

    def register(self, config: AgentConfig) -> None:
        """Register a single agent config."""
        current = self._configs.get(config.name)
        if current is not None and current.source_priority < config.source_priority:
            return

        self._configs[config.name] = config
        self._config_sources[config.name] = (config.source_scope, config.source_priority)

    def unregister(self, name: str) -> None:
        """Remove a registered agent config if present."""
        self._configs.pop(name, None)
        self._config_sources.pop(name, None)

    def get_config(self, name: str) -> AgentConfig | None:
        return self._configs.get(name)

    def list_agents(self) -> list[str]:
        return list(self._configs.keys())

    def list_callable_agents(self) -> list[str]:
        return list(self._configs.keys())


_global_registry: AgentRegistry | None = None


def default_agent_sources(
    workspace_root: Path,
    *,
    builtin_agents_dir: Path | None = None,
    user_agents_dir: Path | None = None,
) -> list[tuple[str, Path, int]]:
    """Return the default agent source chain ordered by precedence metadata."""
    resolved_builtin_dir = (
        builtin_agents_dir or Path(__file__).parent.parent / "prompts" / "agents"
    )
    resolved_user_dir = user_agents_dir or (Path.home() / ".tg_agent" / "agents")
    return [
        ("builtin", resolved_builtin_dir, 4),
        ("user", resolved_user_dir, 2),
        ("workspace", workspace_root / ".tg_agent" / "agents", 1),
    ]


def get_agent_registry(
    workspace_root: Path | None = None,
    *,
    builtin_agents_dir: Path | None = None,
) -> AgentRegistry:
    global _global_registry
    if _global_registry is None:
        if workspace_root is None:
            workspace_root = Path.cwd()
        _global_registry = AgentRegistry(
            agent_sources=default_agent_sources(
                workspace_root,
                builtin_agents_dir=builtin_agents_dir,
            )
        )
    return _global_registry
