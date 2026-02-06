from pathlib import Path
from typing import Dict, Optional, List
from .config import AgentConfig, parse_agent_config


class AgentRegistry:
    """Agent注册表，管理所有agent配置"""

    def __init__(self, agents_dir: Path):
        self._configs: Dict[str, AgentConfig] = {}
        self._load_configs(agents_dir)

    def _load_configs(self, agents_dir: Path):
        if not agents_dir.exists():
            return

        for md_file in agents_dir.glob("*.md"):
            config = parse_agent_config(md_file)
            if config:
                self._configs[config.name] = config

    def get_config(self, name: str) -> AgentConfig | None:
        return self._configs.get(name)

    def list_agents(self, mode: str | None = None) -> list[str]:
        if mode:
            return [name for name, cfg in self._configs.items() if cfg.mode == mode]
        return list(self._configs.keys())

    def list_callable_agents(self) -> list[str]:
        return [name for name, cfg in self._configs.items()
                if cfg.mode in ("subagent", "all")]


_global_registry: AgentRegistry | None = None


def get_agent_registry(agents_dir: Path | None = None) -> AgentRegistry:
    global _global_registry
    if _global_registry is None:
        if agents_dir is None:
            agents_dir = Path(__file__).parent.parent / "prompts" / "agents"
        _global_registry = AgentRegistry(agents_dir)
    return _global_registry
