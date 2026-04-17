from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from agent_core.llm.messages import SystemMessage
from agent_core.skill.manager import SkillManager

if TYPE_CHECKING:
    from agent_core.agent.service import Agent
    from agent_core.plugin import PluginManager
    from cli.at_commands import AtCommand, AtCommandRegistry


@dataclass(frozen=True, slots=True)
class SkillReloadResult:
    total: int
    counts_by_source: dict[str, int] = field(default_factory=dict)
    refreshed_agent_prompt: bool = False

    def format_summary(self) -> str:
        if not self.counts_by_source:
            return f"total={self.total}"
        counts = ", ".join(
            f"{source}={count}" for source, count in sorted(self.counts_by_source.items())
        )
        return f"total={self.total}, {counts}"


class SkillRuntimeService:
    """Shared skill registry service for slash commands and skill tools."""

    def __init__(
        self,
        *,
        skill_registry: "AtCommandRegistry",
        plugin_manager: "PluginManager | None" = None,
        agent: "Agent | None" = None,
        system_prompt_builder: Callable[[], str] | None = None,
        manager: SkillManager | None = None,
    ) -> None:
        self.skill_registry = skill_registry
        self.plugin_manager = plugin_manager
        self.agent = agent
        self.system_prompt_builder = system_prompt_builder
        self.manager = manager or SkillManager()

    def bind_agent(self, agent: "Agent") -> None:
        self.agent = agent

    def list(self) -> list["AtCommand"]:
        return sorted(self.skill_registry.get_all(), key=lambda item: item.name)

    def show(self, name: str) -> "AtCommand | None":
        return self.skill_registry.get(name)

    def view(self, name: str) -> str:
        skill = self.show(name)
        if skill is None:
            raise ValueError(f"Skill not found: {name}")
        return skill.path.read_text(encoding="utf-8")

    def reload(self, *, refresh_agent_prompt: bool = True) -> SkillReloadResult:
        self.skill_registry.discover_skills(skill_dirs=list(self.skill_registry._skill_dirs))
        if self.plugin_manager is not None:
            self.plugin_manager.reload_all()

        refreshed = False
        if refresh_agent_prompt:
            refreshed = self.refresh_agent_prompt()

        counts = Counter(getattr(skill, "source", "workspace") for skill in self.list())
        return SkillReloadResult(
            total=sum(counts.values()),
            counts_by_source=dict(counts),
            refreshed_agent_prompt=refreshed,
        )

    def refresh_agent_prompt(self) -> bool:
        if self.agent is None or self.system_prompt_builder is None:
            return False

        new_prompt = self.system_prompt_builder()
        self.agent.system_prompt = new_prompt

        messages = self.agent._context.get_messages()
        if not messages:
            return True

        for index, message in enumerate(messages):
            if not isinstance(message, SystemMessage):
                continue
            content = str(getattr(message, "content", ""))
            if index == 0 or "你是一个名为 TgAgent" in content:
                messages[index] = SystemMessage(content=new_prompt, cache=True)
                self.agent._context.replace_messages(messages)
                return True

        return True

    def is_writable(self, skill: "AtCommand") -> bool:
        return bool(getattr(skill, "writable", False))

    @staticmethod
    def source_of(skill: "AtCommand") -> str:
        return str(getattr(skill, "source", "workspace"))

    @staticmethod
    def path_of(skill: "AtCommand") -> Path:
        return Path(skill.path)
