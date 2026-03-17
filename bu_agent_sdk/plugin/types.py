from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass(slots=True)
class PluginCapabilities:
    """Declared resource types contributed by a plugin."""

    skills: bool | None = None
    agents: bool | None = None
    commands: bool | None = None

    @classmethod
    def from_dict(cls, data: dict | None) -> "PluginCapabilities":
        if not isinstance(data, dict):
            return cls()
        return cls(
            skills=_normalize_optional_bool(data.get("skills")),
            agents=_normalize_optional_bool(data.get("agents")),
            commands=_normalize_optional_bool(data.get("commands")),
        )

    def allows(self, kind: str, exists: bool) -> bool:
        declared = getattr(self, kind)
        return exists if declared is None else declared


@dataclass(slots=True)
class PluginManifest:
    """Plugin metadata loaded from plugin.json."""

    schema_version: int
    name: str
    version: str
    description: str
    min_cli_version: str | None = None
    capabilities: PluginCapabilities = field(default_factory=PluginCapabilities)


@dataclass(slots=True)
class PluginCommand:
    """Slash command resource contributed by a plugin."""

    plugin_name: str
    name: str
    description: str
    path: Path
    mode: Literal["prompt", "python"] = "prompt"
    content: str = ""
    script: Path | None = None
    usage: str = ""
    category: str = "Plugins"
    examples: list[str] = field(default_factory=list)

    @property
    def full_name(self) -> str:
        return f"{self.plugin_name}:{self.name}"

    @property
    def plugin_root(self) -> Path:
        return self.path.parent.parent

    def render_prompt(self, args_text: str) -> str:
        if self.mode != "prompt":
            raise ValueError(f"Command '{self.full_name}' is not a prompt command")

        args_text = args_text.strip()
        prompt = self.content.strip()
        if "{{args}}" in prompt:
            return prompt.replace("{{args}}", args_text)
        if not args_text:
            return prompt
        return f"{prompt}\n\n---\n\n## Command Arguments\n{args_text}"


PluginPromptCommand = PluginCommand


@dataclass(slots=True)
class PluginRecord:
    """Current runtime status for a single plugin."""

    name: str
    path: Path
    status: str
    manifest: PluginManifest | None = None
    commands: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    agents: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def version(self) -> str | None:
        if self.manifest is None:
            return None
        return self.manifest.version

    @property
    def description(self) -> str:
        if self.manifest is None:
            return ""
        return self.manifest.description


def _normalize_optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    return bool(value)
