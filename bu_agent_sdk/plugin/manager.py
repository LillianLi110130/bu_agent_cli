from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from bu_agent_sdk.agent.config import parse_agent_config
from bu_agent_sdk.agent.registry import AgentRegistry
from cli.at_commands import AtCommand, AtCommandRegistry
from cli.slash_commands import SlashCommand, SlashCommandRegistry

from .loader import PluginLoader
from .types import PluginCommand, PluginRecord


class PluginManager:
    """Manage built-in prompt-resource plugins from the repository."""

    def __init__(
        self,
        plugin_dir: Path,
        slash_registry: SlashCommandRegistry,
        skill_registry: AtCommandRegistry,
        agent_registry: AgentRegistry,
        *,
        current_cli_version: str | None = None,
    ):
        self.plugin_dir = plugin_dir
        self._loader = PluginLoader(plugin_dir, current_cli_version=current_cli_version)
        self._slash_registry = slash_registry
        self._skill_registry = skill_registry
        self._agent_registry = agent_registry
        self._plugins: dict[str, PluginRecord] = {}
        self._commands: dict[str, PluginCommand] = {}

    def load_all(self) -> list[PluginRecord]:
        self.unload_all()
        loaded: list[PluginRecord] = []
        seen_manifest_names: dict[str, Path] = {}
        for plugin_path in self._loader.discover():
            loaded.append(self._load_plugin(plugin_path, seen_manifest_names))
        loaded.sort(key=lambda item: item.name)
        return loaded

    def reload_all(self) -> list[PluginRecord]:
        return self.load_all()

    def unload_all(self) -> None:
        for plugin in list(self._plugins.values()):
            if plugin.status == "loaded":
                self._unregister_plugin(plugin)
        self._plugins.clear()
        self._commands.clear()

    def list_plugins(self) -> list[PluginRecord]:
        return sorted(self._plugins.values(), key=lambda item: item.name)

    def get_plugin(self, name: str) -> PluginRecord | None:
        return self._plugins.get(name)

    def get_command(self, full_name: str) -> PluginCommand | None:
        return self._commands.get(full_name)

    def _load_plugin(
        self,
        plugin_path: Path,
        seen_manifest_names: dict[str, Path],
    ) -> PluginRecord:
        try:
            manifest = self._loader.load_manifest(plugin_path)
        except Exception as exc:
            record = PluginRecord(
                name=plugin_path.name,
                path=plugin_path,
                status="failed",
                error=str(exc),
            )
            self._plugins[plugin_path.name] = record
            return record

        if manifest.name in seen_manifest_names:
            original_path = seen_manifest_names[manifest.name]
            record = PluginRecord(
                name=plugin_path.name,
                path=plugin_path,
                status="failed",
                manifest=manifest,
                error=(
                    f"Duplicate plugin name '{manifest.name}' conflicts with "
                    f"'{original_path.name}'"
                ),
            )
            self._plugins[plugin_path.name] = record
            return record

        seen_manifest_names[manifest.name] = plugin_path
        record = PluginRecord(
            name=manifest.name,
            path=plugin_path,
            status="loaded",
            manifest=manifest,
        )

        try:
            self._register_skills(record)
            self._register_agents(record)
            self._register_commands(record)
        except Exception as exc:
            self._unregister_plugin(record)
            record.status = "failed"
            record.error = str(exc)

        self._plugins[record.name] = record
        return record

    def _register_skills(self, plugin: PluginRecord) -> None:
        skills_dir = plugin.path / "skills"
        if not self._loader.should_load(plugin.manifest, "skills", skills_dir):  # type: ignore[arg-type]
            return
        if not skills_dir.exists():
            return

        discovered_paths: set[Path] = set()
        for pattern in ("*/skill.md", "*/SKILL.md"):
            discovered_paths.update(skills_dir.glob(pattern))

        for skill_path in sorted(discovered_paths):
            skill = AtCommand.from_file(skill_path)
            namespaced_name = f"{plugin.name}:{skill.name}"
            self._skill_registry.register(
                AtCommand(
                    name=namespaced_name,
                    description=skill.description,
                    path=skill.path,
                    category=skill.category,
                )
            )
            plugin.skills.append(namespaced_name)

    def _register_agents(self, plugin: PluginRecord) -> None:
        agents_dir = plugin.path / "agents"
        if not self._loader.should_load(plugin.manifest, "agents", agents_dir):  # type: ignore[arg-type]
            return
        if not agents_dir.exists():
            return

        for md_file in sorted(agents_dir.glob("*.md")):
            config = parse_agent_config(md_file)
            if config is None:
                plugin.warnings.append(f"Skipped invalid agent file: {md_file.name}")
                continue
            namespaced_name = f"{plugin.name}:{config.name}"
            self._agent_registry.register(replace(config, name=namespaced_name))
            plugin.agents.append(namespaced_name)

    def _register_commands(self, plugin: PluginRecord) -> None:
        commands_dir = plugin.path / "commands"
        if not self._loader.should_load(plugin.manifest, "commands", commands_dir):  # type: ignore[arg-type]
            return
        if not commands_dir.exists():
            return

        for md_file in sorted(commands_dir.glob("*.md")):
            command = self._loader.load_command(plugin.path, plugin.name, md_file)
            self._commands[command.full_name] = command
            self._slash_registry.register(
                SlashCommand(
                    name=command.full_name,
                    description=command.description,
                    usage=command.usage,
                    category=command.category,
                    examples=command.examples,
                    is_builtin=False,
                )
            )
            plugin.commands.append(command.full_name)

    def _unregister_plugin(self, plugin: PluginRecord) -> None:
        for command_name in plugin.commands:
            self._slash_registry.unregister(command_name)
            self._commands.pop(command_name, None)
        for skill_name in plugin.skills:
            self._skill_registry.unregister(skill_name)
        for agent_name in plugin.agents:
            self._agent_registry.unregister(agent_name)
