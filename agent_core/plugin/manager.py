from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

from agent_core.agent.config import parse_agent_config
from agent_core.agent.registry import AgentRegistry
from cli.at_commands import AtCommand, AtCommandRegistry
from cli.slash_commands import SlashCommand, SlashCommandRegistry

from .loader import PluginLoader
from .types import PluginCommand, PluginRecord


class PluginManager:
    """Manage built-in and workspace plugins."""

    def __init__(
        self,
        plugin_dir: Path | None,
        slash_registry: SlashCommandRegistry,
        skill_registry: AtCommandRegistry,
        agent_registry: AgentRegistry,
        *,
        plugin_dirs: list[tuple[str, Path]] | None = None,
        current_cli_version: str | None = None,
    ):
        if plugin_dirs is None:
            if plugin_dir is None:
                raise ValueError("plugin_dir or plugin_dirs is required")
            plugin_dirs = [("builtin", plugin_dir)]
        elif not plugin_dirs:
            raise ValueError("plugin_dirs must not be empty")

        normalized_dirs: list[tuple[str, Path]] = []
        seen_sources: set[str] = set()
        for source, path in plugin_dirs:
            source_name = source.strip()
            if not source_name:
                raise ValueError("Plugin source name must not be empty")
            if source_name in seen_sources:
                raise ValueError(f"Duplicate plugin source: {source_name}")
            normalized_dirs.append((source_name, path))
            seen_sources.add(source_name)

        self.plugin_dir = plugin_dir or normalized_dirs[0][1]
        self.plugin_dirs = normalized_dirs
        self._loaders = {
            source: PluginLoader(path, current_cli_version=current_cli_version)
            for source, path in normalized_dirs
        }
        self._source_roots = {source: path for source, path in normalized_dirs}
        self._slash_registry = slash_registry
        self._skill_registry = skill_registry
        self._agent_registry = agent_registry
        self._plugins: dict[str, PluginRecord] = {}
        self._commands: dict[str, PluginCommand] = {}
        self._source_plugins: dict[str, dict[str, PluginRecord]] = {
            source: {} for source, _ in normalized_dirs
        }

    def load_all(self) -> list[PluginRecord]:
        self.unload_all()
        merged_records: dict[str, PluginRecord] = {}

        for source, source_root in self.plugin_dirs:
            loader = self._loaders[source]
            source_records = self._discover_source_plugins(source, source_root, loader)
            source_map: dict[str, PluginRecord] = {}
            for record in source_records:
                source_map[record.name] = record
                merged_records[record.name] = record
            self._source_plugins[source] = source_map

        for record in sorted(merged_records.values(), key=lambda item: item.name):
            if record.status != "loaded":
                continue
            try:
                self._register_skills(record)
                self._register_agents(record)
                self._register_commands(record)
            except Exception as exc:
                self._unregister_plugin(record)
                record.status = "failed"
                record.error = str(exc)

        self._plugins = merged_records
        return self.list_plugins()

    def reload_all(self) -> list[PluginRecord]:
        return self.load_all()

    def unload_all(self) -> None:
        for plugin in list(self._plugins.values()):
            if plugin.status == "loaded":
                self._unregister_plugin(plugin)
        self._plugins.clear()
        self._commands.clear()
        self._source_plugins = {source: {} for source, _ in self.plugin_dirs}

    def list_plugins(self) -> list[PluginRecord]:
        return sorted(self._plugins.values(), key=lambda item: item.name)

    def get_plugin(self, name: str) -> PluginRecord | None:
        return self._plugins.get(name)

    def get_plugin_from_source(self, name: str, source: str) -> PluginRecord | None:
        return self._source_plugins.get(source, {}).get(name)

    def get_command(self, full_name: str) -> PluginCommand | None:
        return self._commands.get(full_name)

    def copy_builtin_plugin(self, name: str) -> Path:
        builtin_plugin = self.get_plugin_from_source(name, "builtin")
        if builtin_plugin is None or builtin_plugin.status != "loaded":
            raise ValueError(f"Built-in plugin not found: {name}")

        workspace_root = self._source_roots.get("workspace")
        if workspace_root is None:
            raise ValueError("Workspace plugin directory is not configured")

        target_dir = workspace_root / name
        if target_dir.exists():
            raise FileExistsError(f"Workspace plugin already exists: {target_dir}")

        target_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(builtin_plugin.path, target_dir)
        return target_dir

    def _discover_source_plugins(
        self,
        source: str,
        source_root: Path,
        loader: PluginLoader,
    ) -> list[PluginRecord]:
        discovered: list[PluginRecord] = []
        seen_manifest_names: dict[str, Path] = {}

        for plugin_path in loader.discover():
            try:
                manifest = loader.load_manifest(plugin_path)
            except Exception as exc:
                discovered.append(
                    PluginRecord(
                        name=plugin_path.name,
                        path=plugin_path,
                        status="failed",
                        source=source,
                        source_root=source_root,
                        error=str(exc),
                    )
                )
                continue

            if manifest.name in seen_manifest_names:
                original_path = seen_manifest_names[manifest.name]
                discovered.append(
                    PluginRecord(
                        name=plugin_path.name,
                        path=plugin_path,
                        status="failed",
                        source=source,
                        source_root=source_root,
                        manifest=manifest,
                        error=(
                            f"Duplicate plugin name '{manifest.name}' conflicts with "
                            f"'{original_path.name}'"
                        ),
                    )
                )
                continue

            seen_manifest_names[manifest.name] = plugin_path
            discovered.append(
                PluginRecord(
                    name=manifest.name,
                    path=plugin_path,
                    status="loaded",
                    source=source,
                    source_root=source_root,
                    manifest=manifest,
                )
            )

        return discovered

    def _register_skills(self, plugin: PluginRecord) -> None:
        skills_dir = plugin.path / "skills"
        if not self._loaders[plugin.source].should_load(plugin.manifest, "skills", skills_dir):  # type: ignore[arg-type]
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
        if not self._loaders[plugin.source].should_load(plugin.manifest, "agents", agents_dir):  # type: ignore[arg-type]
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
        if not self._loaders[plugin.source].should_load(plugin.manifest, "commands", commands_dir):  # type: ignore[arg-type]
            return
        if not commands_dir.exists():
            return

        loader = self._loaders[plugin.source]
        for md_file in sorted(commands_dir.glob("*.md")):
            command = loader.load_command(plugin.path, plugin.name, md_file)
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
