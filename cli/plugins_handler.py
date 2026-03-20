from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from bu_agent_sdk.plugin import PluginManager


@dataclass(slots=True)
class PluginSlashResult:
    handled: bool = True
    reloaded: bool = False


class PluginSlashHandler:
    """Handle /plugins slash command operations."""

    def __init__(
        self,
        manager: PluginManager,
        console: Console | None = None,
    ):
        self._manager = manager
        self._console = console or Console()

    async def handle(self, args: list[str]) -> PluginSlashResult:
        if not args:
            return await self._list()

        subcommand = args[0].lower()
        sub_args = args[1:]

        if subcommand in {"list", "ls"}:
            return await self._list()
        if subcommand == "show":
            return await self._show(sub_args)
        if subcommand == "copy":
            return await self._copy(sub_args)
        if subcommand == "reload":
            return await self._reload()
        if subcommand == "install":
            return await self._install(sub_args)
        if subcommand == "uninstall":
            return await self._uninstall(sub_args)

        self._console.print(f"[red]Unknown subcommand: {subcommand}[/red]")
        self._console.print("[dim]Available: list, show, copy, reload, install, uninstall[/dim]")
        return PluginSlashResult()

    async def _list(self) -> PluginSlashResult:
        plugins = self._manager.list_plugins()
        if not plugins:
            self._console.print("[yellow]No plugins found.[/yellow]")
            return PluginSlashResult()

        table = Table(title="Plugins")
        table.add_column("Name", style="cyan")
        table.add_column("Source", style="magenta")
        table.add_column("Status", style="white")
        table.add_column("Version", style="dim")
        table.add_column("Resources", style="white")
        table.add_column("Description", style="white")

        for plugin in plugins:
            resources = f"skills={len(plugin.skills)} agents={len(plugin.agents)} commands={len(plugin.commands)}"
            status_style = "green" if plugin.status == "loaded" else "red"
            version = plugin.version or "-"
            description = plugin.description or (plugin.error or "")
            table.add_row(
                plugin.name,
                plugin.source,
                f"[{status_style}]{plugin.status}[/{status_style}]",
                version,
                resources,
                description,
            )

        self._console.print()
        self._console.print(table)
        self._console.print()
        return PluginSlashResult()

    async def _show(self, args: list[str]) -> PluginSlashResult:
        if not args:
            self._console.print("[red]Usage: /plugins show <name>[/red]")
            return PluginSlashResult()

        plugin = self._manager.get_plugin(args[0])
        if plugin is None:
            self._console.print(f"[red]Plugin not found: {args[0]}[/red]")
            return PluginSlashResult()

        lines = [
            f"[bold cyan]Name:[/] {plugin.name}",
            f"[bold cyan]Source:[/] {plugin.source}",
            f"[bold cyan]Status:[/] {plugin.status}",
            f"[bold cyan]Version:[/] {plugin.version or '-'}",
            f"[bold cyan]Path:[/] [dim]{plugin.path}[/dim]",
        ]

        if plugin.description:
            lines.append(f"[bold cyan]Description:[/] {plugin.description}")
        if plugin.error:
            lines.append(f"[bold red]Error:[/] {plugin.error}")
        if plugin.skills:
            lines.append(f"[bold cyan]Skills:[/] {', '.join(plugin.skills)}")
        if plugin.agents:
            lines.append(f"[bold cyan]Agents:[/] {', '.join(plugin.agents)}")
        if plugin.commands:
            command_labels = []
            for item in plugin.commands:
                command = self._manager.get_command(item)
                mode_suffix = f" ({command.mode})" if command is not None else ""
                command_labels.append("/" + item + mode_suffix)
            lines.append(f"[bold cyan]Commands:[/] {', '.join(command_labels)}")
        if plugin.warnings:
            lines.append(f"[bold yellow]Warnings:[/] {'; '.join(plugin.warnings)}")

        self._console.print()
        self._console.print(
            Panel(
                "\n".join(lines),
                title=f"[bold blue]Plugin: {plugin.name}[/bold blue]",
                border_style="bright_blue",
                padding=(1, 2),
            )
        )
        self._console.print()
        return PluginSlashResult()

    async def _copy(self, args: list[str]) -> PluginSlashResult:
        if not args:
            self._console.print("[red]Usage: /plugins copy <name>[/red]")
            return PluginSlashResult()

        try:
            target_dir = self._manager.copy_builtin_plugin(args[0])
        except FileExistsError as exc:
            self._console.print(f"[red]{exc}[/red]")
            return PluginSlashResult()
        except ValueError as exc:
            self._console.print(f"[red]{exc}[/red]")
            return PluginSlashResult()

        self._console.print(f"[green]Copied plugin to workspace:[/green] [dim]{target_dir}[/dim]")
        self._console.print("[dim]Run /plugins reload after editing.[/dim]")
        return PluginSlashResult()

    async def _reload(self) -> PluginSlashResult:
        plugins = self._manager.reload_all()
        loaded = sum(1 for plugin in plugins if plugin.status == "loaded")
        failed = sum(1 for plugin in plugins if plugin.status != "loaded")
        self._console.print(f"[green]Reloaded plugins.[/green] loaded={loaded} failed={failed}")
        return PluginSlashResult(reloaded=True)

    async def _install(self, args: list[str]) -> PluginSlashResult:
        if not args:
            self._console.print("[red]Usage: /plugins install <plugin_path>[/red]")
            self._console.print("[dim]Example: /plugins install /path/to/my-plugin[/dim]")
            return PluginSlashResult()

        source_path_str = " ".join(args)
        source_path = Path(source_path_str).expanduser().resolve()

        if not source_path.exists():
            self._console.print(f"[red]Path does not exist: {source_path_str}[/red]")
            return PluginSlashResult()

        plugin_json_path = source_path / "plugin.json"
        if not plugin_json_path.exists():
            self._console.print(
                f"[red]plugin.json not found in: {source_path_str}[/red]"
            )
            self._console.print("[dim]A valid plugin must contain a plugin.json file.[/dim]")
            return PluginSlashResult()

        # Load plugin.json to get the plugin name
        try:
            import json
            manifest_data = json.loads(plugin_json_path.read_text(encoding="utf-8"))
            plugin_name = manifest_data.get("name", "")
            if not plugin_name:
                self._console.print(
                    f"[red]Invalid plugin.json: missing 'name' field[/red]"
                )
                return PluginSlashResult()
        except json.JSONDecodeError as e:
            self._console.print(f"[red]Invalid plugin.json: {e}[/red]")
            return PluginSlashResult()

        # Check if plugin already exists
        dest_path = self._manager.plugin_dir / plugin_name
        if dest_path.exists():
            self._console.print(
                f"[yellow]Plugin '{plugin_name}' already exists at: {dest_path}[/yellow]"
            )
            self._console.print("[dim]Use /plugins reload to reload the plugin.[/dim]")
            return PluginSlashResult()

        # Copy plugin directory
        try:
            self._console.print(f"[dim]Copying plugin from {source_path}...[/dim]")
            shutil.copytree(source_path, dest_path)
            self._console.print(f"[green]Plugin '{plugin_name}' installed successfully.[/green]")
            self._console.print(f"[dim]Location: {dest_path}[/dim]")
        except Exception as e:
            self._console.print(f"[red]Failed to copy plugin: {e}[/red]")
            return PluginSlashResult()

        # Reload plugins to pick up the newly installed one
        self._console.print("[dim]Reloading plugins...[/dim]")
        return PluginSlashResult(reloaded=True)

    async def _uninstall(self, args: list[str]) -> PluginSlashResult:
        """Uninstall a plugin by removing its directory."""
        if not args:
            self._console.print("[red]Usage: /plugins uninstall <plugin_name>[/red]")
            self._console.print("[dim]Example: /plugins uninstall my-plugin[/dim]")
            return PluginSlashResult()

        plugin_name = args[0]
        plugin = self._manager.get_plugin(plugin_name)

        if plugin is None:
            self._console.print(f"[red]Plugin not found: {plugin_name}[/red]")
            self._console.print("[dim]Use /plugins list to see available plugins.[/dim]")
            return PluginSlashResult()

        # Check if plugin path exists
        plugin_path = self._manager.plugin_dir / plugin_name
        if not plugin_path.exists():
            self._console.print(f"[red]Plugin directory not found: {plugin_path}[/red]")
            return PluginSlashResult()

        # Safety check: don't delete if it's not under plugins dir
        try:
            plugin_path.resolve().relative_to(self._manager.plugin_dir.resolve())
        except ValueError:
            self._console.print(f"[red]Plugin path is not under plugins directory: {plugin_path}[/red]")
            return PluginSlashResult()

        # Confirm before deleting (unless --force flag is provided)
        force = "--force" in args or "-f" in args
        if not force:
            self._console.print(f"[yellow]About to uninstall plugin: {plugin_name}[/yellow]")
            self._console.print(f"[dim]Path: {plugin_path}[/dim]")
            self._console.print("[dim]Use --force to skip confirmation in non-interactive mode.[/dim]")
            # In non-interactive CLI, we can't wait for input, so require --force
            self._console.print("[red]Please add --force flag to confirm uninstallation.[/red]")
            return PluginSlashResult()

        # Remove plugin directory
        try:
            shutil.rmtree(plugin_path)
            self._console.print(f"[green]Plugin '{plugin_name}' uninstalled successfully.[/green]")
        except Exception as e:
            self._console.print(f"[red]Failed to uninstall plugin: {e}[/red]")
            return PluginSlashResult()

        # Reload plugins to update the registry
        self._console.print("[dim]Reloading plugins...[/dim]")
        return PluginSlashResult(reloaded=True)
