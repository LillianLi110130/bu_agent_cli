from __future__ import annotations

from dataclasses import dataclass

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
        if subcommand == "reload":
            return await self._reload()

        self._console.print(f"[red]Unknown subcommand: {subcommand}[/red]")
        self._console.print("[dim]Available: list, show, reload[/dim]")
        return PluginSlashResult()

    async def _list(self) -> PluginSlashResult:
        plugins = self._manager.list_plugins()
        if not plugins:
            self._console.print("[yellow]No built-in plugins found.[/yellow]")
            return PluginSlashResult()

        table = Table(title="Built-in Plugins")
        table.add_column("Name", style="cyan")
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
            lines.append(f"[bold cyan]Commands:[/] {', '.join('/' + item for item in plugin.commands)}")
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

    async def _reload(self) -> PluginSlashResult:
        plugins = self._manager.reload_all()
        loaded = sum(1 for plugin in plugins if plugin.status == "loaded")
        failed = sum(1 for plugin in plugins if plugin.status != "loaded")
        self._console.print(
            f"[green]Reloaded plugins.[/green] loaded={loaded} failed={failed}"
        )
        return PluginSlashResult(reloaded=True)
