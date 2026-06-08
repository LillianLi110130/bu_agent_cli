"""Slash command handler for LSP management."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from agent_core.lsp.manager import LSPManager


class LspSlashHandler:
    """Handle the /lsp slash command group."""

    def __init__(self, *, manager: LSPManager | None, console: Console) -> None:
        self.manager = manager
        self.console = console

    async def handle(self, args: list[str]) -> bool:
        if self.manager is None:
            self.console.print("[yellow]LSP manager 未初始化。[/yellow]")
            return True

        if not args or args[0].lower() == "status":
            self._print_status()
            return True

        command = args[0].lower()
        if command == "start":
            if len(args) != 2:
                self.console.print("[red]用法：/lsp start <server>[/red]")
                return True
            await self._start(args[1])
            return True

        self.console.print("[yellow]/lsp 仅支持 status、start <server>。[/yellow]")
        return True

    async def _start(self, server_name: str) -> None:
        try:
            client = await self.manager.start_server(server_name)
        except Exception as exc:
            self.console.print(f"[red]LSP server 启动失败：{exc}[/red]")
            return

        status = client.status()
        self.console.print(
            "[green]LSP server 已启动：[/green]"
            f"{status.get('name', server_name)} "
            f"[dim]root={status.get('root', '-')}[/dim]"
        )

    def _print_status(self) -> None:
        status = self.manager.status()
        self.console.print("[bold cyan]LSP Service[/bold cyan]")
        self.console.print(f"  enabled:         {status.get('enabled')}")
        self.console.print(f"  workspace root:  {status.get('workspaceRoot')}")
        clients = status.get("clients")
        active_count = len(clients) if isinstance(clients, list) else 0
        self.console.print(f"  active clients:  {active_count}")
        self.console.print()

        table = Table(title="Registered Servers")
        table.add_column("Server")
        table.add_column("Command")
        table.add_column("Extensions")
        for server in status.get("servers", []):
            if not isinstance(server, dict):
                continue
            table.add_row(
                str(server.get("name", "-")),
                str(server.get("command", "-")),
                ", ".join(server.get("extensions", [])),
            )
        self.console.print(table)
        self.console.print()
        self.console.print("[bold cyan]Usage[/bold cyan]")
        self.console.print("  /lsp status")
        self.console.print("  /lsp start python")
        self.console.print("  /lsp start typescript")
        self.console.print("  /lsp start java")
