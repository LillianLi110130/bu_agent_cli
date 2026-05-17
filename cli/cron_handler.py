"""Slash command handler for local cron job management."""

from __future__ import annotations

from rich.console import Console

from cron.jobs import CronJobStore
from cron.service import CronService


class CronSlashHandler:
    """Handle the /cron slash command group."""

    def __init__(self, *, console: Console, service: CronService | None = None) -> None:
        self.console = console
        self.service = service or CronService(CronJobStore())

    async def handle(self, args: list[str]) -> bool:
        if not args or args[0].lower() == "list":
            self.console.print(self.service.format_list_text())
            return True

        command = args[0].lower()
        if command == "get":
            if len(args) != 2:
                self.console.print("[red]用法：/cron get <job_id>[/red]")
                return True
            self.console.print(self.service.format_detail_text(args[1]))
            return True

        if command == "remove":
            if len(args) != 2:
                self.console.print("[red]用法：/cron remove <job_id>[/red]")
                return True
            removed = self.service.remove_job(args[1])
            if removed:
                self.console.print(f"[green]已删除 cron 任务：{args[1]}[/green]")
            else:
                self.console.print(f"[yellow]未找到 cron 任务：{args[1]}[/yellow]")
            return True

        self.console.print("[yellow]/cron 仅支持 list、get <job_id>、remove <job_id>。[/yellow]")
        return True
