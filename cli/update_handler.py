from __future__ import annotations

from rich.console import Console

from agent_core.updater import run_check, run_status


class UpdateSlashHandler:
    """Handle the /update slash command group."""

    def __init__(self, *, console: Console | None = None):
        self.console = console or Console()

    async def handle(self, args: list[str]) -> bool:
        command = args[0].lower() if args else "status"
        if command == "check":
            exit_code = run_check()
            if exit_code != 0:
                self.console.print("[red]检查更新失败。[/red]")
            return True
        if command == "status":
            run_status()
            return True

        self.console.print("[red]用法：/update [check|status][/red]")
        return True
