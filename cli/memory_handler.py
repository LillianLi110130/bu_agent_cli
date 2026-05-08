from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from rich.console import Console
from rich.table import Table

from agent_core.memory.store import MemoryStore


@dataclass(slots=True)
class MemorySlashResult:
    handled: bool = True


@dataclass(frozen=True, slots=True)
class MemoryReviewHistoryItem:
    created_at: datetime
    status: str
    summary: str
    target: str | None = None


class MemorySlashHandler:
    """Handle /memory slash command operations."""

    def __init__(
        self,
        store: MemoryStore,
        console: Console | None = None,
        review_history: list[MemoryReviewHistoryItem] | None = None,
    ) -> None:
        self._store = store
        self._console = console or Console()
        self._review_history = review_history if review_history is not None else []

    async def handle(self, args: list[str]) -> MemorySlashResult:
        if not args:
            return await self._list()

        subcommand = args[0].lower()
        if subcommand == "list":
            return await self._list()
        if subcommand == "review":
            return await self._review()

        self._console.print(f"[red]未知子命令：{subcommand}[/red]")
        self._console.print("[dim]可用子命令：list、review[/dim]")
        return MemorySlashResult()

    async def _list(self) -> MemorySlashResult:
        snapshot = self._store.load_from_disk()
        if not snapshot.user_entries and not snapshot.memory_entries:
            self._console.print("[yellow]暂无 memory。[/yellow]")
            return MemorySlashResult()

        table = Table(title="Memory")
        table.add_column("目标", style="cyan")
        table.add_column("内容", style="white")

        for entry in snapshot.user_entries:
            table.add_row("user", entry)
        for entry in snapshot.memory_entries:
            table.add_row("memory", entry)

        self._console.print()
        self._console.print(table)
        self._console.print()
        return MemorySlashResult()

    async def _review(self) -> MemorySlashResult:
        if not self._review_history:
            self._console.print("[yellow]暂无 memory review 记录。[/yellow]")
            return MemorySlashResult()

        table = Table(title="最近 memory review")
        table.add_column("时间", style="dim")
        table.add_column("状态", style="cyan")
        table.add_column("目标", style="magenta")
        table.add_column("结果", style="white")

        for item in self._review_history[-20:]:
            table.add_row(
                item.created_at.strftime("%H:%M:%S"),
                item.status,
                item.target or "-",
                item.summary,
            )

        self._console.print()
        self._console.print(table)
        self._console.print()
        return MemorySlashResult()
