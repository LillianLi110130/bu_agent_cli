from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agent_core.skill.runtime_service import SkillRuntimeService


@dataclass(slots=True)
class SkillSlashResult:
    handled: bool = True
    reloaded: bool = False


@dataclass(frozen=True, slots=True)
class SkillReviewHistoryItem:
    created_at: datetime
    status: str
    summary: str
    skill_name: str | None = None


class SkillSlashHandler:
    """Handle /skills slash command operations."""

    def __init__(
        self,
        service: SkillRuntimeService,
        console: Console | None = None,
        review_history: list[SkillReviewHistoryItem] | None = None,
    ) -> None:
        self._service = service
        self._console = console or Console()
        self._review_history = review_history if review_history is not None else []

    async def handle(self, args: list[str]) -> SkillSlashResult:
        if not args:
            return await self._list()

        subcommand = args[0].lower()
        sub_args = args[1:]

        if subcommand == "list":
            return await self._list()
        if subcommand == "reload":
            return await self._reload()
        if subcommand == "show":
            return await self._show(sub_args)
        if subcommand == "review":
            return await self._review()

        self._console.print(f"[red]未知子命令：{subcommand}[/red]")
        self._console.print("[dim]可用子命令：list、reload、show、review[/dim]")
        return SkillSlashResult()

    async def _list(self) -> SkillSlashResult:
        skills = self._service.list()
        if not skills:
            self._console.print("[yellow]未找到技能。[/yellow]")
            return SkillSlashResult()

        table = Table(title="技能")
        table.add_column("名称", style="cyan")
        table.add_column("来源", style="magenta")
        table.add_column("可修改", style="white")
        table.add_column("路径", style="dim")
        table.add_column("描述", style="white")

        for skill in skills:
            writable = "yes" if self._service.is_writable(skill) else "no"
            table.add_row(
                skill.name,
                self._service.source_of(skill),
                writable,
                str(skill.path),
                skill.description,
            )

        self._console.print()
        self._console.print(table)
        self._console.print()
        return SkillSlashResult()

    async def _review(self) -> SkillSlashResult:
        if not self._review_history:
            self._console.print("[yellow]暂无 skill review 记录。[/yellow]")
            return SkillSlashResult()

        table = Table(title="最近 skill review")
        table.add_column("时间", style="dim")
        table.add_column("状态", style="cyan")
        table.add_column("Skill", style="magenta")
        table.add_column("结果", style="white")

        for item in self._review_history[-20:]:
            table.add_row(
                item.created_at.strftime("%H:%M:%S"),
                item.status,
                item.skill_name or "-",
                item.summary,
            )

        self._console.print()
        self._console.print(table)
        self._console.print()
        return SkillSlashResult()

    async def _reload(self) -> SkillSlashResult:
        result = self._service.reload(refresh_agent_prompt=True)
        refreshed = (
            "已刷新当前 agent skill index"
            if result.refreshed_agent_prompt
            else "未刷新当前上下文"
        )
        self._console.print(
            f"[green]技能已重新加载。[/green] {result.format_summary()}，{refreshed}"
        )
        return SkillSlashResult(reloaded=True)

    async def _show(self, args: list[str]) -> SkillSlashResult:
        if not args:
            self._console.print("[red]用法：/skills show <name>[/red]")
            return SkillSlashResult()

        name = args[0]
        skill = self._service.show(name)
        if skill is None:
            self._console.print(f"[red]未找到技能：{name}[/red]")
            return SkillSlashResult()

        lines = [
            f"[bold cyan]名称：[/] {skill.name}",
            f"[bold cyan]描述：[/] {skill.description or '-'}",
            f"[bold cyan]分类：[/] {skill.category}",
            f"[bold cyan]来源：[/] {self._service.source_of(skill)}",
            f"[bold cyan]可修改：[/] {'yes' if self._service.is_writable(skill) else 'no'}",
            f"[bold cyan]路径：[/] [dim]{skill.path}[/dim]",
        ]
        self._console.print()
        self._console.print(
            Panel(
                "\n".join(lines),
                title=f"[bold blue]技能：{skill.name}[/bold blue]",
                border_style="bright_blue",
                padding=(1, 2),
            )
        )
        self._console.print()
        return SkillSlashResult()
