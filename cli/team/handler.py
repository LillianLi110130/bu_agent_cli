"""Slash command handler for filesystem-backed agent teams."""

from __future__ import annotations

import shlex
from typing import Any

from agent_core.team import team_experiment_disabled_message
from rich.console import Console
from rich.table import Table


class TeamSlashHandler:
    """Handle /team commands for the primary CLI lead."""

    def __init__(self, *, runtime: Any, console: Console | None = None):
        self.runtime = runtime
        self.console = console or Console()

    async def handle(self, args: list[str]) -> bool:
        if self.runtime is None:
            self.console.print(f"[yellow]{team_experiment_disabled_message()}[/yellow]")
            return True
        if not args:
            self._print_usage()
            return True

        subcommand = args[0].lower()
        rest = args[1:]
        if subcommand in {"list", "ls"}:
            self._list_teams()
        elif subcommand == "create":
            self._create(rest)
        elif subcommand == "spawn":
            self._spawn(rest)
        elif subcommand in {"task", "task-add", "add-task"}:
            self._task(rest)
        elif subcommand == "tasks":
            self._tasks(rest)
        elif subcommand == "inbox":
            self._inbox(rest)
        elif subcommand == "send":
            self._send(rest)
        elif subcommand == "status":
            self._status(rest)
        elif subcommand == "stop":
            self._stop(rest)
        elif subcommand == "shutdown":
            self._shutdown(rest)
        else:
            self.console.print(f"[red]未知 /team 子命令：{subcommand}[/red]")
            self._print_usage()
        return True

    def _create(self, args: list[str]) -> None:
        if not args:
            self.console.print("[red]用法：/team create <name> [goal][/red]")
            return
        name = args[0]
        goal = " ".join(args[1:]).strip() or name
        team = self.runtime.create_team(name=name, goal=goal)
        self.console.print(f"[green]已创建 team：[/green]{team.team_id}")

    def _list_teams(self) -> None:
        teams = self.runtime.list_teams()
        table = Table(title="Agent Teams")
        table.add_column("Team ID", style="cyan")
        table.add_column("Name")
        table.add_column("Status")
        table.add_column("Goal")
        for team in teams:
            table.add_row(team.team_id, team.name, team.status, team.goal)
        self.console.print(table)

    def _spawn(self, args: list[str]) -> None:
        if len(args) < 2:
            self.console.print(
                "[red]用法：/team spawn <team_id> <member_id> --agent <agent_type>[/red]"
            )
            return
        team_id, member_id = args[0], args[1]
        agent_type = self._option(args[2:], "--agent") or "general-purpose"
        role = self._option(args[2:], "--role") or "member"
        member = self.runtime.spawn_member(
            team_id=team_id,
            member_id=member_id,
            agent_type=agent_type,
            role=role,
        )
        self.console.print(
            f"[green]已启动 teammate：[/green]{member.member_id} "
            f"[dim](pid={member.pid}, agent={member.agent_type})[/dim]"
        )

    def _task(self, args: list[str]) -> None:
        if len(args) < 2:
            self.console.print(
                "[red]用法：/team task <team_id> <title> "
                "[--assign <member>] [--desc <desc>][/red]"
            )
            return
        team_id = args[0]
        assigned_to = self._option(args[1:], "--assign")
        description = self._option(args[1:], "--desc")
        write_scope = self._multi_option(args[1:], "--write")
        title_parts = self._positional_without_options(args[1:], {"--assign", "--desc", "--write"})
        title = " ".join(title_parts).strip()
        if not title:
            self.console.print("[red]任务标题不能为空。[/red]")
            return
        task = self.runtime.create_task(
            team_id=team_id,
            title=title,
            description=description or title,
            assigned_to=assigned_to,
            write_scope=write_scope,
        )
        self.console.print(f"[green]已创建任务：[/green]{task.task_id}")

    def _tasks(self, args: list[str]) -> None:
        if not args:
            self.console.print("[red]用法：/team tasks <team_id>[/red]")
            return
        tasks = self.runtime.list_tasks(args[0])
        table = Table(title=f"Tasks: {args[0]}")
        table.add_column("Task ID", style="cyan")
        table.add_column("Status")
        table.add_column("Assigned")
        table.add_column("Claimed")
        table.add_column("Title")
        for task in tasks:
            table.add_row(
                task.task_id,
                task.status,
                task.assigned_to or "-",
                task.claimed_by or "-",
                task.title,
            )
        self.console.print(table)

    def _inbox(self, args: list[str]) -> None:
        if not args:
            self.console.print("[red]用法：/team inbox <team_id> [--peek][/red]")
            return
        ack = "--peek" not in args[1:]
        messages = self.runtime.read_lead_inbox(args[0], ack=ack)
        if not messages:
            self.console.print("[dim]lead inbox 为空。[/dim]")
            return
        for message in messages:
            self.console.print(
                f"[cyan]{message.message_id}[/cyan] "
                f"{message.sender} -> lead [{message.type}]\n{message.body}"
            )

    def _send(self, args: list[str]) -> None:
        if len(args) < 3:
            self.console.print(
                "[red]用法：/team send <team_id> <member_id> <message> [--from <sender>][/red]"
            )
            return
        sender = self._option(args[2:], "--from") or "lead"
        body_parts = self._positional_without_options(args[2:], {"--from"})
        message = self.runtime.send_message(
            team_id=args[0],
            recipient=args[1],
            sender=sender,
            body=" ".join(body_parts),
        )
        self.console.print(f"[green]已发送消息：[/green]{message.message_id}")

    def _status(self, args: list[str]) -> None:
        if not args:
            self.console.print("[red]用法：/team status <team_id>[/red]")
            return
        status = self.runtime.status(args[0])
        team = status["team"]
        self.console.print(
            f"[bold cyan]{team['team_id']}[/bold cyan] "
            f"{team['status']} - {team['goal']}"
        )
        self.console.print(f"members={len(status['members'])}, tasks={len(status['tasks'])}")

    def _stop(self, args: list[str]) -> None:
        if len(args) < 2:
            self.console.print("[red]用法：/team stop <team_id> <member_id>[/red]")
            return
        self.runtime.stop_member(args[0], args[1])
        self.console.print(f"[yellow]已请求停止 teammate：[/yellow]{args[1]}")

    def _shutdown(self, args: list[str]) -> None:
        if not args:
            self.console.print("[red]用法：/team shutdown <team_id>[/red]")
            return
        self.runtime.shutdown_team(args[0])
        self.console.print(f"[yellow]已请求关闭 team：[/yellow]{args[0]}")

    def _print_usage(self) -> None:
        self.console.print(
            "[dim]用法：/team create|list|spawn|task|tasks|inbox|send|status|stop|shutdown[/dim]"
        )

    @staticmethod
    def parse_args_text(args_text: str) -> list[str]:
        return shlex.split(args_text)

    @staticmethod
    def _option(args: list[str], name: str) -> str | None:
        if name not in args:
            return None
        index = args.index(name)
        if index + 1 >= len(args):
            return None
        return args[index + 1]

    @staticmethod
    def _multi_option(args: list[str], name: str) -> list[str]:
        values: list[str] = []
        index = 0
        while index < len(args):
            if args[index] == name and index + 1 < len(args):
                values.append(args[index + 1])
                index += 2
            else:
                index += 1
        return values

    @staticmethod
    def _positional_without_options(args: list[str], options: set[str]) -> list[str]:
        values: list[str] = []
        skip_next = False
        for item in args:
            if skip_next:
                skip_next = False
                continue
            if item in options:
                skip_next = True
                continue
            values.append(item)
        return values
