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
            self._cockpit([])
            return True

        subcommand = args[0].lower()
        rest = args[1:]
        if subcommand in {"list", "ls"}:
            self._list_teams()
        elif subcommand in {"create", "start"}:
            self._create(rest)
        elif subcommand == "use":
            self._use(rest)
        elif subcommand in {"members", "member"}:
            self._members(rest)
        elif self._team_exists(subcommand):
            self._cockpit(args)
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
            self.console.print("[red]用法：/team create <goal> [--name <name>][/red]")
            return
        name = self._option(args, "--name")
        goal = " ".join(self._positional_without_options(args, {"--name"})).strip()
        if not goal:
            self.console.print("[red]team goal 不能为空。[/red]")
            return
        try:
            team = self.runtime.start_team(goal=goal, name=name)
        except ValueError as exc:
            self.console.print(f"[yellow]{exc}[/yellow]")
            return
        self.console.print(f"[green]已创建并切换到 team：[/green]{team.team_id}")
        self.console.print(f"[dim]Goal:[/] {team.goal}")

    def _use(self, args: list[str]) -> None:
        if not args:
            active = self.runtime.get_active_team()
            if active:
                self.console.print(f"[green]当前 active team：[/green]{active}")
            else:
                self.console.print("[yellow]当前 workspace 没有 active team。[/yellow]")
            return
        team_id = args[0]
        try:
            self.runtime.set_active_team(team_id)
        except FileNotFoundError:
            self.console.print(f"[red]Team 不存在：{team_id}[/red]")
            return
        self.console.print(f"[green]已切换 active team：[/green]{team_id}")

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
                "[red]用法：/team spawn <team_id> <member_id> [--agent <agent_type>][/red]"
            )
            return
        team_id, member_id = args[0], args[1]
        agent_type = self._option(args[2:], "--agent") or "general-purpose"
        member = self.runtime.spawn_member(
            team_id=team_id,
            member_id=member_id,
            agent_type=agent_type,
        )
        self.console.print(
            f"[green]已启动 teammate：[/green]{member.member_id} "
            f"[dim](pid={member.pid}, agent={member.agent_type})[/dim]"
        )

    def _task(self, args: list[str]) -> None:
        resolved = self._resolve_team_args(args)
        if resolved is None or not resolved[1]:
            self.console.print(
                "[red]用法：/team task <team_id> <title> "
                "[--assign <member>] [--to <member>] [--desc <desc>][/red]"
            )
            return
        team_id, rest = resolved
        assigned_to = self._option(rest, "--assign") or self._option(rest, "--to")
        description = self._option(rest, "--desc")
        write_scope = self._multi_option(rest, "--write")
        title_parts = self._positional_without_options(
            rest,
            {"--assign", "--to", "--desc", "--write"},
        )
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
        team_id = self._resolve_team_id(args)
        if team_id is None:
            self.console.print("[red]用法：/team tasks [team_id][/red]")
            return
        tasks = self.runtime.list_tasks(team_id)
        table = Table(title=f"Tasks: {team_id}")
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
        team_id = self._resolve_team_id(args)
        if team_id is None:
            self.console.print("[red]用法：/team inbox [team_id] [--peek][/red]")
            return
        ack = "--peek" not in args
        messages = self.runtime.read_lead_inbox(team_id, ack=ack)
        if not messages:
            self.console.print("[dim]lead inbox 为空。[/dim]")
            return
        for message in messages:
            self.console.print(
                f"[cyan]{message.message_id}[/cyan] "
                f"{message.sender} -> lead [{message.type}]\n{message.body}"
            )

    def _send(self, args: list[str]) -> None:
        resolved = self._resolve_team_args(args)
        if resolved is None or len(resolved[1]) < 2:
            self.console.print(
                "[red]用法：/team send [team_id] <member_id> <message> [--from <sender>][/red]"
            )
            return
        team_id, rest = resolved
        sender = self._option(rest[1:], "--from") or "lead"
        body_parts = self._positional_without_options(rest[1:], {"--from"})
        message = self.runtime.send_message(
            team_id=team_id,
            recipient=rest[0],
            sender=sender,
            body=" ".join(body_parts),
        )
        self.console.print(f"[green]已发送消息：[/green]{message.message_id}")

    def _status(self, args: list[str]) -> None:
        team_id = self._resolve_team_id(args)
        if team_id is None:
            self.console.print("[red]用法：/team status [team_id][/red]")
            return
        self._print_cockpit(team_id)

    def _stop(self, args: list[str]) -> None:
        if len(args) < 2:
            self.console.print("[red]用法：/team stop <team_id> <member_id>[/red]")
            return
        self.runtime.stop_member(args[0], args[1])
        self.console.print(f"[yellow]已请求停止 teammate：[/yellow]{args[1]}")

    def _shutdown(self, args: list[str]) -> None:
        team_id = self._resolve_team_id(args)
        if team_id is None:
            self.console.print("[red]用法：/team shutdown [team_id][/red]")
            return
        self.runtime.shutdown_team(team_id)
        self.console.print(f"[yellow]已请求关闭 team：[/yellow]{team_id}")

    def _members(self, args: list[str]) -> None:
        team_id = self._resolve_team_id(args)
        if team_id is None:
            self.console.print("[red]用法：/team members [team_id][/red]")
            return
        status = self.runtime.status(team_id)
        table = Table(title=f"Members: {team_id}")
        table.add_column("Member", style="cyan")
        table.add_column("Agent")
        table.add_column("Status")
        table.add_column("PID")
        table.add_column("Heartbeat")
        for member in status["members"]:
            heartbeat = member.get("heartbeat") or {}
            table.add_row(
                member["member_id"],
                member["agent_type"],
                heartbeat.get("status") or member["status"],
                str(member.get("pid") or "-"),
                member.get("last_heartbeat_at") or "-",
            )
        self.console.print(table)

    def _cockpit(self, args: list[str]) -> None:
        team_id = args[0] if args else self.runtime.get_active_team()
        if team_id is None:
            self.console.print("[yellow]当前 workspace 没有 active team。[/yellow]")
            self.console.print("[dim]使用 /team create <goal> [--name <name>] 创建一个 team。[/dim]")
            self.console.print(
                "[dim]或者使用 /team auto <goal> [--name <name>] 让 lead 自动创建、拆分并编排 team。[/dim]"
            )
            return
        self._print_cockpit(team_id)

    def _print_cockpit(self, team_id: str) -> None:
        try:
            status = self.runtime.status(team_id)
        except FileNotFoundError:
            self.console.print(f"[red]Team 不存在：{team_id}[/red]")
            return
        team = status["team"]
        state = status.get("state") or {}
        active_marker = " [green](active)[/green]" if self.runtime.get_active_team() == team_id else ""
        self.console.print(
            f"[bold cyan]{team['team_id']}[/bold cyan]{active_marker} "
            f"{team['status']} / {state.get('phase', '-')}"
        )
        self.console.print(f"[dim]Goal:[/] {team['goal']}")
        self.console.print(f"members={len(status['members'])}, tasks={len(status['tasks'])}")

        member_table = Table(title="Members")
        member_table.add_column("Member", style="cyan")
        member_table.add_column("Agent")
        member_table.add_column("State")
        for member in status["members"]:
            heartbeat = member.get("heartbeat") or {}
            member_table.add_row(
                member["member_id"],
                member["agent_type"],
                heartbeat.get("status") or member["status"],
            )
        self.console.print(member_table)

        task_counts: dict[str, int] = {}
        for task in status["tasks"]:
            task_counts[task["status"]] = task_counts.get(task["status"], 0) + 1
        task_summary = ", ".join(f"{count} {name}" for name, count in sorted(task_counts.items()))
        self.console.print(f"[dim]Tasks:[/] {task_summary or 'none'}")

    def _print_usage(self) -> None:
        self.console.print(
            "[dim]用法：/team auto <goal> [--name <name>] | create <goal> [--name <name>] | use|list|spawn|task|tasks|members|inbox|send|status|stop|shutdown[/dim]"
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
                skip_next = item in {"--name", "--worker-agent", "--assign", "--to", "--desc", "--write", "--from"}
                continue
            values.append(item)
        return values

    def _resolve_team_id(self, args: list[str]) -> str | None:
        if args and not args[0].startswith("--"):
            return args[0]
        return self.runtime.get_active_team()

    def _resolve_team_args(self, args: list[str]) -> tuple[str, list[str]] | None:
        active = self.runtime.get_active_team()
        if args and self._team_exists(args[0]):
            return args[0], args[1:]
        if active is not None:
            return active, args
        if not args:
            return None
        return args[0], args[1:]

    def _team_exists(self, team_id: str) -> bool:
        if team_id.startswith("--"):
            return False
        try:
            self.runtime.status(team_id)
        except FileNotFoundError:
            return False
        return True
