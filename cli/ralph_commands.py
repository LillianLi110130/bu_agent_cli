from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from cli.ralph_service import RalphService


class RalphArgumentError(Exception):
    pass


class RalphArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise RalphArgumentError(message)


class RalphSlashHandler:
    """Handle the /ralph slash command group."""

    def __init__(
        self,
        *,
        workspace_root: Path,
        console: Console | None = None,
        service: RalphService | None = None,
    ):
        self.console = console or Console()
        self.service = service or RalphService(workspace_root)
        self._parser = self._build_parser()

    async def handle(self, args: list[str]) -> bool:
        if not args:
            self.console.print(self._format_help())
            return True

        try:
            namespace = self._parser.parse_args(args)
        except RalphArgumentError as exc:
            self.console.print(f"[red]参数错误：{exc}[/red]")
            self.console.print(self._format_help())
            return True

        command = namespace.command
        if command == "init-spec":
            result = await self.service.init_spec(
                spec_name=namespace.spec_name,
                target_dir=namespace.target_dir,
                force=namespace.force,
            )
            return self._print_result(result.success, result.message)

        if command == "init-agent":
            result = await self.service.init_agent(target_dir=namespace.target_dir)
            return self._print_result(result.success, result.message)

        if command == "dry-run":
            if not self._validate_execution_args("dry-run", namespace.spec_name, namespace.plan_file):
                return True
            result = await self.service.dry_run(
                spec_name=namespace.spec_name,
                plan_file=namespace.plan_file,
                log_dir=namespace.log_dir,
                max_retry=namespace.max_retry,
                delay=namespace.delay,
                enable_git=namespace.enable_git,
                main_branch=namespace.main_branch,
                work_branch=namespace.work_branch,
                silent=namespace.silent,
            )
            return self._print_result(result.success, result.message)

        if command == "run":
            if not self._validate_execution_args("run", namespace.spec_name, namespace.plan_file):
                return True
            result = await self.service.run(
                spec_name=namespace.spec_name,
                plan_file=namespace.plan_file,
                log_dir=namespace.log_dir,
                max_retry=namespace.max_retry,
                delay=namespace.delay,
                enable_git=namespace.enable_git,
                main_branch=namespace.main_branch,
                work_branch=namespace.work_branch,
                silent=namespace.silent,
            )
            return self._print_result(result.success, result.message)

        if command == "status":
            result = await self.service.status(run_id=namespace.run_id)
            return self._print_result(result.success, result.message)

        if command == "cancel":
            result = await self.service.cancel(run_id=namespace.run_id)
            return self._print_result(result.success, result.message)

        self.console.print(f"[red]未知 Ralph 子命令：{command}[/red]")
        return True

    def _print_result(self, success: bool, message: str) -> bool:
        style = "green" if success else "red"
        self.console.print(f"[{style}]{message}[/{style}]")
        return True

    def _validate_execution_args(
        self,
        command: str,
        spec_name: str | None,
        plan_file: str | None,
    ) -> bool:
        if spec_name or plan_file:
            return True

        self.console.print(f"[red]/ralph {command} 需要提供 <spec_name> 或 --plan-file。[/red]")
        self.console.print(f"[dim]用法：/ralph {command} <spec_name> <可选参数>[/dim]")
        self.console.print(f"[dim]   或：/ralph {command} --plan-file <path> <可选参数>[/dim]")
        return False

    def _format_help(self) -> str:
        return "\n".join(
            [
                "Ralph 工作流命令",
                "",
                "用法：",
                "  /ralph init-spec <spec_name> [--target-dir <path>] [--force]",
                "  /ralph init-agent [--target-dir <path>]",
                "  /ralph dry-run <spec_name> [--plan-file <path>] [--log-dir <path>] [--max-retry <n>] [--delay <秒>] [--enable-git] [--main-branch <name>] [--work-branch <name>] [--silent]",
                "  /ralph run <spec_name> [--plan-file <path>] [--log-dir <path>] [--max-retry <n>] [--delay <秒>] [--enable-git] [--main-branch <name>] [--work-branch <name>] [--silent]",
                "  /ralph status [run_id]",
                "  /ralph cancel <run_id>",
            ]
        )

    def _build_parser(self) -> argparse.ArgumentParser:
        parser = RalphArgumentParser(
            prog="/ralph",
            add_help=False,
            description="Ralph workflow commands",
        )
        subparsers = parser.add_subparsers(dest="command")

        init_spec = subparsers.add_parser("init-spec", add_help=False)
        init_spec.add_argument("spec_name")
        init_spec.add_argument("--target-dir", default=".")
        init_spec.add_argument("--force", action="store_true")

        init_agent = subparsers.add_parser("init-agent", add_help=False)
        init_agent.add_argument("--target-dir", default=".")

        dry_run = subparsers.add_parser("dry-run", add_help=False)
        self._add_execution_args(dry_run)

        run = subparsers.add_parser("run", add_help=False)
        self._add_execution_args(run)

        status = subparsers.add_parser("status", add_help=False)
        status.add_argument("run_id", nargs="?")

        cancel = subparsers.add_parser("cancel", add_help=False)
        cancel.add_argument("run_id")

        return parser

    def _add_execution_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("spec_name", nargs="?")
        parser.add_argument("--plan-file")
        parser.add_argument("--log-dir")
        parser.add_argument("--max-retry", type=int)
        parser.add_argument("--delay", type=float)
        parser.add_argument("--enable-git", action="store_true")
        parser.add_argument("--main-branch")
        parser.add_argument("--work-branch")
        parser.add_argument("--silent", action="store_true")
