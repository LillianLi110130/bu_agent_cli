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
            self.console.print(self._parser.format_help().strip())
            return True

        try:
            namespace = self._parser.parse_args(args)
        except RalphArgumentError as exc:
            self.console.print(f"[red]{exc}[/red]")
            self.console.print(self._parser.format_help().strip())
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

        if command == "ta":
            description = " ".join(namespace.description).strip()
            result = await self.service.ta(
                spec_name=namespace.spec_name,
                description=description,
            )
            return self._print_result(result.success, result.message)

        if command == "decompose":
            description = " ".join(namespace.description).strip()
            result = await self.service.decompose(
                spec_name=namespace.spec_name,
                description=description,
            )
            return self._print_result(result.success, result.message)

        if command == "dry-run":
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

        self.console.print(f"[red]Unknown Ralph subcommand: {command}[/red]")
        return True

    def _print_result(self, success: bool, message: str) -> bool:
        style = "green" if success else "red"
        self.console.print(f"[{style}]{message}[/{style}]")
        return True

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

        ta = subparsers.add_parser("ta", add_help=False)
        ta.add_argument("spec_name")
        ta.add_argument("description", nargs="*")

        decompose = subparsers.add_parser("decompose", add_help=False)
        decompose.add_argument("spec_name")
        decompose.add_argument("description", nargs="*")

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
