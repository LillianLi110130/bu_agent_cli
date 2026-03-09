from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from cli.ralph_models import RalphCommandResult, RalphPaths, RalphRunRecord
from cli.ralph_process_manager import RalphProcessManager


class RalphService:
    """Application-facing Ralph workflow service."""

    def __init__(
        self,
        workspace_root: Path,
        *,
        script_root: Path | None = None,
        process_manager: RalphProcessManager | None = None,
        python_executable: str | None = None,
    ):
        self._workspace_root = workspace_root.resolve()
        self._script_root = (script_root or Path(__file__).resolve().parent.parent).resolve()
        self._process_manager = process_manager or RalphProcessManager(self._workspace_root)
        self._python_executable = python_executable or sys.executable
        self._ralph_init_script = self._script_root / "ralph_init.py"
        self._ralph_loop_script = self._script_root / "ralph_loop.py"
        self._decompose_prompt = (
            self._workspace_root / ".devagent" / "commands" / "ralph" / "DECOMPOSE_TASK.md"
        )
        self._implement_prompt = (
            self._workspace_root / ".devagent" / "commands" / "ralph" / "implement.md"
        )

    async def init_spec(
        self,
        *,
        spec_name: str,
        target_dir: str = ".",
        force: bool = False,
    ) -> RalphCommandResult:
        return await asyncio.to_thread(
            self._init_spec_sync,
            spec_name=spec_name,
            target_dir=target_dir,
            force=force,
        )

    async def init_agent(self, *, target_dir: str = ".") -> RalphCommandResult:
        return await asyncio.to_thread(
            self._init_agent_sync,
            target_dir=target_dir,
        )

    async def decompose(
        self,
        *,
        spec_name: str,
        description: str = "",
    ) -> RalphCommandResult:
        paths = self._resolve_paths(spec_name)
        check_error = self._check_decompose_ready(paths)
        if check_error:
            return RalphCommandResult(False, check_error)

        prompt = self._decompose_prompt.read_text(encoding="utf-8")
        prompt += (
            "\n\n## Runtime Context\n"
            f"- workspace: {self._workspace_root}\n"
            f"- spec_name: {spec_name}\n"
            f"- spec_dir: {paths.spec_dir}\n"
            f"- requirement_dir: {paths.requirement_dir}\n"
            f"- plan_dir: {paths.plan_dir}\n"
            f"- plan_file: {paths.plan_file}\n"
        )
        if description:
            prompt += f"\n## Additional Request\n{description.strip()}\n"

        paths.log_dir.mkdir(parents=True, exist_ok=True)
        decompose_log = paths.log_dir / "decompose.log"
        result = await asyncio.to_thread(
            subprocess.run,
            "devagent --yolo",
            cwd=str(self._workspace_root),
            input=prompt,
            text=True,
            capture_output=True,
            encoding="utf-8",
            shell=True,
        )

        self._write_command_log(
            decompose_log,
            command="devagent --yolo",
            stdout=result.stdout,
            stderr=result.stderr,
            return_code=result.returncode,
        )

        if result.returncode != 0:
            return RalphCommandResult(
                False,
                (
                    "Ralph decompose failed.\n"
                    f"log: {decompose_log}\n"
                    f"stderr: {self._tail_text(result.stderr or result.stdout)}"
                ),
                {
                    "log_file": str(decompose_log),
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "exit_code": result.returncode,
                },
            )

        return RalphCommandResult(
            True,
            (
                "Ralph decompose completed.\n"
                f"log: {decompose_log}\n"
                f"stdout: {self._tail_text(result.stdout)}"
            ),
            {
                "log_file": str(decompose_log),
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.returncode,
            },
        )

    async def dry_run(
        self,
        *,
        spec_name: str | None = None,
        plan_file: str | None = None,
        log_dir: str | None = None,
        max_retry: int | None = None,
        delay: float | None = None,
        enable_git: bool = False,
        main_branch: str | None = None,
        work_branch: str | None = None,
        silent: bool = False,
    ) -> RalphCommandResult:
        paths = self._resolve_paths_for_execution(
            spec_name=spec_name,
            plan_file=plan_file,
            log_dir=log_dir,
        )
        check_error = self._check_loop_ready(paths)
        if check_error:
            return RalphCommandResult(False, check_error)

        command = self._build_loop_command(
            spec_name=spec_name,
            plan_file=str(paths.plan_file) if not spec_name else None,
            log_dir=str(paths.log_dir) if log_dir else None,
            max_retry=max_retry,
            delay=delay,
            enable_git=enable_git,
            main_branch=main_branch,
            work_branch=work_branch,
            silent=silent,
            dry_run=True,
        )

        result = await asyncio.to_thread(
            subprocess.run,
            command,
            cwd=str(self._workspace_root),
            capture_output=True,
            text=True,
            encoding="utf-8",
        )

        if result.returncode != 0:
            return RalphCommandResult(
                False,
                f"Ralph dry-run failed.\nstderr: {self._tail_text(result.stderr or result.stdout)}",
                {
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "exit_code": result.returncode,
                },
            )

        return RalphCommandResult(
            True,
            self._tail_text(result.stdout),
            {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.returncode,
            },
        )

    async def run(
        self,
        *,
        spec_name: str | None = None,
        plan_file: str | None = None,
        log_dir: str | None = None,
        max_retry: int | None = None,
        delay: float | None = None,
        enable_git: bool = False,
        main_branch: str | None = None,
        work_branch: str | None = None,
        silent: bool = False,
    ) -> RalphCommandResult:
        paths = self._resolve_paths_for_execution(
            spec_name=spec_name,
            plan_file=plan_file,
            log_dir=log_dir,
        )
        check_error = self._check_loop_ready(paths)
        if check_error:
            return RalphCommandResult(False, check_error)

        command = self._build_loop_command(
            spec_name=spec_name,
            plan_file=str(paths.plan_file) if not spec_name else None,
            log_dir=str(paths.log_dir) if log_dir else None,
            max_retry=max_retry,
            delay=delay,
            enable_git=enable_git,
            main_branch=main_branch,
            work_branch=work_branch,
            silent=silent,
            dry_run=False,
        )

        record = await asyncio.to_thread(
            self._process_manager.start_run,
            command=command,
            cwd=self._workspace_root,
            spec_name=paths.spec_name,
            plan_file=paths.plan_file,
            log_dir=paths.log_dir,
        )

        return RalphCommandResult(
            True,
            (
                "Ralph run started.\n"
                f"run_id: {record.run_id}\n"
                f"spec: {record.spec_name}\n"
                f"plan: {record.plan_file}\n"
                f"log_dir: {record.log_dir}"
            ),
            {"run_id": record.run_id, "record": record},
        )

    async def status(self, *, run_id: str | None = None) -> RalphCommandResult:
        if run_id:
            record = await asyncio.to_thread(self._process_manager.get_status, run_id)
            if record is None:
                return RalphCommandResult(False, f"Ralph run '{run_id}' not found.")
            return RalphCommandResult(True, self._format_record(record), {"record": record})

        records = await asyncio.to_thread(self._process_manager.list_runs)
        if not records:
            return RalphCommandResult(True, "No Ralph runs have been started in this workspace yet.")

        lines = ["Ralph runs:"]
        for record in records:
            summary = self._summarize_plan(Path(record.plan_file))
            lines.append(
                f"- {record.run_id} [{record.status}] spec={record.spec_name} "
                f"pid={record.pid or '-'} done={summary['done']} failed={summary['failed']} "
                f"todo={summary['todo']}"
            )
        return RalphCommandResult(True, "\n".join(lines), {"records": records})

    async def cancel(self, *, run_id: str) -> RalphCommandResult:
        record = await asyncio.to_thread(self._process_manager.cancel, run_id)
        if record is None:
            return RalphCommandResult(False, f"Ralph run '{run_id}' not found.")
        return RalphCommandResult(
            True,
            f"Ralph run '{run_id}' cancelled.\nstatus: {record.status}\npid: {record.pid}",
            {"record": record},
        )

    def _init_spec_sync(
        self,
        *,
        spec_name: str,
        target_dir: str,
        force: bool,
    ) -> RalphCommandResult:
        initializer = self._load_initializer()
        normalized_target_dir = self._normalize_target_dir(target_dir)
        with self._pushd(self._workspace_root):
            success = initializer.initialize_spec(
                target_dir=normalized_target_dir,
                spec_name=spec_name,
                force=force,
            )
        target_root = (
            self._workspace_root if normalized_target_dir == "." else Path(normalized_target_dir).resolve()
        )
        target_spec_dir = (
            target_root / "docs" / "spec" / spec_name
            if normalized_target_dir == "."
            else target_root / spec_name
        )
        if not success:
            return RalphCommandResult(False, f"Failed to initialize Ralph spec '{spec_name}'.")
        return RalphCommandResult(
            True,
            f"Ralph spec initialized at {target_spec_dir}",
            {"spec_dir": str(target_spec_dir)},
        )

    def _init_agent_sync(self, *, target_dir: str) -> RalphCommandResult:
        initializer = self._load_initializer()
        normalized_target_dir = self._normalize_target_dir(target_dir, keep_dot=False)
        success = initializer.initialize_agent_setting(
            target_dir=normalized_target_dir,
            agent_type="devagent",
            force=False,
        )
        target_root = Path(normalized_target_dir).resolve()
        if not success:
            return RalphCommandResult(False, "Failed to initialize .devagent settings.")
        return RalphCommandResult(
            True,
            f".devagent initialized at {target_root / '.devagent'}",
            {"devagent_dir": str(target_root / '.devagent')},
        )

    def _load_initializer(self) -> Any:
        spec = importlib.util.spec_from_file_location(
            "bu_agent_cli_ralph_init",
            self._ralph_init_script,
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load {self._ralph_init_script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.RalphInitializer()

    def _resolve_paths(self, spec_name: str) -> RalphPaths:
        spec_dir = self._workspace_root / "docs" / "spec" / spec_name
        return RalphPaths(
            workspace_root=self._workspace_root,
            spec_name=spec_name,
            spec_dir=spec_dir,
            requirement_dir=spec_dir / "requirement",
            plan_dir=spec_dir / "plan",
            plan_file=spec_dir / "plan" / "plan.json",
            implement_dir=spec_dir / "implement",
            log_dir=spec_dir / "logs",
        )

    def _resolve_paths_for_execution(
        self,
        *,
        spec_name: str | None,
        plan_file: str | None,
        log_dir: str | None,
    ) -> RalphPaths:
        if spec_name:
            paths = self._resolve_paths(spec_name)
            if log_dir:
                paths.log_dir = Path(log_dir).resolve()
            return paths

        if not plan_file:
            raise ValueError("spec_name or plan_file is required")

        plan_path = Path(plan_file).resolve()
        plan_dir_path = plan_path.parent
        spec_dir = plan_dir_path.parent
        resolved_log_dir = Path(log_dir).resolve() if log_dir else spec_dir / "logs"
        return RalphPaths(
            workspace_root=self._workspace_root,
            spec_name=spec_dir.name,
            spec_dir=spec_dir,
            requirement_dir=spec_dir / "requirement",
            plan_dir=plan_dir_path,
            plan_file=plan_path,
            implement_dir=spec_dir / "implement",
            log_dir=resolved_log_dir,
        )

    def _check_decompose_ready(self, paths: RalphPaths) -> str | None:
        if not paths.spec_dir.exists():
            return f"Spec directory not found: {paths.spec_dir}"
        if not paths.requirement_dir.exists():
            return f"Requirement directory not found: {paths.requirement_dir}"
        if not any(paths.requirement_dir.iterdir()):
            return f"Requirement directory is empty: {paths.requirement_dir}"
        if not self._decompose_prompt.exists():
            return f"Decompose prompt not found: {self._decompose_prompt}"
        if shutil.which("devagent") is None:
            return "devagent command not found in PATH."
        return None

    def _check_loop_ready(self, paths: RalphPaths) -> str | None:
        if not self._ralph_loop_script.exists():
            return f"Ralph loop script not found: {self._ralph_loop_script}"
        if not paths.plan_file.exists():
            return f"Plan file not found: {paths.plan_file}"
        if not self._implement_prompt.exists():
            return f"Implement prompt not found: {self._implement_prompt}"
        if shutil.which("devagent") is None:
            return "devagent command not found in PATH."
        return None

    def _build_loop_command(
        self,
        *,
        spec_name: str | None,
        plan_file: str | None,
        log_dir: str | None,
        max_retry: int | None,
        delay: float | None,
        enable_git: bool,
        main_branch: str | None,
        work_branch: str | None,
        silent: bool,
        dry_run: bool,
    ) -> list[str]:
        command = [self._python_executable, str(self._ralph_loop_script)]
        if spec_name:
            command.append(spec_name)
        elif plan_file:
            command.extend(["--plan-file", plan_file])

        if log_dir:
            command.extend(["--log-dir", log_dir])
        if max_retry is not None:
            command.extend(["--max-retry", str(max_retry)])
        if delay is not None:
            command.extend(["--delay", str(delay)])
        if enable_git:
            command.append("--enable-git")
        if main_branch:
            command.extend(["--main-branch", main_branch])
        if work_branch:
            command.extend(["--work-branch", work_branch])
        if silent:
            command.append("--silent")
        if dry_run:
            command.append("--dry-run")
        return command

    def _format_record(self, record: RalphRunRecord) -> str:
        summary = self._summarize_plan(Path(record.plan_file))
        lines = [
            f"run_id: {record.run_id}",
            f"status: {record.status}",
            f"spec: {record.spec_name}",
            f"pid: {record.pid or '-'}",
            f"started_at: {record.started_at}",
            f"ended_at: {record.ended_at or '-'}",
            f"exit_code: {record.exit_code if record.exit_code is not None else '-'}",
            f"plan_file: {record.plan_file}",
            f"log_dir: {record.log_dir}",
            f"stdout_log: {record.stdout_log}",
            f"stderr_log: {record.stderr_log}",
            (
                "plan_summary: "
                f"done={summary['done']} failed={summary['failed']} todo={summary['todo']} "
                f"other={summary['other']}"
            ),
        ]
        if record.error:
            lines.append(f"error: {record.error}")
        return "\n".join(lines)

    def _summarize_plan(self, plan_file: Path) -> dict[str, int]:
        if not plan_file.exists():
            return {"done": 0, "failed": 0, "todo": 0, "other": 0}
        try:
            payload = json.loads(plan_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {"done": 0, "failed": 0, "todo": 0, "other": 0}

        summary = {"done": 0, "failed": 0, "todo": 0, "other": 0}
        if not isinstance(payload, list):
            return summary
        for item in payload:
            if not isinstance(item, dict):
                summary["other"] += 1
                continue
            status = str(item.get("status", "")).upper()
            if status == "DONE":
                summary["done"] += 1
            elif status == "FAILED":
                summary["failed"] += 1
            elif status == "TODO":
                summary["todo"] += 1
            else:
                summary["other"] += 1
        return summary

    def _write_command_log(
        self,
        log_file: Path,
        *,
        command: str,
        stdout: str,
        stderr: str,
        return_code: int,
    ) -> None:
        log_file.write_text(
            "\n".join(
                [
                    f"command: {command}",
                    f"return_code: {return_code}",
                    "=== STDOUT ===",
                    stdout or "",
                    "=== STDERR ===",
                    stderr or "",
                ]
            ),
            encoding="utf-8",
        )

    def _tail_text(self, text: str, max_chars: int = 1600) -> str:
        normalized = (text or "").strip()
        if not normalized:
            return "(no output)"
        if len(normalized) <= max_chars:
            return normalized
        return "..." + normalized[-max_chars:]

    def _normalize_target_dir(self, target_dir: str, *, keep_dot: bool = True) -> str:
        target_dir = (target_dir or ".").strip()
        if target_dir == "." and keep_dot:
            return "."
        target_path = Path(target_dir)
        if not target_path.is_absolute():
            target_path = self._workspace_root / target_path
        return str(target_path.resolve())

    @contextlib.contextmanager
    def _pushd(self, path: Path):
        previous = Path.cwd()
        os.chdir(path)
        try:
            yield
        finally:
            os.chdir(previous)
