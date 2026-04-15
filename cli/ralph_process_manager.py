from __future__ import annotations

import json
import os
import subprocess
import threading
import uuid
from pathlib import Path

from cli.ralph_models import RalphRunRecord, now_iso


class RalphProcessManager:
    """Manage long-running Ralph loop processes."""

    def __init__(self, workspace_root: Path | None):
        resolved_root = workspace_root.resolve() if workspace_root is not None else Path.cwd().resolve()
        self._workspace_root = resolved_root
        self._state_dir = self._workspace_root / ".tg_agent" / "ralph"
        self._log_dir = self._state_dir / "process_logs"
        self._state_file = self._state_dir / "runs.json"
        self._lock = threading.Lock()
        self._processes: dict[str, subprocess.Popen[str]] = {}
        self._records: dict[str, RalphRunRecord] = {}
        self._ensure_state_dirs()
        self._load_state()

    def start_run(
        self,
        *,
        command: list[str],
        cwd: Path,
        spec_name: str,
        plan_file: Path,
        log_dir: Path,
    ) -> RalphRunRecord:
        self._ensure_state_dirs()
        run_id = uuid.uuid4().hex[:12]
        stdout_log = self._log_dir / f"{run_id}.stdout.log"
        stderr_log = self._log_dir / f"{run_id}.stderr.log"

        with stdout_log.open("w", encoding="utf-8") as stdout_handle, stderr_log.open(
            "w", encoding="utf-8"
        ) as stderr_handle:
            process = subprocess.Popen(
                command,
                cwd=str(cwd),
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
            )

        record = RalphRunRecord(
            run_id=run_id,
            spec_name=spec_name,
            status="running",
            pid=process.pid,
            command=command,
            cwd=str(cwd),
            plan_file=str(plan_file),
            log_dir=str(log_dir),
            stdout_log=str(stdout_log),
            stderr_log=str(stderr_log),
            created_at=now_iso(),
            started_at=now_iso(),
        )

        with self._lock:
            self._processes[run_id] = process
            self._records[run_id] = record
            self._save_state_locked()

        watcher = threading.Thread(
            target=self._watch_process,
            args=(run_id,),
            daemon=True,
        )
        watcher.start()
        return record

    def get_status(self, run_id: str) -> RalphRunRecord | None:
        with self._lock:
            record = self._records.get(run_id)
        if not record:
            return None
        self._refresh_record(run_id)
        with self._lock:
            return self._records.get(run_id)

    def list_runs(self) -> list[RalphRunRecord]:
        run_ids = list(self._records.keys())
        for run_id in run_ids:
            self._refresh_record(run_id)
        with self._lock:
            return sorted(
                self._records.values(),
                key=lambda item: item.started_at,
                reverse=True,
            )

    def cancel(self, run_id: str) -> RalphRunRecord | None:
        with self._lock:
            record = self._records.get(run_id)
            process = self._processes.get(run_id)
        if not record:
            return None

        if record.pid and self._is_pid_running(record.pid):
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(record.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            elif process is not None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()

        with self._lock:
            record.status = "cancelled"
            record.ended_at = now_iso()
            if process is not None:
                record.exit_code = process.poll()
                self._processes.pop(run_id, None)
            self._records[run_id] = record
            self._save_state_locked()
            return record

    def _watch_process(self, run_id: str) -> None:
        with self._lock:
            process = self._processes.get(run_id)
        if process is None:
            return

        return_code = process.wait()
        with self._lock:
            record = self._records.get(run_id)
            if record is None:
                return
            record.exit_code = return_code
            if record.status != "cancelled":
                record.status = "completed" if return_code == 0 else "failed"
            if record.ended_at is None:
                record.ended_at = now_iso()
            self._processes.pop(run_id, None)
            self._records[run_id] = record
            self._save_state_locked()

    def _refresh_record(self, run_id: str) -> None:
        with self._lock:
            record = self._records.get(run_id)
            process = self._processes.get(run_id)
        if record is None:
            return

        if process is not None:
            return_code = process.poll()
            if return_code is None:
                return
            with self._lock:
                record.exit_code = return_code
                if record.status != "cancelled":
                    record.status = "completed" if return_code == 0 else "failed"
                if record.ended_at is None:
                    record.ended_at = now_iso()
                self._processes.pop(run_id, None)
                self._records[run_id] = record
                self._save_state_locked()
            return

        if record.status == "running" and record.pid and not self._is_pid_running(record.pid):
            with self._lock:
                record.status = "finished"
                if record.ended_at is None:
                    record.ended_at = now_iso()
                self._records[run_id] = record
                self._save_state_locked()

    def _load_state(self) -> None:
        if not self._state_file.exists():
            return
        try:
            raw = json.loads(self._state_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return

        if not isinstance(raw, list):
            return

        for item in raw:
            if not isinstance(item, dict):
                continue
            record = RalphRunRecord.from_dict(item)
            if record.status == "running" and record.pid and not self._is_pid_running(record.pid):
                record.status = "finished"
                if record.ended_at is None:
                    record.ended_at = now_iso()
            self._records[record.run_id] = record

    def _save_state_locked(self) -> None:
        payload = [record.to_dict() for record in self._records.values()]
        self._state_file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _ensure_state_dirs(self) -> None:
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._log_dir.mkdir(parents=True, exist_ok=True)

    def _is_pid_running(self, pid: int) -> bool:
        if pid <= 0:
            return False

        if os.name == "nt":
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            return str(pid) in result.stdout

        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

