from __future__ import annotations

import asyncio
import locale
import os
import signal
import subprocess
import sys
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path


def shell_output_encodings() -> list[str]:
    encodings = ["utf-8", "utf-8-sig"]

    preferred = locale.getpreferredencoding(False)
    if preferred:
        encodings.append(preferred)

    if sys.platform == "win32":
        codepage = windows_console_encoding()
        if codepage:
            encodings.append(codepage)
        encodings.extend(["gb18030", "gbk", "cp936"])

    unique: list[str] = []
    seen: set[str] = set()
    for encoding in encodings:
        lowered = encoding.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        unique.append(encoding)
    return unique


def windows_console_encoding() -> str | None:
    if sys.platform != "win32":
        return None

    try:
        import ctypes

        codepage = ctypes.windll.kernel32.GetConsoleOutputCP()
    except Exception:
        return None

    if not codepage:
        return None
    return f"cp{codepage}"


def decode_process_stream(data: bytes | str | None) -> str:
    if data is None:
        return ""
    if isinstance(data, str):
        return data
    for encoding in shell_output_encodings():
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


async def terminate_process_tree(process: subprocess.Popen) -> None:
    if process.returncode is not None:
        return

    try:
        if sys.platform == "win32":
            await _terminate_windows_process_tree(process.pid)
        else:
            os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    try:
        await asyncio.wait_for(asyncio.to_thread(process.wait), timeout=1)
        return
    except (asyncio.TimeoutError, ProcessLookupError):
        pass
    except asyncio.CancelledError:
        process.kill()
        raise

    try:
        process.kill()
    except ProcessLookupError:
        return

    try:
        await asyncio.wait_for(asyncio.to_thread(process.wait), timeout=1)
    except (asyncio.TimeoutError, ProcessLookupError):
        pass


async def _terminate_windows_process_tree(pid: int | None) -> None:
    if not pid:
        return

    killer = await asyncio.create_subprocess_exec(
        "taskkill",
        "/PID",
        str(pid),
        "/T",
        "/F",
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await killer.wait()


@dataclass
class ShellTask:
    task_id: str
    command: str
    cwd: str
    log_path: Path
    process: subprocess.Popen
    log_handle: object
    created_at: float = field(default_factory=time.time)
    status: str = "running"
    returncode: int | None = None
    cancel_requested: bool = False
    completed_at: float | None = None
    watcher: asyncio.Task[None] | None = None

    @property
    def pid(self) -> int | None:
        return self.process.pid

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "command": self.command,
            "cwd": self.cwd,
            "pid": self.pid,
            "status": self.status,
            "returncode": self.returncode,
            "log_path": str(self.log_path),
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }


class ShellTaskManager:
    def __init__(self, workspace: Path, session_id: str):
        self._workspace = workspace
        self._session_id = session_id
        self._tasks_dir = workspace / ".tg_agent" / "shell_tasks" / session_id
        self._tasks: dict[str, ShellTask] = {}
        self._closed = False

    def _ensure_tasks_dir(self) -> Path:
        self._tasks_dir.mkdir(parents=True, exist_ok=True)
        return self._tasks_dir

    async def start(self, *, command: str, cwd: str) -> ShellTask:
        if self._closed:
            raise RuntimeError("Shell task manager is already closed")
        tasks_dir = self._ensure_tasks_dir()
        task_id = str(uuid.uuid4())[:8]
        log_path = tasks_dir / f"{task_id}.log"
        log_handle = open(log_path, "ab", buffering=0)

        popen_kwargs: dict[str, object] = {
            "shell": True,
            "cwd": cwd,
            "stdin": subprocess.DEVNULL,
            "stdout": log_handle,
            "stderr": subprocess.STDOUT,
        }
        if sys.platform == "win32":
            popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            popen_kwargs["start_new_session"] = True

        process = subprocess.Popen(command, **popen_kwargs)
        task = ShellTask(
            task_id=task_id,
            command=command,
            cwd=cwd,
            log_path=log_path,
            process=process,
            log_handle=log_handle,
        )
        self._tasks[task_id] = task
        task.watcher = asyncio.create_task(self._watch_task(task))
        return task

    async def _watch_task(self, task: ShellTask) -> None:
        try:
            returncode = await self._wait_for_process(task.process)
            task.returncode = returncode
            task.completed_at = time.time()
            if task.cancel_requested:
                task.status = "cancelled"
            elif returncode == 0:
                task.status = "completed"
            else:
                task.status = "failed"
        except asyncio.CancelledError:
            return
        finally:
            with suppress(Exception):
                task.log_handle.close()

    async def _wait_for_process(
        self,
        process: subprocess.Popen,
        *,
        poll_interval: float = 0.2,
    ) -> int:
        while True:
            returncode = process.poll()
            if returncode is not None:
                return returncode
            await asyncio.sleep(poll_interval)

    def get_task(self, task_id: str) -> ShellTask | None:
        return self._tasks.get(task_id)

    def list_tasks(self) -> list[ShellTask]:
        return sorted(self._tasks.values(), key=lambda item: item.created_at)

    async def cancel(self, task_id: str) -> str:
        task = self.get_task(task_id)
        if task is None:
            return f"Error: Task '{task_id}' not found"
        if task.status in {"completed", "failed", "cancelled"}:
            return f"Task '{task_id}' is already {task.status}"

        task.cancel_requested = True
        await terminate_process_tree(task.process)
        with suppress(Exception):
            await asyncio.wait_for(asyncio.to_thread(task.process.wait), timeout=2)
        task.status = "cancelled"
        task.completed_at = time.time()
        task.returncode = task.process.returncode
        return f"Cancelled shell task '{task_id}'"

    async def shutdown(self, *, cancel_running: bool = True) -> None:
        self._closed = True
        tasks = list(self._tasks.values())

        if cancel_running:
            for task in tasks:
                if task.status == "running":
                    task.cancel_requested = True
                    await terminate_process_tree(task.process)
                    task.status = "cancelled"
                    task.completed_at = time.time()
                    task.returncode = task.process.returncode

        for task in tasks:
            if task.watcher is not None and not task.watcher.done():
                task.watcher.cancel()
                with suppress(asyncio.CancelledError):
                    await task.watcher
            with suppress(Exception):
                task.log_handle.close()

    def read_output(self, task_id: str, *, max_chars: int = 4000) -> str:
        task = self.get_task(task_id)
        if task is None:
            raise KeyError(task_id)

        try:
            data = task.log_path.read_bytes()
        except FileNotFoundError:
            return ""

        text = decode_process_stream(data)
        if len(text) <= max_chars:
            return text
        return text[-max_chars:]

    async def wait_for_output(
        self,
        task_id: str,
        *,
        pattern: str,
        timeout: float,
        poll_interval: float = 0.2,
        max_chars: int = 12000,
    ) -> tuple[str, bool]:
        deadline = time.monotonic() + timeout
        while True:
            output = self.read_output(task_id, max_chars=max_chars)
            if pattern in output:
                return output, True

            task = self.get_task(task_id)
            if task is None:
                return output, False
            if task.status in {"completed", "failed", "cancelled"}:
                return output, False
            if time.monotonic() >= deadline:
                return output, False
            await asyncio.sleep(poll_interval)
