"""Directory based locks for cross-process team coordination."""

from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from agent_core.team.atomic_io import atomic_write_json, read_json


class FileLockTimeout(TimeoutError):
    """Raised when a filesystem lock cannot be acquired before timeout."""


@dataclass(slots=True)
class FileLock:
    """A small mkdir-based lock with stale-lock cleanup via TTL."""

    path: Path
    owner: str
    ttl_sec: float = 300.0
    poll_interval: float = 0.05
    acquired: bool = False

    def acquire(self, timeout: float = 10.0) -> "FileLock":
        deadline = time.monotonic() + timeout
        while True:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.path.mkdir()
                self.acquired = True
                self._write_owner()
                return self
            except FileExistsError:
                self._cleanup_if_stale()
                if time.monotonic() >= deadline:
                    raise FileLockTimeout(f"Timed out acquiring lock: {self.path}")
                time.sleep(self.poll_interval)

    def release(self) -> None:
        if not self.acquired:
            return
        shutil.rmtree(self.path, ignore_errors=True)
        self.acquired = False

    def __enter__(self) -> "FileLock":
        return self.acquire()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()

    def _write_owner(self) -> None:
        now = time.time()
        atomic_write_json(
            self.path / "owner.json",
            {
                "owner": self.owner,
                "pid": os.getpid(),
                "created_at": datetime.fromtimestamp(now, timezone.utc).isoformat(),
                "expires_at": datetime.fromtimestamp(now + self.ttl_sec, timezone.utc).isoformat(),
            },
        )

    def _cleanup_if_stale(self) -> None:
        owner_path = self.path / "owner.json"
        payload = read_json(owner_path, {})
        expires_at = payload.get("expires_at") if isinstance(payload, dict) else None
        if not isinstance(expires_at, str):
            return
        try:
            expires = datetime.fromisoformat(expires_at)
        except ValueError:
            return
        if expires <= datetime.now(timezone.utc):
            shutil.rmtree(self.path, ignore_errors=True)
