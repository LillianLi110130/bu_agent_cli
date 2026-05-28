"""Cross-process lock helpers for cron scheduler ticks."""

from __future__ import annotations

import time
from pathlib import Path


class CronTickLock:
    """Directory-backed non-blocking lock for scheduler ticks."""

    def __init__(self, lock_path: Path, *, timeout_seconds: float = 0.0) -> None:
        self.lock_path = lock_path
        self.timeout_seconds = timeout_seconds
        self.acquired = False

    def __enter__(self) -> "CronTickLock":
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            try:
                self.lock_path.mkdir(parents=True, exist_ok=False)
                self.acquired = True
                return self
            except FileExistsError:
                if self.timeout_seconds <= 0 or time.monotonic() >= deadline:
                    self.acquired = False
                    return self
                time.sleep(0.05)

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self.acquired:
            return
        self.acquired = False
        try:
            self.lock_path.rmdir()
        except FileNotFoundError:
            return
