"""Cron job parsing and persistent store."""

from __future__ import annotations

import json
import logging
import os
import re
import secrets
import shutil
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from croniter import croniter

from agent_core.runtime_paths import tg_agent_home
from cron.models import (
    CronDelivery,
    CronExecution,
    CronExecutionMode,
    CronJob,
    CronRepeat,
    CronSchedule,
    CronSource,
)

logger = logging.getLogger("cron.jobs")

STORE_VERSION = 1
_DURATION_RE = re.compile(r"^(?P<count>\d+)(?P<unit>[mhd])$")
_INVISIBLE_CONTROL_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2066-\u2069]")
_STORE_LOCK = threading.Lock()


class CronValidationError(ValueError):
    """Raised when a cron job definition is invalid."""


def cron_home() -> Path:
    return tg_agent_home() / "cron"


def utc_now() -> datetime:
    return datetime.now(timezone.utc).astimezone()


def local_timezone_label(now: datetime | None = None) -> str:
    current = now or utc_now()
    name = current.tzname()
    if name:
        return name
    offset = current.utcoffset() or timedelta(0)
    total_seconds = int(offset.total_seconds())
    sign = "+" if total_seconds >= 0 else "-"
    total_seconds = abs(total_seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes = remainder // 60
    return f"UTC{sign}{hours:02d}:{minutes:02d}"


def parse_schedule(text: str, *, now: datetime | None = None) -> CronSchedule:
    """Parse supported schedule expressions into a structured schedule."""
    raw = text.strip()
    if not raw:
        raise CronValidationError("schedule 不能为空")

    base_now = ensure_aware(now or utc_now())
    timezone_label = local_timezone_label(base_now)
    lowered = raw.lower()

    if lowered.startswith("every "):
        seconds = _parse_duration_seconds(lowered.removeprefix("every ").strip())
        return CronSchedule(
            kind="interval",
            seconds=seconds,
            display=raw,
            timezone=timezone_label,
        )

    relative_seconds = _try_parse_duration_seconds(lowered)
    if relative_seconds is not None:
        return CronSchedule(
            kind="once",
            run_at=base_now + timedelta(seconds=relative_seconds),
            display=raw,
            timezone=timezone_label,
        )

    iso_run_at = _try_parse_iso_datetime(raw, base_now)
    if iso_run_at is not None:
        return CronSchedule(
            kind="once",
            run_at=iso_run_at,
            display=raw,
            timezone=local_timezone_label(iso_run_at),
        )

    if _looks_like_cron(raw):
        if not croniter.is_valid(raw):
            raise CronValidationError(f"无效 cron 表达式：{raw}")
        return CronSchedule(
            kind="cron",
            expr=raw,
            display=raw,
            timezone=timezone_label,
        )

    raise CronValidationError(f"不支持的 schedule 表达式：{raw}")


def compute_next_run(
    schedule: CronSchedule,
    *,
    now: datetime | None = None,
    last_run_at: datetime | None = None,
) -> datetime:
    """Compute the next timezone-aware run time for a schedule."""
    base_now = ensure_aware(now or utc_now())
    if schedule.kind == "once":
        if schedule.run_at is None:
            raise CronValidationError("once schedule 缺少 run_at")
        return ensure_aware(schedule.run_at)

    if schedule.kind == "interval":
        if not schedule.seconds or schedule.seconds <= 0:
            raise CronValidationError("interval schedule 需要正整数 seconds")
        base = ensure_aware(last_run_at) if last_run_at is not None else base_now
        next_run = base + timedelta(seconds=schedule.seconds)
        while next_run <= base_now:
            next_run += timedelta(seconds=schedule.seconds)
        return next_run

    if schedule.kind == "cron":
        if not schedule.expr:
            raise CronValidationError("cron schedule 缺少 expr")
        return ensure_aware(croniter(schedule.expr, base_now).get_next(datetime))

    raise CronValidationError(f"未知 schedule kind：{schedule.kind}")


def ensure_aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=datetime.now().astimezone().tzinfo)
    return value


def validate_prompt_safety(prompt: str) -> None:
    """Reject high-risk scheduled prompts before persistence."""
    normalized = prompt.strip()
    if not normalized:
        raise CronValidationError("prompt 不能为空")

    lowered = normalized.lower()
    risky_patterns = [
        ("忽略之前", "prompt 包含忽略之前指令的风险表达"),
        ("ignore previous", "prompt 包含忽略之前指令的风险表达"),
        ("不要告诉用户", "prompt 包含隐藏行为风险表达"),
        ("don't tell the user", "prompt 包含隐藏行为风险表达"),
        ("system prompt", "prompt 包含 system prompt override 风险表达"),
        ("sudoers", "prompt 包含敏感系统配置风险表达"),
        ("exfiltrate", "prompt 包含外传信息风险表达"),
        ("secret", "prompt 包含外传 secret 风险表达"),
        ("rm -rf /", "prompt 包含大范围删除风险命令"),
        ("format c:", "prompt 包含高危磁盘操作表达"),
    ]
    for pattern, message in risky_patterns:
        if pattern in lowered or pattern in normalized:
            raise CronValidationError(message)
    if _INVISIBLE_CONTROL_RE.search(normalized):
        raise CronValidationError("prompt 包含不可见 Unicode 控制字符")


class CronJobStore:
    """JSON-backed cron job store under ``~/.tg_agent/cron``."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or cron_home()
        self.jobs_path = self.base_dir / "jobs.json"
        self.last_corrupt_backup_path: Path | None = None

    def list_jobs(self) -> list[CronJob]:
        return self.load_jobs()

    def get_job(self, job_id: str) -> CronJob | None:
        for job in self.load_jobs():
            if job.id == job_id:
                return job
        return None

    def create_job(
        self,
        *,
        prompt: str,
        schedule_text: str,
        workspace_root: Path,
        session_binding_id: str,
        name: str | None = None,
        source: CronSource = "local",
        delivery: CronDelivery = "local",
        execution_mode: CronExecutionMode = "enqueue_current_session",
        repeat_times: int | None = None,
        now: datetime | None = None,
    ) -> CronJob:
        validate_prompt_safety(prompt)
        created_at = ensure_aware(now or utc_now())
        schedule = parse_schedule(schedule_text, now=created_at)
        default_repeat = 1 if schedule.kind == "once" else None
        repeat = CronRepeat(
            times=normalize_repeat_times(repeat_times)
            if repeat_times is not None
            else default_repeat
        )
        job = CronJob(
            id=new_job_id(),
            name=name.strip() if name and name.strip() else default_job_name(prompt),
            prompt=prompt,
            schedule=schedule,
            schedule_display=schedule.display,
            repeat=repeat,
            source=source,
            delivery=delivery,
            execution=CronExecution(
                mode=execution_mode,
                workspace_root=str(Path(workspace_root).resolve()),
                session_binding_id=session_binding_id,
            ),
            enabled=True,
            state="scheduled",
            next_run_at=compute_next_run(schedule, now=created_at),
            created_at=created_at,
            updated_at=created_at,
        )

        def add(jobs: list[CronJob]) -> list[CronJob]:
            jobs.append(job)
            return jobs

        self.modify_jobs(add)
        return job

    def update_job(self, job_id: str, updater: Callable[[CronJob], CronJob]) -> CronJob | None:
        updated_job: CronJob | None = None

        def update(jobs: list[CronJob]) -> list[CronJob]:
            nonlocal updated_job
            next_jobs: list[CronJob] = []
            for job in jobs:
                if job.id != job_id:
                    next_jobs.append(job)
                    continue
                updated_job = updater(job)
                next_jobs.append(updated_job)
            return next_jobs

        self.modify_jobs(update)
        return updated_job

    def remove_job(self, job_id: str) -> bool:
        removed = False

        def remove(jobs: list[CronJob]) -> list[CronJob]:
            nonlocal removed
            kept = [job for job in jobs if job.id != job_id]
            removed = len(kept) != len(jobs)
            return kept

        self.modify_jobs(remove)
        return removed

    def modify_jobs(self, modifier: Callable[[list[CronJob]], list[CronJob]]) -> list[CronJob]:
        with _STORE_LOCK:
            jobs = self.load_jobs()
            updated_jobs = modifier(jobs)
            self.save_jobs(updated_jobs)
            return updated_jobs

    def load_jobs(self) -> list[CronJob]:
        self.last_corrupt_backup_path = None
        if not self.jobs_path.exists():
            return []
        try:
            payload = json.loads(self.jobs_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            self.last_corrupt_backup_path = self._backup_corrupt_jobs_file()
            logger.warning(
                f"Cron jobs file is corrupt; backed up to {self.last_corrupt_backup_path}"
            )
            return []
        raw_jobs = payload.get("jobs", [])
        if not isinstance(raw_jobs, list):
            logger.warning(f"Cron jobs file has invalid jobs payload: {self.jobs_path}")
            return []
        return [CronJob.from_dict(item) for item in raw_jobs]

    def save_jobs(self, jobs: list[CronJob]) -> None:
        payload = {
            "version": STORE_VERSION,
            "jobs": [job.to_dict() for job in jobs],
        }
        self._write_json_atomic(self.jobs_path, payload)

    def _backup_corrupt_jobs_file(self) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_path = self.jobs_path.with_name(f"jobs.json.corrupt.{timestamp}")
        counter = 1
        while backup_path.exists():
            backup_path = self.jobs_path.with_name(f"jobs.json.corrupt.{timestamp}.{counter}")
            counter += 1
        shutil.copyfile(self.jobs_path, backup_path)
        return backup_path

    @staticmethod
    def _write_json_atomic(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}.{threading.get_ident()}")
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)


def new_job_id() -> str:
    return f"cron_{secrets.token_hex(5)}"


def default_job_name(prompt: str) -> str:
    stripped = " ".join(prompt.strip().split())
    return stripped[:40] or "Scheduled Job"


def normalize_repeat_times(repeat_times: int | None) -> int | None:
    if repeat_times == 0:
        return None
    return repeat_times


def _parse_duration_seconds(text: str) -> int:
    seconds = _try_parse_duration_seconds(text)
    if seconds is None:
        raise CronValidationError(f"无效时间间隔：{text}")
    return seconds


def _try_parse_duration_seconds(text: str) -> int | None:
    match = _DURATION_RE.match(text.strip())
    if match is None:
        return None
    count = int(match.group("count"))
    unit = match.group("unit")
    multiplier = {"m": 60, "h": 3600, "d": 86400}[unit]
    return count * multiplier


def _try_parse_iso_datetime(text: str, now: datetime) -> datetime | None:
    normalized = text.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=now.tzinfo)
    return parsed


def _looks_like_cron(text: str) -> bool:
    return len(text.split()) == 5
