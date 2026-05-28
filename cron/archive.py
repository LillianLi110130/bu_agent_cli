"""Archive scheduled job outputs."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path

from agent_core.runtime_paths import tg_agent_home
from cron.models import CronJob, CronRun, datetime_to_iso


@dataclass(slots=True)
class CronArchiveWriter:
    base_dir: Path | None = None

    @property
    def output_dir(self) -> Path:
        return (self.base_dir or tg_agent_home() / "cron") / "output"

    def write(
        self,
        *,
        job: CronJob,
        run: CronRun,
        output: str = "",
        error: str | None = None,
    ) -> Path:
        archive_path = self.output_dir / job.id / f"{run.run_id}.md"
        content = self._render(job=job, run=run, output=output, error=error)
        self._write_text_atomic(archive_path, content)
        return archive_path

    @staticmethod
    def _render(*, job: CronJob, run: CronRun, output: str, error: str | None) -> str:
        error_text = error or ""
        lines = [
            "# Cron Job Output",
            "",
            f"- Job ID: {job.id}",
            f"- Run ID: {run.run_id}",
            f"- Job Name: {job.name}",
            f"- Scheduled At: {datetime_to_iso(run.scheduled_at)}",
            f"- Claimed At: {datetime_to_iso(run.claimed_at)}",
            f"- Started At: {datetime_to_iso(run.started_at)}",
            f"- Finished At: {datetime_to_iso(run.finished_at)}",
            f"- Status: {run.status}",
            f"- Execution Mode: {run.execution_mode}",
            f"- Delivery Mode: {job.delivery}",
            "",
            "## Prompt",
            "",
            job.prompt,
            "",
            "## Output",
            "",
            output,
            "",
            "## Error",
            "",
            error_text,
            "",
        ]
        return "\n".join(lines)

    @staticmethod
    def _write_text_atomic(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}.{threading.get_ident()}")
        with temp_path.open("w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
