"""SQLite-backed queue storage for the local IM bridge."""

from __future__ import annotations

import json
import logging
import re
import shutil
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterator

from cli.im_bridge.models import (
    BridgeOutboundEvent,
    BridgeProgress,
    BridgeRequest,
    BridgeResult,
    classify_input_kind,
    from_iso8601,
    normalize_bridge_source,
    to_iso8601,
    utc_now,
)

logger = logging.getLogger("cli.im_bridge.store")


class SqliteBridgeStore:
    """Manage worker-scoped bridge queues in one workspace SQLite database."""

    VERSION = 1
    DEFAULT_PROCESSING_TIMEOUT_SECONDS = 24 * 60 * 60
    WORKER_TOUCH_INTERVAL_SECONDS = 60.0
    LEASE_RECOVERY_INTERVAL_SECONDS = 60.0
    DEFAULT_WORKER_LOG_RETENTION_SECONDS = 7 * 24 * 60 * 60
    DEFAULT_MAX_WORKER_LOG_DIRS_PER_CLEANUP = 100

    def __init__(
        self,
        root_dir: Path | str | None = None,
        session_binding_id: str = "local-cli",
        *,
        processing_timeout_seconds: float = DEFAULT_PROCESSING_TIMEOUT_SECONDS,
        worker_log_retention_seconds: float = DEFAULT_WORKER_LOG_RETENTION_SECONDS,
        max_worker_log_dirs_per_cleanup: int = DEFAULT_MAX_WORKER_LOG_DIRS_PER_CLEANUP,
    ) -> None:
        resolved_root = Path(root_dir).resolve() if root_dir is not None else Path.cwd().resolve()
        self.workspace_root = resolved_root
        self.session_binding_id = session_binding_id
        self.worker_no = session_binding_id
        self.processing_timeout_seconds = processing_timeout_seconds
        self.worker_log_retention_seconds = worker_log_retention_seconds
        self.max_worker_log_dirs_per_cleanup = max_worker_log_dirs_per_cleanup
        self.bridge_dir = self.workspace_root / ".tg_agent" / "im_bridge"
        self.db_path = self.bridge_dir / "bridge.sqlite3"
        self.migration_path = self.bridge_dir / "migration.json"
        self.legacy_cleanup_lock_dir = self.bridge_dir / "legacy-cleanup.lock"
        self.worker_logs_root = self.workspace_root / ".tg_agent" / "logs" / "workers"
        self.logs_dir = self.worker_logs_root / self.session_binding_id
        self.worker_logs_cleanup_lock_dir = self.worker_logs_root / ".cleanup.lock"
        self._initialize_lock = threading.Lock()
        self._initialized = False
        self._last_worker_touch_monotonic: float | None = None
        self._last_lease_recovery_monotonic: float | None = None

    def initialize(self) -> None:
        """Initialize the shared database and discard obsolete file-backed queues."""
        if self._initialized:
            return
        with self._initialize_lock:
            if self._initialized:
                return
            self.bridge_dir.mkdir(parents=True, exist_ok=True)
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            self._cleanup_old_worker_log_dirs()
            self._cleanup_legacy_worker_dirs()
            with self._connection() as conn:
                conn.execute("PRAGMA journal_mode = WAL")
                self._initialize_schema(conn)
                self._ensure_worker(conn)
                self._recover_expired_leases(conn)
            self._initialized = True
            self._last_worker_touch_monotonic = time.monotonic()
            self._last_lease_recovery_monotonic = time.monotonic()

    def enqueue_text(
        self,
        content: str,
        *,
        source: str = "local",
        source_meta: dict[str, Any] | None = None,
        remote_response_required: bool = False,
        request_id: str | None = None,
        input_kind: str | None = None,
    ) -> BridgeRequest:
        """Append one text request to the worker's pending queue."""
        self.initialize()
        source_meta = dict(source_meta or {})
        normalized_source = normalize_bridge_source(source, source_meta=source_meta)
        if normalized_source in {"im", "web"} and not source_meta.get("origin"):
            source_meta["origin"] = normalized_source
        request = BridgeRequest(
            version=self.VERSION,
            request_id=request_id or f"req_{uuid.uuid4().hex}",
            seq=0,
            source=normalized_source,
            source_meta=source_meta,
            content=content,
            input_kind=input_kind or classify_input_kind(content),
            content_type="text",
            remote_response_required=remote_response_required,
            status="pending",
        )
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            seq = self._take_next_sequence(conn, "next_request_seq")
            request.seq = seq
            conn.execute(
                """
                INSERT INTO bridge_requests (
                    request_id, worker_no, seq, source, source_meta_json, content,
                    input_kind, content_type, remote_response_required, status, enqueue_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request.request_id,
                    self.worker_no,
                    seq,
                    request.source,
                    json.dumps(request.source_meta, ensure_ascii=False),
                    request.content,
                    request.input_kind,
                    request.content_type,
                    int(request.remote_response_required),
                    request.status,
                    to_iso8601(request.enqueue_time),
                ),
            )
        return request

    def claim_next_pending(self) -> BridgeRequest | None:
        """Claim the lowest-sequence pending request for this worker."""
        self.initialize()
        now = utc_now()
        with self._connection() as conn:
            self._recover_expired_leases_if_due(conn, now=now)
            row = conn.execute(
                """
                SELECT *
                FROM bridge_requests
                WHERE worker_no = ? AND status = 'pending'
                ORDER BY seq
                LIMIT 1
                """,
                (self.worker_no,),
            ).fetchone()
            if row is None:
                return None
            started_at = to_iso8601(now)
            lease_until = to_iso8601(now + timedelta(seconds=self.processing_timeout_seconds))
            cursor = conn.execute(
                """
                UPDATE bridge_requests
                SET status = 'running', started_at = ?, lease_until = ?
                WHERE request_id = ? AND status = 'pending'
                """,
                (started_at, lease_until, row["request_id"]),
            )
            if cursor.rowcount != 1:
                return None
            claimed = dict(row)
            claimed["status"] = "running"
            claimed["started_at"] = started_at
        return self._row_to_request(claimed)

    def complete_request(self, request: BridgeRequest, final_content: str = "") -> BridgeResult:
        """Persist a completed result for a running request."""
        result = self._make_result(request, final_status="completed", final_content=final_content)
        self._finalize_request(result)
        return result

    def fail_request(
        self,
        request: BridgeRequest,
        *,
        final_content: str,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> BridgeResult:
        """Persist a failed result for a running request."""
        result = self._make_result(
            request,
            final_status="failed",
            final_content=final_content,
            error_code=error_code,
            error_message=error_message,
        )
        self._finalize_request(result)
        return result

    def record_progress(self, request: BridgeRequest, content: str) -> BridgeProgress:
        """Persist one intermediate plain-text response for a running request."""
        self.initialize()
        progress = BridgeProgress(
            version=self.VERSION,
            request_id=request.request_id,
            seq=request.seq,
            source=request.source,
            progress_id=f"progress_{time.time_ns()}_{uuid.uuid4().hex}",
            content=content,
        )
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO bridge_progress (
                    progress_id, request_id, seq, source, content, status, created_at
                ) VALUES (?, ?, ?, ?, ?, 'pending', ?)
                """,
                (
                    progress.progress_id,
                    progress.request_id,
                    progress.seq,
                    progress.source,
                    progress.content,
                    to_iso8601(progress.created_at),
                ),
            )
        return progress

    def list_progress(self, request_id: str) -> list[BridgeProgress]:
        """Return undelivered progress items in creation order."""
        self.initialize()
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM bridge_progress
                WHERE request_id = ? AND status = 'pending'
                ORDER BY created_at, progress_id
                """,
                (request_id,),
            ).fetchall()
        return [self._row_to_progress(row) for row in rows]

    def complete_progress(self, progress: BridgeProgress) -> None:
        """Mark one intermediate progress item as delivered."""
        self.initialize()
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE bridge_progress
                SET status = 'delivered', delivered_at = ?
                WHERE progress_id = ?
                """,
                (to_iso8601(utc_now()), progress.progress_id),
            )

    def find_result(self, request_id: str) -> BridgeResult | None:
        """Return a completed or failed result by request id."""
        self.initialize()
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM bridge_requests
                WHERE request_id = ? AND status IN ('completed', 'failed')
                """,
                (request_id,),
            ).fetchone()
        return self._row_to_result(row) if row is not None else None

    def has_request(self, request_id: str) -> bool:
        """Return whether a request already exists."""
        self.initialize()
        with self._connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM bridge_requests WHERE request_id = ?",
                (request_id,),
            ).fetchone()
        return row is not None

    def pending_count(self) -> int:
        """Return the number of queued pending requests for this worker."""
        self.initialize()
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS count
                FROM bridge_requests
                WHERE worker_no = ? AND status = 'pending'
                """,
                (self.worker_no,),
            ).fetchone()
        return int(row["count"])

    def enqueue_outbound_text(self, text: str) -> BridgeOutboundEvent:
        """Append one outbound text event to this worker's pending outbox."""
        return self._enqueue_outbound_event(action="text", text=text)

    def enqueue_outbound_attachment(
        self,
        *,
        file_path: str,
        file_name: str,
        mime_type: str,
        file_size: int,
    ) -> BridgeOutboundEvent:
        """Append one outbound attachment event to this worker's pending outbox."""
        return self._enqueue_outbound_event(
            action="attachment",
            file_path=file_path,
            file_name=file_name,
            mime_type=mime_type,
            file_size=file_size,
        )

    def claim_next_pending_outbound(self) -> BridgeOutboundEvent | None:
        """Claim the lowest-sequence outbound event for this worker."""
        self.initialize()
        now = utc_now()
        with self._connection() as conn:
            self._recover_expired_leases_if_due(conn, now=now)
            row = conn.execute(
                """
                SELECT *
                FROM bridge_outbound_events
                WHERE worker_no = ? AND status = 'pending'
                ORDER BY seq
                LIMIT 1
                """,
                (self.worker_no,),
            ).fetchone()
            if row is None:
                return None
            started_at = to_iso8601(now)
            lease_until = to_iso8601(now + timedelta(seconds=self.processing_timeout_seconds))
            cursor = conn.execute(
                """
                UPDATE bridge_outbound_events
                SET status = 'running', started_at = ?, lease_until = ?, attempts = attempts + 1
                WHERE event_id = ? AND status = 'pending'
                """,
                (started_at, lease_until, row["event_id"]),
            )
            if cursor.rowcount != 1:
                return None
            claimed = dict(row)
            claimed["status"] = "running"
            claimed["started_at"] = started_at
        return self._row_to_outbound_event(claimed)

    def complete_outbound_event(self, event: BridgeOutboundEvent) -> None:
        """Mark one outbound event as delivered."""
        self.initialize()
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE bridge_outbound_events
                SET status = 'delivered', delivered_at = ?, lease_until = NULL
                WHERE event_id = ?
                """,
                (to_iso8601(utc_now()), event.event_id),
            )

    def touch_worker(self, *, force: bool = False) -> None:
        """Refresh worker activity at most once per minute unless forced."""
        self.initialize()
        now_monotonic = time.monotonic()
        if (
            not force
            and self._last_worker_touch_monotonic is not None
            and now_monotonic - self._last_worker_touch_monotonic
            < self.WORKER_TOUCH_INTERVAL_SECONDS
        ):
            return
        with self._connection() as conn:
            self._ensure_worker(conn, now=utc_now())
        self._last_worker_touch_monotonic = now_monotonic

    def recover_expired_leases(self) -> None:
        """Return expired running work to pending state."""
        self.initialize()
        with self._connection() as conn:
            self._recover_expired_leases(conn)

    def _enqueue_outbound_event(
        self,
        *,
        action: str,
        text: str = "",
        file_path: str = "",
        file_name: str = "",
        mime_type: str = "",
        file_size: int = 0,
    ) -> BridgeOutboundEvent:
        self.initialize()
        event = BridgeOutboundEvent(
            version=self.VERSION,
            event_id=f"evt_{uuid.uuid4().hex}",
            seq=0,
            action=action,
            text=text,
            file_path=file_path,
            file_name=file_name,
            mime_type=mime_type,
            file_size=file_size,
        )
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            seq = self._take_next_sequence(conn, "next_outbound_seq")
            event.seq = seq
            conn.execute(
                """
                INSERT INTO bridge_outbound_events (
                    event_id, worker_no, seq, action, text, file_path, file_name,
                    mime_type, file_size, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
                """,
                (
                    event.event_id,
                    self.worker_no,
                    event.seq,
                    event.action,
                    event.text,
                    event.file_path,
                    event.file_name,
                    event.mime_type,
                    event.file_size,
                    to_iso8601(event.created_at),
                ),
            )
        return event

    def _finalize_request(self, result: BridgeResult) -> None:
        self.initialize()
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE bridge_requests
                SET status = ?, input_kind = ?, finished_at = ?, lease_until = NULL,
                    final_content = ?, error_code = ?, error_message = ?
                WHERE request_id = ?
                """,
                (
                    result.final_status,
                    result.input_kind,
                    to_iso8601(result.finished_at),
                    result.final_content,
                    result.error_code,
                    result.error_message,
                    result.request_id,
                ),
            )

    def _make_result(
        self,
        request: BridgeRequest,
        *,
        final_status: str,
        final_content: str,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> BridgeResult:
        return BridgeResult(
            version=self.VERSION,
            request_id=request.request_id,
            seq=request.seq,
            source=request.source,
            final_status=final_status,
            input_content=request.content,
            input_kind=request.input_kind,
            final_content=final_content,
            error_code=error_code,
            error_message=error_message,
            started_at=request.started_at,
        )

    def _take_next_sequence(self, conn: sqlite3.Connection, column: str) -> int:
        if column not in {"next_request_seq", "next_outbound_seq"}:
            raise ValueError(f"Unsupported sequence column: {column}")
        self._ensure_worker(conn)
        row = conn.execute(
            f"SELECT {column} FROM bridge_workers WHERE worker_no = ?",
            (self.worker_no,),
        ).fetchone()
        seq = int(row[column])
        conn.execute(
            f"""
            UPDATE bridge_workers
            SET {column} = {column} + 1, last_seen_at = ?
            WHERE worker_no = ?
            """,
            (to_iso8601(utc_now()), self.worker_no),
        )
        return seq

    def _ensure_worker(self, conn: sqlite3.Connection, *, now: datetime | None = None) -> None:
        timestamp = to_iso8601(now or utc_now())
        conn.execute(
            """
            INSERT INTO bridge_workers (worker_no, created_at, last_seen_at)
            VALUES (?, ?, ?)
            ON CONFLICT(worker_no) DO UPDATE SET last_seen_at = excluded.last_seen_at
            """,
            (self.worker_no, timestamp, timestamp),
        )

    def _recover_expired_leases(
        self,
        conn: sqlite3.Connection,
        *,
        now: datetime | None = None,
    ) -> None:
        timestamp = to_iso8601(now or utc_now())
        conn.execute(
            """
            UPDATE bridge_requests
            SET status = 'pending', started_at = NULL, lease_until = NULL
            WHERE status = 'running' AND lease_until IS NOT NULL AND lease_until < ?
            """,
            (timestamp,),
        )
        conn.execute(
            """
            UPDATE bridge_outbound_events
            SET status = 'pending', started_at = NULL, lease_until = NULL
            WHERE status = 'running' AND lease_until IS NOT NULL AND lease_until < ?
            """,
            (timestamp,),
        )

    def _recover_expired_leases_if_due(
        self,
        conn: sqlite3.Connection,
        *,
        now: datetime,
    ) -> None:
        now_monotonic = time.monotonic()
        if (
            self._last_lease_recovery_monotonic is not None
            and now_monotonic - self._last_lease_recovery_monotonic
            < self.LEASE_RECOVERY_INTERVAL_SECONDS
        ):
            return
        self._recover_expired_leases(conn, now=now)
        self._last_lease_recovery_monotonic = now_monotonic

    def _initialize_schema(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS bridge_workers (
                worker_no TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                next_request_seq INTEGER NOT NULL DEFAULT 1,
                next_outbound_seq INTEGER NOT NULL DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS bridge_requests (
                request_id TEXT PRIMARY KEY,
                worker_no TEXT NOT NULL,
                session_no TEXT,
                seq INTEGER NOT NULL,
                source TEXT NOT NULL,
                source_meta_json TEXT NOT NULL,
                content TEXT NOT NULL,
                input_kind TEXT NOT NULL,
                content_type TEXT NOT NULL,
                remote_response_required INTEGER NOT NULL,
                status TEXT NOT NULL,
                enqueue_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                lease_until TEXT,
                final_content TEXT,
                error_code TEXT,
                error_message TEXT,
                UNIQUE(worker_no, seq)
            );

            CREATE INDEX IF NOT EXISTS idx_bridge_requests_pending
            ON bridge_requests(worker_no, status, seq);

            CREATE INDEX IF NOT EXISTS idx_bridge_requests_finished
            ON bridge_requests(status, finished_at);

            CREATE TABLE IF NOT EXISTS bridge_progress (
                progress_id TEXT PRIMARY KEY,
                request_id TEXT NOT NULL,
                seq INTEGER NOT NULL,
                source TEXT NOT NULL,
                content TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                delivered_at TEXT,
                FOREIGN KEY(request_id) REFERENCES bridge_requests(request_id)
            );

            CREATE INDEX IF NOT EXISTS idx_bridge_progress_pending
            ON bridge_progress(request_id, status, created_at);

            CREATE TABLE IF NOT EXISTS bridge_outbound_events (
                event_id TEXT PRIMARY KEY,
                worker_no TEXT NOT NULL,
                seq INTEGER NOT NULL,
                action TEXT NOT NULL,
                text TEXT,
                file_path TEXT,
                file_name TEXT,
                mime_type TEXT,
                file_size INTEGER,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                started_at TEXT,
                delivered_at TEXT,
                lease_until TEXT,
                attempts INTEGER NOT NULL DEFAULT 0,
                last_error TEXT,
                UNIQUE(worker_no, seq)
            );

            CREATE INDEX IF NOT EXISTS idx_bridge_outbound_pending
            ON bridge_outbound_events(worker_no, status, seq);

            CREATE INDEX IF NOT EXISTS idx_bridge_outbound_delivered
            ON bridge_outbound_events(status, delivered_at);
            """
        )

    def _cleanup_legacy_worker_dirs(self) -> None:
        try:
            self.legacy_cleanup_lock_dir.mkdir(exist_ok=False)
        except FileExistsError:
            return

        try:
            bridge_root = self.bridge_dir.resolve()
            for path in self.bridge_dir.iterdir():
                if not path.is_dir() or path == self.legacy_cleanup_lock_dir:
                    continue
                if path.name != "local-cli" and not path.name.startswith("worker-"):
                    continue
                resolved_path = path.resolve()
                if resolved_path.parent != bridge_root:
                    logger.warning("Skipping unsafe legacy bridge cleanup target: %s", path)
                    continue
                shutil.rmtree(resolved_path)
            self._write_migration_marker()
        finally:
            try:
                self.legacy_cleanup_lock_dir.rmdir()
            except OSError:
                logger.warning("Failed to remove bridge cleanup lock: %s", self.legacy_cleanup_lock_dir)

    def _cleanup_old_worker_log_dirs(self) -> None:
        if self.max_worker_log_dirs_per_cleanup <= 0:
            return
        try:
            self.worker_logs_cleanup_lock_dir.mkdir(exist_ok=False)
        except FileExistsError:
            return

        try:
            logs_root = self.worker_logs_root.resolve()
            cutoff = time.time() - self.worker_log_retention_seconds
            candidates: list[tuple[float, Path]] = []
            for path in self.worker_logs_root.iterdir():
                if not path.is_dir() or path == self.worker_logs_cleanup_lock_dir:
                    continue
                if path.name == self.session_binding_id or not path.name.startswith("worker-"):
                    continue
                try:
                    modified_at = path.stat().st_mtime
                except FileNotFoundError:
                    continue
                if modified_at < cutoff:
                    candidates.append((modified_at, path))

            for _modified_at, path in sorted(candidates)[: self.max_worker_log_dirs_per_cleanup]:
                resolved_path = path.resolve()
                if resolved_path.parent != logs_root:
                    logger.warning("Skipping unsafe worker log cleanup target: %s", path)
                    continue
                try:
                    shutil.rmtree(resolved_path)
                except FileNotFoundError:
                    continue
                except OSError:
                    logger.warning("Failed to remove stale worker log directory: %s", path)
        finally:
            try:
                self.worker_logs_cleanup_lock_dir.rmdir()
            except OSError:
                logger.warning(
                    "Failed to remove worker log cleanup lock: %s",
                    self.worker_logs_cleanup_lock_dir,
                )

    def _write_migration_marker(self) -> None:
        payload = {
            "schema_version": self.VERSION,
            "legacy_fs_cleaned": True,
            "cleaned_at": to_iso8601(utc_now()),
        }
        temp_path = self.migration_path.with_suffix(".json.tmp")
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp_path.replace(self.migration_path)

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA synchronous = NORMAL")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    @staticmethod
    def _row_to_request(row: sqlite3.Row | dict[str, Any]) -> BridgeRequest:
        return BridgeRequest(
            version=SqliteBridgeStore.VERSION,
            request_id=str(row["request_id"]),
            seq=int(row["seq"]),
            source=normalize_bridge_source(
                str(row["source"]),
                source_meta=json.loads(str(row["source_meta_json"])),
            ),
            source_meta=json.loads(str(row["source_meta_json"])),
            content=str(row["content"]),
            input_kind=str(row["input_kind"]),
            content_type=str(row["content_type"]),
            enqueue_time=from_iso8601(str(row["enqueue_at"])),
            remote_response_required=bool(row["remote_response_required"]),
            status=str(row["status"]),
            started_at=(
                from_iso8601(str(row["started_at"])) if row["started_at"] is not None else None
            ),
        )

    @staticmethod
    def _row_to_result(row: sqlite3.Row) -> BridgeResult:
        return BridgeResult(
            version=SqliteBridgeStore.VERSION,
            request_id=str(row["request_id"]),
            seq=int(row["seq"]),
            source=normalize_bridge_source(str(row["source"])),
            final_status=str(row["status"]),
            input_content=str(row["content"]),
            input_kind=str(row["input_kind"]),
            final_content=str(row["final_content"] or ""),
            error_code=str(row["error_code"]) if row["error_code"] is not None else None,
            error_message=(
                str(row["error_message"]) if row["error_message"] is not None else None
            ),
            started_at=(
                from_iso8601(str(row["started_at"])) if row["started_at"] is not None else None
            ),
            finished_at=from_iso8601(str(row["finished_at"])),
        )

    @staticmethod
    def _row_to_progress(row: sqlite3.Row) -> BridgeProgress:
        return BridgeProgress(
            version=SqliteBridgeStore.VERSION,
            request_id=str(row["request_id"]),
            seq=int(row["seq"]),
            source=normalize_bridge_source(str(row["source"])),
            progress_id=str(row["progress_id"]),
            content=str(row["content"]),
            created_at=from_iso8601(str(row["created_at"])),
        )

    @staticmethod
    def _row_to_outbound_event(row: sqlite3.Row | dict[str, Any]) -> BridgeOutboundEvent:
        return BridgeOutboundEvent(
            version=SqliteBridgeStore.VERSION,
            event_id=str(row["event_id"]),
            seq=int(row["seq"]),
            action=str(row["action"]),
            status=str(row["status"]),
            text=str(row["text"] or ""),
            file_path=str(row["file_path"] or ""),
            file_name=str(row["file_name"] or ""),
            mime_type=str(row["mime_type"] or ""),
            file_size=int(row["file_size"] or 0),
            created_at=from_iso8601(str(row["created_at"])),
            started_at=(
                from_iso8601(str(row["started_at"])) if row["started_at"] is not None else None
            ),
        )


# Compatibility alias for existing integrations. New code should use SqliteBridgeStore.
FileBridgeStore = SqliteBridgeStore


def resolve_session_binding_id(session_key: str | None) -> str:
    """Convert a terminal key into a safe worker number."""
    if not session_key:
        return "local-cli"
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", session_key).strip("._-")
    return sanitized or "local-cli"
