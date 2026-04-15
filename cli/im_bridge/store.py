"""File-backed queue storage for the local IM bridge."""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from typing import Any

from cli.im_bridge.models import BridgeRequest, BridgeResult, classify_input_kind, utc_now


class FileBridgeStore:
    """Manage bridge requests and results on the local filesystem."""

    VERSION = 1

    def __init__(self, root_dir: Path | str | None = None, session_binding_id: str = "local-cli") -> None:
        resolved_root = Path(root_dir).resolve() if root_dir is not None else Path.cwd().resolve()
        self.workspace_root = resolved_root
        self.session_binding_id = session_binding_id
        self.bridge_dir = self.workspace_root / ".tg_agent" / "im_bridge" / session_binding_id
        self.state_dir = self.bridge_dir / "state"
        self.locks_dir = self.bridge_dir / "locks"
        self.logs_dir = self.bridge_dir / "logs"
        self.inbox_pending_dir = self.bridge_dir / "inbox" / "pending"
        self.inbox_processing_dir = self.bridge_dir / "inbox" / "processing"
        self.results_dir = self.bridge_dir / "results"
        self.results_completed_dir = self.bridge_dir / "results" / "completed"
        self.results_failed_dir = self.bridge_dir / "results" / "failed"
        self.results_index_path = self.results_dir / "results.json"
        self.sequence_path = self.state_dir / "sequence.json"
        self.enqueue_lock_dir = self.locks_dir / "enqueue.lock"

    def initialize(self) -> None:
        """Create the bridge directory structure if it does not already exist."""
        for directory in (
            self.state_dir,
            self.locks_dir,
            self.logs_dir,
            self.inbox_pending_dir,
            self.inbox_processing_dir,
            self.results_dir,
            self.results_completed_dir,
            self.results_failed_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        if not self.sequence_path.exists():
            self._write_json_atomic(self.sequence_path, {"next_seq": 1})
        if not self.results_index_path.exists():
            self._write_json_atomic(self.results_index_path, {"version": self.VERSION, "results": []})

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
        """Append one text request to the pending queue."""
        self.initialize()
        source_meta = dict(source_meta or {})
        request_id = request_id or f"req_{uuid.uuid4().hex}"

        with self._enqueue_lock():
            seq = self._read_next_seq()
            request = BridgeRequest(
                version=self.VERSION,
                request_id=request_id,
                seq=seq,
                source=source,
                source_meta=source_meta,
                content=content,
                input_kind=input_kind or classify_input_kind(content),
                content_type="text",
                remote_response_required=remote_response_required,
                status="pending",
            )
            self._write_json_atomic(self.inbox_pending_dir / f"{seq:020d}.json", request.to_dict())
            self._write_json_atomic(self.sequence_path, {"next_seq": seq + 1})
            return request

    def claim_next_pending(self) -> BridgeRequest | None:
        """Claim the lowest-sequence pending request for processing."""
        self.initialize()
        for path in sorted(self.inbox_pending_dir.glob("*.json")):
            processing_path = self.inbox_processing_dir / path.name
            try:
                path.replace(processing_path)
            except FileNotFoundError:
                continue

            request = BridgeRequest.from_dict(self._read_json(processing_path))
            request.status = "running"
            request.started_at = utc_now()
            self._write_json_atomic(processing_path, request.to_dict())
            return request
        return None

    def complete_request(self, request: BridgeRequest, final_content: str = "") -> BridgeResult:
        """Persist a completed result and clear its processing file."""
        result = BridgeResult(
            version=self.VERSION,
            request_id=request.request_id,
            seq=request.seq,
            source=request.source,
            final_status="completed",
            input_content=request.content,
            input_kind=request.input_kind,
            final_content=final_content,
            started_at=request.started_at,
        )
        self._finalize_request(request, result, success=True)
        return result

    def fail_request(
        self,
        request: BridgeRequest,
        *,
        final_content: str,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> BridgeResult:
        """Persist a failed result and clear its processing file."""
        result = BridgeResult(
            version=self.VERSION,
            request_id=request.request_id,
            seq=request.seq,
            source=request.source,
            final_status="failed",
            input_content=request.content,
            input_kind=request.input_kind,
            final_content=final_content,
            error_code=error_code,
            error_message=error_message,
            started_at=request.started_at,
        )
        self._finalize_request(request, result, success=False)
        return result

    def find_result(self, request_id: str) -> BridgeResult | None:
        """Return a completed or failed result by request id."""
        self.initialize()
        payload = self._read_json(self.results_index_path)
        for item in payload.get("results", []):
            if str(item.get("request_id")) == request_id:
                return BridgeResult.from_dict(item)

        for path in (
            self.results_completed_dir / f"{request_id}.json",
            self.results_failed_dir / f"{request_id}.json",
        ):
            if path.exists():
                return BridgeResult.from_dict(self._read_json(path))
        return None

    def has_request(self, request_id: str) -> bool:
        """Return whether a request or result already exists for *request_id*."""
        if self.find_result(request_id) is not None:
            return True

        for directory in (self.inbox_pending_dir, self.inbox_processing_dir):
            for path in directory.glob("*.json"):
                payload = self._read_json(path)
                if str(payload.get("request_id")) == request_id:
                    return True
        return False

    def pending_count(self) -> int:
        """Return the number of queued pending requests."""
        self.initialize()
        return len(list(self.inbox_pending_dir.glob("*.json")))

    def _finalize_request(
        self,
        request: BridgeRequest,
        result: BridgeResult,
        *,
        success: bool,
    ) -> None:
        payload = self._read_json(self.results_index_path)
        existing_results = [
            item for item in payload.get("results", []) if str(item.get("request_id")) != request.request_id
        ]
        existing_results.append(result.to_dict())
        self._write_json_atomic(
            self.results_index_path,
            {"version": self.VERSION, "results": existing_results},
        )
        processing_path = self.inbox_processing_dir / f"{request.seq:020d}.json"
        if processing_path.exists():
            processing_path.unlink()

    def _read_next_seq(self) -> int:
        payload = self._read_json(self.sequence_path)
        return int(payload.get("next_seq", 1))

    def _read_json(self, path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_json_atomic(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp_path.replace(path)

    def _enqueue_lock(self):
        return _DirectoryLock(self.enqueue_lock_dir)


def resolve_session_binding_id(session_key: str | None) -> str:
    """Convert a session key into a filesystem-safe binding id."""
    if not session_key:
        return "local-cli"
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", session_key).strip("._-")
    return sanitized or "local-cli"


class _DirectoryLock:
    """A simple directory-based lock for cross-process coordination."""

    def __init__(
        self,
        lock_dir: Path,
        *,
        timeout_seconds: float = 5.0,
        retry_interval_seconds: float = 0.05,
    ) -> None:
        self.lock_dir = lock_dir
        self.timeout_seconds = timeout_seconds
        self.retry_interval_seconds = retry_interval_seconds

    def __enter__(self) -> "_DirectoryLock":
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            try:
                self.lock_dir.mkdir(parents=True, exist_ok=False)
                return self
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out waiting for bridge lock: {self.lock_dir}")
                time.sleep(self.retry_interval_seconds)

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            self.lock_dir.rmdir()
        except FileNotFoundError:
            return
