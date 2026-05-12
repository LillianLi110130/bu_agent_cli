"""Mock relay server for local worker and Web console debugging."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
import json
import time
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import uvicorn


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _encode_sse(event: str, payload: dict[str, Any]) -> bytes:
    data = json.dumps(payload, ensure_ascii=False)
    return f"event: {event}\ndata: {data}\n\n".encode("utf-8")


def _encode_web_sse(payload: dict[str, Any]) -> bytes:
    data = json.dumps(payload, ensure_ascii=False)
    return f"data: {data}\n\n".encode("utf-8")


@dataclass
class MockWorkerMessage:
    """One fake worker message queued for delivery."""

    content: str
    worker_id: str
    source: str = "im"


@dataclass(slots=True)
class _WebConsoleMessage:
    id: str
    role: str
    content: str
    created_at: str
    status: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "createdAt": self.created_at,
            "status": self.status,
        }


@dataclass(slots=True)
class _WebConsoleSession:
    id: str
    title: str
    worker_id: str
    updated_at: str
    last_message: str = ""
    messages: list[_WebConsoleMessage] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "updatedAt": self.updated_at,
            "lastMessage": self.last_message,
        }


@dataclass(slots=True)
class _RequestRecord:
    request_id: str
    worker_id: str
    session_id: str
    content: str
    accepted_at: str
    source: str = "web"
    status: str = "submitted"
    final_content: str | None = None
    finished_at: str | None = None
    error_message: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    subscribers: list[asyncio.Queue[dict[str, Any] | None]] = field(default_factory=list)

    @property
    def is_terminal(self) -> bool:
        return self.status in {"completed", "failed", "stopped"}

    def to_result_dict(self) -> dict[str, Any]:
        return {
            "requestId": self.request_id,
            "workerId": self.worker_id,
            "status": self.status,
            "finalContent": self.final_content,
            "errorMessage": self.error_message,
            "finishedAt": self.finished_at,
        }


@dataclass
class MockGatewayState:
    """In-memory state for the mock relay/gateway API."""

    queued_messages: list[MockWorkerMessage] = field(default_factory=list)
    completions: list[dict[str, Any]] = field(default_factory=list)
    progress_updates: list[dict[str, Any]] = field(default_factory=list)
    sent_texts: list[dict[str, Any]] = field(default_factory=list)
    uploaded_attachments: list[dict[str, Any]] = field(default_factory=list)
    online_workers: dict[str, dict[str, Any]] = field(default_factory=dict)
    worker_ttl_seconds: float = 30.0
    stream_heartbeat_interval_seconds: float = 0.1
    stream_versions: dict[str, int] = field(default_factory=dict)
    stream_notifications: dict[str, asyncio.Event] = field(default_factory=dict)

    _sessions_by_worker: dict[str, list[_WebConsoleSession]] = field(default_factory=dict)
    _requests: dict[str, _RequestRecord] = field(default_factory=dict)
    _active_web_request_id_by_worker: dict[str, str] = field(default_factory=dict)
    _last_web_request_id_by_worker: dict[str, str] = field(default_factory=dict)
    _last_completed_at_by_worker: dict[str, str] = field(default_factory=dict)
    _request_seq: int = 0

    def __post_init__(self) -> None:
        self._seed_sessions()

    def _seed_sessions(self) -> None:
        worker_id = "worker-hk-01"
        now = _utc_now_iso()
        self._sessions_by_worker.setdefault(
            worker_id,
            [
                _WebConsoleSession(
                    id="session-packaging",
                    title="打包调试",
                    worker_id=worker_id,
                    updated_at=now,
                    last_message="检查 Windows 打包链路",
                ),
                _WebConsoleSession(
                    id="session-release",
                    title="发布验证",
                    worker_id=worker_id,
                    updated_at=now,
                    last_message="验证 portable 发布产物",
                ),
                _WebConsoleSession(
                    id="session-readme",
                    title="README 修改",
                    worker_id=worker_id,
                    updated_at=now,
                    last_message="同步 readme 内容",
                ),
            ],
        )

    def _next_request_id(self) -> str:
        self._request_seq += 1
        return f"webreq-{self._request_seq:06d}"

    @staticmethod
    def _normalize_source(source: str | None) -> str:
        normalized = (source or "").strip().lower()
        if normalized in {"im", "web"}:
            return normalized
        return "im"

    def enqueue_message(
        self,
        *,
        worker_id: str,
        content: str,
        source: str = "im",
    ) -> MockWorkerMessage:
        message = MockWorkerMessage(worker_id=worker_id, content=content, source=source)
        self.queued_messages.append(message)
        self.notify_stream(worker_id)
        return message

    def mark_online(self, *, worker_id: str) -> None:
        self.online_workers[worker_id] = {
            "last_seen_at": time.monotonic(),
            "online": True,
        }
        self.notify_stream(worker_id)

    def mark_worker_online(self, *, worker_id: str) -> None:
        self.mark_online(worker_id=worker_id)

    def mark_seen(self, *, worker_id: str) -> None:
        record = self.online_workers.get(worker_id)
        if record is None:
            return
        record["last_seen_at"] = time.monotonic()

    def mark_worker_seen(self, *, worker_id: str) -> None:
        self.mark_seen(worker_id=worker_id)

    def mark_offline(self, *, worker_id: str) -> None:
        record = self.online_workers.get(worker_id)
        if record is None:
            return
        record["online"] = False
        record["last_seen_at"] = time.monotonic()
        self.notify_stream(worker_id)

    def mark_worker_offline(self, *, worker_id: str) -> None:
        self.mark_offline(worker_id=worker_id)

    def is_online(self, worker_id: str) -> bool:
        record = self.online_workers.get(worker_id)
        if record is None or not record.get("online", False):
            return False
        last_seen_at = float(record.get("last_seen_at", 0.0))
        if time.monotonic() - last_seen_at > self.worker_ttl_seconds:
            record["online"] = False
            return False
        return True

    def is_worker_online(self, worker_id: str) -> bool:
        return self.is_online(worker_id)

    def register_stream(self, *, worker_id: str) -> int:
        version = self.stream_versions.get(worker_id, 0) + 1
        self.stream_versions[worker_id] = version
        self.mark_seen(worker_id=worker_id)
        self.notify_stream(worker_id)
        return version

    def is_current_stream(self, *, worker_id: str, version: int) -> bool:
        return self.stream_versions.get(worker_id) == version

    def dequeue_message(self, *, worker_id: str) -> MockWorkerMessage | None:
        for index, message in enumerate(list(self.queued_messages)):
            if message.worker_id != worker_id:
                continue
            return self.queued_messages.pop(index)
        return None

    async def worker_dequeue_next(self, worker_id: str) -> _RequestRecord | MockWorkerMessage | None:
        message = self.dequeue_message(worker_id=worker_id)
        if message is None:
            return None
        if message.source == "web":
            active_request_id = self._active_web_request_id_by_worker.get(worker_id)
            if active_request_id is not None:
                record = self._requests.get(active_request_id)
                if record is not None and record.content == message.content and not record.is_terminal:
                    return record
        return message

    def notify_stream(self, worker_id: str) -> None:
        self._get_stream_notification(worker_id).set()

    async def wait_for_stream_activity(self, *, worker_id: str, timeout: float) -> bool:
        event = self._get_stream_notification(worker_id)
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
            return True
        except TimeoutError:
            return False
        finally:
            event.clear()

    def _get_stream_notification(self, worker_id: str) -> asyncio.Event:
        event = self.stream_notifications.get(worker_id)
        if event is None:
            event = asyncio.Event()
            self.stream_notifications[worker_id] = event
        return event

    async def list_sessions(self, worker_id: str) -> list[dict[str, Any]]:
        sessions = self._sessions_by_worker.get(worker_id)
        if sessions is None:
            sessions = [
                _WebConsoleSession(
                    id=f"{worker_id}-default",
                    title="当前对话",
                    worker_id=worker_id,
                    updated_at=_utc_now_iso(),
                )
            ]
            self._sessions_by_worker[worker_id] = sessions
        return [session.to_dict() for session in sessions]

    async def list_messages(self, session_id: str) -> list[dict[str, Any]]:
        for sessions in self._sessions_by_worker.values():
            for session in sessions:
                if session.id == session_id:
                    return [message.to_dict() for message in session.messages]
        raise KeyError(session_id)

    async def register_submit(self, *, worker_id: str, session_id: str, content: str) -> _RequestRecord:
        active_request_id = self._active_web_request_id_by_worker.get(worker_id)
        if active_request_id is not None:
            existing = self._requests.get(active_request_id)
            if existing is not None and not existing.is_terminal:
                raise ValueError("worker_busy")

        accepted_at = _utc_now_iso()
        request_id = self._next_request_id()
        record = _RequestRecord(
            request_id=request_id,
            worker_id=worker_id,
            session_id=session_id,
            content=content,
            accepted_at=accepted_at,
        )
        record.events.append(
            {
                "type": "submitted",
                "requestId": request_id,
                "workerId": worker_id,
                "ts": accepted_at,
            }
        )
        self._requests[request_id] = record
        self._active_web_request_id_by_worker[worker_id] = request_id
        self._last_web_request_id_by_worker[worker_id] = request_id
        self._record_session_message(
            worker_id=worker_id,
            session_id=session_id,
            role="user",
            content=content,
            created_at=accepted_at,
        )
        self.enqueue_message(worker_id=worker_id, content=content, source="web")
        return record

    def _record_session_message(
        self,
        *,
        worker_id: str,
        session_id: str,
        role: str,
        content: str,
        created_at: str,
        status: str | None = None,
    ) -> None:
        sessions = self._sessions_by_worker.setdefault(worker_id, [])
        session = next((item for item in sessions if item.id == session_id), None)
        if session is None:
            session = _WebConsoleSession(
                id=session_id,
                title="当前对话",
                worker_id=worker_id,
                updated_at=created_at,
            )
            sessions.insert(0, session)
        message = _WebConsoleMessage(
            id=f"{role}-{len(session.messages) + 1}",
            role=role,
            content=content,
            created_at=created_at,
            status=status,
        )
        session.messages.append(message)
        session.updated_at = created_at
        session.last_message = content.strip().replace("\n", " ")[:120]

    def _append_request_event(self, record: _RequestRecord, payload: dict[str, Any]) -> None:
        record.events.append(payload)
        for queue in list(record.subscribers):
            queue.put_nowait(payload)

    def _finish_request(self, record: _RequestRecord) -> None:
        for queue in list(record.subscribers):
            queue.put_nowait(None)

    async def get_worker_summary(self, worker_id: str) -> dict[str, Any]:
        return {
            "workerId": worker_id,
            "isOnline": self.is_worker_online(worker_id),
            "lastCompletedAt": self._last_completed_at_by_worker.get(worker_id),
        }

    async def get_active_request_for_worker(
        self,
        worker_id: str,
    ) -> tuple[str, list[dict[str, Any]], bool]:
        request_id = self._active_web_request_id_by_worker.get(worker_id)
        if request_id is None:
            request_id = self._last_web_request_id_by_worker.get(worker_id)
        if request_id is None or request_id not in self._requests:
            raise KeyError(worker_id)
        record = self._requests[request_id]
        return record.request_id, list(record.events), record.is_terminal

    async def get_request_history(self, request_id: str) -> tuple[list[dict[str, Any]], bool]:
        record = self._requests.get(request_id)
        if record is None:
            raise KeyError(request_id)
        return list(record.events), record.is_terminal

    async def add_subscriber(self, request_id: str) -> asyncio.Queue[dict[str, Any] | None]:
        record = self._requests.get(request_id)
        if record is None:
            raise KeyError(request_id)
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        record.subscribers.append(queue)
        return queue

    async def remove_subscriber(
        self,
        request_id: str,
        queue: asyncio.Queue[dict[str, Any] | None],
    ) -> None:
        record = self._requests.get(request_id)
        if record is None:
            return
        if queue in record.subscribers:
            record.subscribers.remove(queue)

    async def stop_request_if_current_worker_stream_closed(self, *, worker_id: str, request_id: str) -> None:
        current_request_id = self._active_web_request_id_by_worker.get(worker_id)
        if current_request_id != request_id:
            return
        record = self._requests.get(request_id)
        if record is None or record.is_terminal:
            return
        record.status = "stopped"
        record.error_message = "stopped_by_web_disconnect"
        record.finished_at = _utc_now_iso()
        self._append_request_event(
            record,
            {
                "type": "failed",
                "requestId": request_id,
                "workerId": worker_id,
                "errorMessage": record.error_message,
                "finishedAt": record.finished_at,
            },
        )
        self._finish_request(record)
        self._active_web_request_id_by_worker.pop(worker_id, None)

    async def get_request_result(self, request_id: str) -> dict[str, Any]:
        record = self._requests.get(request_id)
        if record is None:
            raise KeyError(request_id)
        return record.to_result_dict()

    async def worker_append_progress(self, *, worker_id: str, content: str, source: str | None) -> bool:
        normalized_source = self._normalize_source(source)
        self.progress_updates.append(
            {
                "worker_id": worker_id,
                "content": content,
                "source": normalized_source,
                "created_at": time.time(),
            }
        )

        if normalized_source != "web":
            return True

        request_id = self._active_web_request_id_by_worker.get(worker_id)
        if request_id is None:
            return False
        record = self._requests.get(request_id)
        if record is None or record.is_terminal:
            return False

        if record.status == "submitted":
            record.status = "processing"
            self._append_request_event(
                record,
                {
                    "type": "processing",
                    "requestId": record.request_id,
                    "workerId": worker_id,
                    "ts": _utc_now_iso(),
                },
            )

        self._append_request_event(
            record,
            {
                "type": "progress",
                "requestId": record.request_id,
                "workerId": worker_id,
                "content": content,
                "ts": _utc_now_iso(),
            },
        )
        return True

    async def worker_complete_next(
        self,
        *,
        worker_id: str,
        final_content: str,
        source: str | None,
        final_status: str = "completed",
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> bool:
        normalized_source = self._normalize_source(source)
        self.completions.append(
            {
                "worker_id": worker_id,
                "final_content": final_content,
                "source": normalized_source,
                "final_status": final_status,
                "error_code": error_code,
                "error_message": error_message,
                "completed_at": time.time(),
            }
        )

        if normalized_source != "web":
            return True

        request_id = self._active_web_request_id_by_worker.get(worker_id)
        if request_id is None:
            return False
        record = self._requests.get(request_id)
        if record is None or record.is_terminal:
            return False

        finished_at = _utc_now_iso()
        record.finished_at = finished_at
        record.final_content = final_content

        if final_status == "failed":
            record.status = "failed"
            record.error_message = error_message or error_code or "worker_failed"
            self._append_request_event(
                record,
                {
                    "type": "failed",
                    "requestId": request_id,
                    "workerId": worker_id,
                    "errorMessage": record.error_message,
                    "finishedAt": finished_at,
                },
            )
        else:
            record.status = "completed"
            self._last_completed_at_by_worker[worker_id] = finished_at
            self._append_request_event(
                record,
                {
                    "type": "completed",
                    "requestId": request_id,
                    "workerId": worker_id,
                    "finalContent": final_content,
                    "finishedAt": finished_at,
                },
            )
            self._record_session_message(
                worker_id=worker_id,
                session_id=record.session_id,
                role="assistant",
                content=final_content,
                created_at=finished_at,
                status="completed",
            )

        self._finish_request(record)
        self._active_web_request_id_by_worker.pop(worker_id, None)
        return True


class WorkerRequest(BaseModel):
    worker_id: str


class CompleteRequest(BaseModel):
    worker_id: str
    final_content: str
    source: str | None = None
    final_status: str = "completed"
    error_code: str | None = None
    error_message: str | None = None


class ProgressRequest(BaseModel):
    worker_id: str
    content: str
    source: str | None = None


class EnqueueRequest(BaseModel):
    worker_id: str
    content: str
    source: str = "im"


class SendTextRequest(BaseModel):
    worker_id: str
    text: str


class WorkerSummaryResponse(BaseModel):
    workerId: str
    isOnline: bool = False
    lastCompletedAt: str | None = None


class SessionListItem(BaseModel):
    id: str
    title: str
    updatedAt: str
    lastMessage: str | None = None


class MessageItem(BaseModel):
    id: str
    role: str
    content: str
    createdAt: str
    status: str | None = None


class SubmitMessageRequest(BaseModel):
    workerId: str = Field(..., alias="workerId")
    sessionId: str = Field(..., alias="sessionId")
    content: str


class SubmitMessageResponse(BaseModel):
    ok: bool = True
    acceptedAt: str
    requestId: str | None = None


class PollResultResponse(BaseModel):
    requestId: str
    workerId: str
    status: str
    finalContent: str | None = None
    errorMessage: str | None = None
    finishedAt: str | None = None


def create_mock_gateway_app(state: MockGatewayState | None = None) -> FastAPI:
    """Create a FastAPI app that mimics the worker gateway contract plus Web relay routes."""
    state = state or MockGatewayState()
    app = FastAPI(title="Mock Worker Relay Gateway")
    app.state.mock_gateway = state
    app.state.web_console_state = state

    @app.get("/web-console/workers/{worker_id}", response_model=WorkerSummaryResponse, tags=["WebConsole"])
    async def get_worker_summary(worker_id: str):
        return await state.get_worker_summary(worker_id)

    @app.get(
        "/web-console/workers/{worker_id}/sessions",
        response_model=list[SessionListItem],
        tags=["WebConsole"],
    )
    async def list_sessions(worker_id: str):
        return await state.list_sessions(worker_id)

    @app.get(
        "/web-console/sessions/{session_id}/messages",
        response_model=list[MessageItem],
        tags=["WebConsole"],
    )
    async def list_messages(session_id: str):
        try:
            return await state.list_messages(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found") from exc

    @app.post("/web-console/messages", response_model=SubmitMessageResponse, tags=["WebConsole"])
    async def submit_message(request: SubmitMessageRequest):
        try:
            record = await state.register_submit(
                worker_id=request.workerId,
                session_id=request.sessionId,
                content=request.content,
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return SubmitMessageResponse(ok=True, acceptedAt=record.accepted_at, requestId=record.request_id)

    @app.get("/web-console/workers/{worker_id}/events", tags=["WebConsole"])
    async def get_worker_events(worker_id: str):
        async def event_stream():
            try:
                request_id, history, is_terminal = await state.get_active_request_for_worker(worker_id)
            except KeyError as exc:
                raise HTTPException(
                    status_code=404,
                    detail=f"No active web request for worker {worker_id}",
                ) from exc

            for payload in history:
                yield _encode_web_sse(payload)
            if is_terminal:
                yield b": done\n\n"
                return

            queue = await state.add_subscriber(request_id)
            try:
                while True:
                    try:
                        payload = await asyncio.wait_for(queue.get(), timeout=15.0)
                    except TimeoutError:
                        yield b": heartbeat\n\n"
                        continue

                    if payload is None:
                        yield b": done\n\n"
                        return
                    yield _encode_web_sse(payload)
            finally:
                await state.remove_subscriber(request_id, queue)
                await state.stop_request_if_current_worker_stream_closed(
                    worker_id=worker_id,
                    request_id=request_id,
                )

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/web-console/messages/{request_id}/result", response_model=PollResultResponse, tags=["WebConsole"])
    async def get_message_result(request_id: str):
        try:
            return await state.get_request_result(request_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Request {request_id} not found") from exc

    @app.post("/api/worker/poll", tags=["Worker"])
    async def poll(request: WorkerRequest) -> dict[str, Any]:
        state.mark_worker_seen(worker_id=request.worker_id)
        queued = await state.worker_dequeue_next(request.worker_id)
        if queued is None:
            return {"messages": []}
        if isinstance(queued, _RequestRecord):
            return {
                "messages": [
                    {
                        "content": queued.content,
                        "worker_id": request.worker_id,
                        "source": queued.source,
                    }
                ]
            }
        return {"messages": [asdict(queued)]}

    @app.get("/api/worker/stream", tags=["Worker"])
    async def stream(worker_id: str) -> StreamingResponse:
        stream_version = state.register_stream(worker_id=worker_id)

        async def event_stream():
            yield _encode_sse("ready", {"worker_id": worker_id})
            while state.is_online(worker_id) and state.is_current_stream(
                worker_id=worker_id,
                version=stream_version,
            ):
                queued = await state.worker_dequeue_next(worker_id)
                if queued is not None:
                    state.mark_seen(worker_id=worker_id)
                    if isinstance(queued, _RequestRecord):
                        payload = {"content": queued.content, "source": queued.source}
                    else:
                        payload = {"content": queued.content, "source": queued.source}
                    yield _encode_sse("message", payload)
                    continue

                has_activity = await state.wait_for_stream_activity(
                    worker_id=worker_id,
                    timeout=state.stream_heartbeat_interval_seconds,
                )
                if not state.is_current_stream(worker_id=worker_id, version=stream_version):
                    yield _encode_sse(
                        "error",
                        {"code": "replaced", "message": "connection replaced by newer stream"},
                    )
                    break
                if not state.is_online(worker_id):
                    break

                state.mark_seen(worker_id=worker_id)
                if not has_activity:
                    yield _encode_sse("heartbeat", {"ts": time.time()})

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    @app.post("/api/worker/online", tags=["Worker"])
    async def online(request: WorkerRequest) -> dict[str, bool]:
        state.mark_online(worker_id=request.worker_id)
        return {"ok": True}

    @app.post("/api/worker/offline", tags=["Worker"])
    async def offline(request: WorkerRequest) -> dict[str, bool]:
        state.mark_offline(worker_id=request.worker_id)
        return {"ok": True}

    @app.post("/api/worker/complete", tags=["Worker"])
    async def complete(request: CompleteRequest) -> dict[str, bool]:
        ok = await state.worker_complete_next(
            worker_id=request.worker_id,
            final_content=request.final_content,
            source=request.source,
            final_status=request.final_status,
            error_code=request.error_code,
            error_message=request.error_message,
        )
        return {"ok": ok}

    @app.post("/api/worker/progress", tags=["Worker"])
    async def progress(request: ProgressRequest) -> dict[str, bool]:
        ok = await state.worker_append_progress(
            worker_id=request.worker_id,
            content=request.content,
            source=request.source,
        )
        return {"ok": ok}

    @app.post("/api/worker/send_text", tags=["Worker"])
    async def send_text(request: SendTextRequest) -> dict[str, bool]:
        state.sent_texts.append(
            {
                "worker_id": request.worker_id,
                "text": request.text,
            }
        )
        return {"ok": True}

    @app.post("/api/worker/upload_attachment", tags=["Worker"])
    async def upload_attachment(request: Request) -> dict[str, bool]:
        content_type = request.headers.get("content-type", "")
        body = await request.body()
        state.uploaded_attachments.append(
            {
                "content_type": content_type,
                "body": body,
            }
        )
        return {"ok": True}

    @app.post("/mock/messages", tags=["WorkerDebug"])
    async def enqueue(request: EnqueueRequest):
        if not state.is_online(request.worker_id):
            return JSONResponse(
                status_code=409,
                content={
                    "ok": False,
                    "error": "no_online_worker",
                    "worker_id": request.worker_id,
                },
            )
        message = state.enqueue_message(
            worker_id=request.worker_id,
            content=request.content,
            source=request.source,
        )
        return {"ok": True, "worker_id": message.worker_id}

    @app.get("/mock/messages", tags=["WorkerDebug"])
    async def list_pending_messages() -> dict[str, Any]:
        return {"messages": [asdict(message) for message in state.queued_messages]}

    @app.get("/mock/completions", tags=["WorkerDebug"])
    async def list_completions() -> dict[str, Any]:
        return {"completions": list(state.completions)}

    @app.get("/mock/progress", tags=["WorkerDebug"])
    async def list_progress() -> dict[str, Any]:
        return {"progress": list(state.progress_updates)}

    @app.get("/mock/online", tags=["WorkerDebug"])
    async def list_online() -> dict[str, Any]:
        online = {
            worker_id: record
            for worker_id, record in state.online_workers.items()
            if state.is_online(worker_id)
        }
        return {"online_workers": online}

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mock worker relay gateway server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8888)
    return parser.parse_args()


def cli_main() -> None:
    args = parse_args()
    uvicorn.run(create_mock_gateway_app(), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    cli_main()
