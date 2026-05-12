"""Local Web console relay routes backed by the worker delivery protocol.

This module intentionally avoids calling the in-process agent directly. Web messages are
queued onto the same worker-style delivery channel used by the local CLI worker so that
manual validation through ``test_server.py`` matches the IM-style relay architecture:

Browser -> Python server -> worker poll/SSE -> local CLI bridge -> worker progress/complete -> Web SSE
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _serialize_sse(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


def _serialize_worker_sse(event: str, payload: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


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
    delivered_at: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    subscribers: list[asyncio.Queue[dict[str, Any] | None]] = field(default_factory=list)

    def to_result_dict(self) -> dict[str, Any]:
        return {
            "requestId": self.request_id,
            "workerId": self.worker_id,
            "status": self.status,
            "finalContent": self.final_content,
            "errorMessage": self.error_message,
            "finishedAt": self.finished_at,
        }

    @property
    def is_terminal(self) -> bool:
        return self.status in {"completed", "failed", "stopped"}


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


class WorkerRequest(BaseModel):
    worker_id: str


class WorkerCompleteRequest(BaseModel):
    worker_id: str
    final_content: str
    source: str | None = None


class WorkerProgressRequest(BaseModel):
    worker_id: str
    content: str
    source: str | None = None


class _WebConsoleState:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._sessions_by_worker: dict[str, list[_WebConsoleSession]] = {}
        self._requests: dict[str, _RequestRecord] = {}
        self._pending_request_ids_by_worker: dict[str, list[str]] = {}
        self._inflight_request_ids_by_worker: dict[str, list[str]] = {}
        self._active_web_request_id_by_worker: dict[str, str] = {}
        self._last_completed_at_by_worker: dict[str, str] = {}
        self._online_workers: dict[str, dict[str, Any]] = {}
        self._stream_versions: dict[str, int] = {}
        self._stream_notifications: dict[str, asyncio.Event] = {}
        self.worker_ttl_seconds = 30.0
        self.stream_heartbeat_interval_seconds = 0.2
        self._seed_sessions()

    def _seed_sessions(self) -> None:
        worker_id = "worker-hk-01"
        now = _utc_now_iso()
        self._sessions_by_worker[worker_id] = [
            _WebConsoleSession(
                id="session-packaging",
                title="打包发布",
                worker_id=worker_id,
                updated_at=now,
                last_message="请帮我检查 Windows 打包相关脚本。",
            ),
            _WebConsoleSession(
                id="session-release",
                title="发布检查",
                worker_id=worker_id,
                updated_at=now,
                last_message="请帮我看一下 portable 发布流程。",
            ),
            _WebConsoleSession(
                id="session-readme",
                title="README 更新",
                worker_id=worker_id,
                updated_at=now,
                last_message="请帮我整理一下 readme。",
            ),
        ]

    async def get_worker_summary(self, worker_id: str) -> WorkerSummaryResponse:
        return WorkerSummaryResponse(
            workerId=worker_id,
            isOnline=self.is_worker_online(worker_id),
            lastCompletedAt=self._last_completed_at_by_worker.get(worker_id),
        )

    async def list_sessions(self, worker_id: str) -> list[dict[str, Any]]:
        async with self._lock:
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
        async with self._lock:
            session = self._find_session(session_id)
            if session is None:
                raise KeyError(session_id)
            return [message.to_dict() for message in session.messages]

    async def register_submit(self, worker_id: str, session_id: str, content: str) -> _RequestRecord:
        async with self._lock:
            active_request_id = self._active_web_request_id_by_worker.get(worker_id)
            if active_request_id:
                active_record = self._requests.get(active_request_id)
                if active_record is not None and not active_record.is_terminal:
                    raise ValueError("worker_busy")

            session = self._find_session(session_id)
            if session is None:
                session = _WebConsoleSession(
                    id=session_id,
                    title="当前对话",
                    worker_id=worker_id,
                    updated_at=_utc_now_iso(),
                )
                self._sessions_by_worker.setdefault(worker_id, []).append(session)

            accepted_at = _utc_now_iso()
            request_id = f"web-{datetime.now(UTC).timestamp():.6f}".replace(".", "")

            session.messages.append(
                _WebConsoleMessage(
                    id=f"user-{request_id}",
                    role="user",
                    content=content,
                    created_at=accepted_at,
                )
            )
            session.last_message = content
            session.updated_at = accepted_at

            record = _RequestRecord(
                request_id=request_id,
                worker_id=worker_id,
                session_id=session_id,
                content=content,
                accepted_at=accepted_at,
            )
            self._requests[request_id] = record
            self._pending_request_ids_by_worker.setdefault(worker_id, []).append(request_id)
            self._active_web_request_id_by_worker[worker_id] = request_id

        await self.emit_submitted(request_id)
        self.notify_stream(worker_id)
        return record

    async def get_request_result(self, request_id: str) -> dict[str, Any]:
        async with self._lock:
            record = self._requests.get(request_id)
            if record is None:
                raise KeyError(request_id)
            return record.to_result_dict()

    async def get_request_history(self, request_id: str) -> tuple[list[dict[str, Any]], bool]:
        async with self._lock:
            record = self._requests.get(request_id)
            if record is None:
                raise KeyError(request_id)
            return list(record.events), record.is_terminal

    async def get_active_request_for_worker(
        self,
        worker_id: str,
    ) -> tuple[str, list[dict[str, Any]], bool]:
        async with self._lock:
            request_id = self._active_web_request_id_by_worker.get(worker_id)
            if request_id is None:
                raise KeyError(worker_id)
            record = self._requests.get(request_id)
            if record is None:
                self._active_web_request_id_by_worker.pop(worker_id, None)
                raise KeyError(worker_id)
            return request_id, list(record.events), record.is_terminal

    async def add_subscriber(self, request_id: str) -> asyncio.Queue[dict[str, Any] | None]:
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        async with self._lock:
            record = self._requests.get(request_id)
            if record is None:
                raise KeyError(request_id)
            if not record.is_terminal:
                record.subscribers.append(queue)
                return queue
        await queue.put(None)
        return queue

    async def remove_subscriber(
        self,
        request_id: str,
        queue: asyncio.Queue[dict[str, Any] | None],
    ) -> None:
        async with self._lock:
            record = self._requests.get(request_id)
            if record is None:
                return
            if queue in record.subscribers:
                record.subscribers.remove(queue)

    async def stop_request_if_current_worker_stream_closed(
        self,
        *,
        worker_id: str,
        request_id: str,
    ) -> None:
        async with self._lock:
            active_request_id = self._active_web_request_id_by_worker.get(worker_id)
            record = self._requests.get(request_id)
            if active_request_id != request_id or record is None or record.is_terminal:
                return
        await self.stop_request(request_id)

    async def worker_dequeue_next(self, worker_id: str) -> _RequestRecord | None:
        request_id: str | None = None
        async with self._lock:
            pending = self._pending_request_ids_by_worker.setdefault(worker_id, [])
            if pending:
                request_id = pending.pop(0)
                self._inflight_request_ids_by_worker.setdefault(worker_id, []).append(request_id)

        if request_id is None:
            return None

        await self.emit_processing(request_id)
        async with self._lock:
            return self._requests.get(request_id)

    async def worker_complete_next(
        self,
        *,
        worker_id: str,
        final_content: str,
        source: str | None,
    ) -> bool:
        normalized_source = self._normalize_source(source)
        request_id: str | None = None

        async with self._lock:
            inflight = self._inflight_request_ids_by_worker.setdefault(worker_id, [])
            for index, candidate_request_id in enumerate(list(inflight)):
                record = self._requests.get(candidate_request_id)
                if record is None or record.is_terminal:
                    inflight.pop(index)
                    continue
                if self._normalize_source(record.source) != normalized_source:
                    continue
                request_id = inflight.pop(index)
                break

        if request_id is None:
            return False

        await self.complete_request(request_id, final_content=final_content)
        return True

    async def worker_append_progress(
        self,
        *,
        worker_id: str,
        content: str,
        source: str | None,
    ) -> bool:
        normalized_source = self._normalize_source(source)
        request_id: str | None = None

        async with self._lock:
            inflight = self._inflight_request_ids_by_worker.setdefault(worker_id, [])
            for candidate_request_id in inflight:
                record = self._requests.get(candidate_request_id)
                if record is None or record.is_terminal:
                    continue
                if self._normalize_source(record.source) != normalized_source:
                    continue
                request_id = candidate_request_id
                break

        if request_id is None:
            return False

        await self.append_progress_event(request_id, content=content)
        return True

    async def stop_request(self, request_id: str) -> str:
        stopped_at = _utc_now_iso()
        async with self._lock:
            record = self._requests.get(request_id)
            if record is None:
                raise KeyError(request_id)
            if record.is_terminal:
                return stopped_at

            pending = self._pending_request_ids_by_worker.get(record.worker_id, [])
            if request_id in pending:
                pending.remove(request_id)

            inflight = self._inflight_request_ids_by_worker.get(record.worker_id, [])
            if request_id in inflight:
                inflight.remove(request_id)

            record.status = "stopped"
            record.error_message = "stopped_by_web_disconnect"
            record.finished_at = stopped_at

        await self._append_event(
            request_id,
            {
                "type": "failed",
                "workerId": record.worker_id,
                "requestId": request_id,
                "errorMessage": "当前页面已停止接收，本地任务可能仍在继续执行。",
                "finishedAt": stopped_at,
            },
        )
        await self._finish_request(request_id)
        return stopped_at

    async def emit_submitted(self, request_id: str) -> None:
        record = self._requests[request_id]
        await self._append_event(
            request_id,
            {
                "type": "submitted",
                "workerId": record.worker_id,
                "requestId": request_id,
                "ts": _utc_now_iso(),
            },
            session_role="system",
            session_content="消息已提交到服务端。",
            session_status="submitted",
        )

    async def emit_processing(self, request_id: str) -> None:
        now = _utc_now_iso()
        async with self._lock:
            record = self._requests.get(request_id)
            if record is None or record.is_terminal:
                return
            if record.status == "processing":
                return
            record.status = "processing"
            record.delivered_at = now
            worker_id = record.worker_id

        await self._append_event(
            request_id,
            {
                "type": "processing",
                "workerId": worker_id,
                "requestId": request_id,
                "ts": now,
            },
            session_role="system",
            session_content="本地终端已开始处理当前消息。",
            session_status="processing",
        )

    async def complete_request(self, request_id: str, *, final_content: str) -> None:
        finished_at = _utc_now_iso()
        async with self._lock:
            record = self._requests.get(request_id)
            if record is None or record.is_terminal:
                return
            record.status = "completed"
            record.final_content = final_content
            record.finished_at = finished_at
            self._last_completed_at_by_worker[record.worker_id] = finished_at
            worker_id = record.worker_id

            session = self._find_session(record.session_id)
            if session is not None:
                session.messages.append(
                    _WebConsoleMessage(
                        id=f"assistant-{request_id}",
                        role="assistant",
                        content=final_content,
                        created_at=finished_at,
                        status="completed",
                    )
                )
                session.last_message = final_content
                session.updated_at = finished_at

        await self._append_event(
            request_id,
            {
                "type": "completed",
                "workerId": worker_id,
                "requestId": request_id,
                "finalContent": final_content,
                "finishedAt": finished_at,
            },
        )
        await self._finish_request(request_id)

    async def append_progress_event(self, request_id: str, *, content: str) -> None:
        content = content.strip()
        if not content:
            return
        async with self._lock:
            record = self._requests.get(request_id)
            if record is None or record.is_terminal:
                return
            worker_id = record.worker_id
        await self._append_event(
            request_id,
            {
                "type": "progress",
                "workerId": worker_id,
                "requestId": request_id,
                "content": content,
                "ts": _utc_now_iso(),
            },
        )

    async def fail_request(self, request_id: str, error_message: str) -> None:
        finished_at = _utc_now_iso()
        async with self._lock:
            record = self._requests.get(request_id)
            if record is None or record.is_terminal:
                return
            record.status = "failed"
            record.error_message = error_message
            record.finished_at = finished_at
            worker_id = record.worker_id

            session = self._find_session(record.session_id)
            if session is not None:
                session.messages.append(
                    _WebConsoleMessage(
                        id=f"error-{request_id}",
                        role="error",
                        content=error_message,
                        created_at=finished_at,
                        status="failed",
                    )
                )
                session.updated_at = finished_at

        await self._append_event(
            request_id,
            {
                "type": "failed",
                "workerId": worker_id,
                "requestId": request_id,
                "errorMessage": error_message,
                "finishedAt": finished_at,
            },
        )
        await self._finish_request(request_id)

    async def _append_event(
        self,
        request_id: str,
        payload: dict[str, Any],
        *,
        session_role: str | None = None,
        session_content: str | None = None,
        session_status: str | None = None,
    ) -> None:
        subscribers: list[asyncio.Queue[dict[str, Any] | None]] = []
        async with self._lock:
            record = self._requests.get(request_id)
            if record is None:
                return
            record.events.append(payload)
            subscribers = list(record.subscribers)

            if session_role and session_content is not None:
                session = self._find_session(record.session_id)
                if session is not None:
                    event_time = payload.get("ts") or payload.get("finishedAt") or _utc_now_iso()
                    session.messages.append(
                        _WebConsoleMessage(
                            id=f"{session_role}-{request_id}-{len(record.events)}",
                            role=session_role,
                            content=session_content,
                            created_at=event_time,
                            status=session_status,
                        )
                    )
                    session.updated_at = event_time
                    session.last_message = session.last_message or record.content

        for queue in subscribers:
            await queue.put(payload)

    async def _finish_request(self, request_id: str) -> None:
        subscribers: list[asyncio.Queue[dict[str, Any] | None]] = []
        async with self._lock:
            record = self._requests.get(request_id)
            if record is None:
                return
            subscribers = list(record.subscribers)
            record.subscribers.clear()
            if self._active_web_request_id_by_worker.get(record.worker_id) == request_id:
                self._active_web_request_id_by_worker.pop(record.worker_id, None)
        for queue in subscribers:
            await queue.put(None)

    def mark_worker_online(self, *, worker_id: str) -> None:
        self._online_workers[worker_id] = {
            "last_seen_at": time.monotonic(),
            "online": True,
        }
        self.notify_stream(worker_id)

    def mark_worker_seen(self, *, worker_id: str) -> None:
        record = self._online_workers.get(worker_id)
        if record is None:
            return
        record["last_seen_at"] = time.monotonic()

    def mark_worker_offline(self, *, worker_id: str) -> None:
        record = self._online_workers.get(worker_id)
        if record is None:
            return
        record["online"] = False
        record["last_seen_at"] = time.monotonic()
        self.notify_stream(worker_id)

    def is_worker_online(self, worker_id: str) -> bool:
        record = self._online_workers.get(worker_id)
        if record is None or not record.get("online", False):
            return False
        last_seen_at = float(record.get("last_seen_at", 0.0))
        if time.monotonic() - last_seen_at > self.worker_ttl_seconds:
            record["online"] = False
            return False
        return True

    def register_stream(self, *, worker_id: str) -> int:
        version = self._stream_versions.get(worker_id, 0) + 1
        self._stream_versions[worker_id] = version
        self.mark_worker_seen(worker_id=worker_id)
        self.notify_stream(worker_id)
        return version

    def is_current_stream(self, *, worker_id: str, version: int) -> bool:
        return self._stream_versions.get(worker_id) == version

    async def wait_for_stream_activity(self, *, worker_id: str, timeout: float) -> bool:
        event = self._get_stream_notification(worker_id)
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
            return True
        except TimeoutError:
            return False
        finally:
            event.clear()

    def notify_stream(self, worker_id: str) -> None:
        self._get_stream_notification(worker_id).set()

    def _get_stream_notification(self, worker_id: str) -> asyncio.Event:
        event = self._stream_notifications.get(worker_id)
        if event is None:
            event = asyncio.Event()
            self._stream_notifications[worker_id] = event
        return event

    def _find_session(self, session_id: str) -> _WebConsoleSession | None:
        for sessions in self._sessions_by_worker.values():
            for session in sessions:
                if session.id == session_id:
                    return session
        return None

    def _normalize_source(self, source: str | None) -> str:
        normalized = (source or "").strip().lower()
        if normalized in {"im", "web"}:
            return normalized
        return "im"


def install_web_console_routes(app: FastAPI) -> None:
    state = _WebConsoleState()
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

    @app.post(
        "/web-console/messages",
        response_model=SubmitMessageResponse,
        tags=["WebConsole"],
    )
    async def submit_message(request: SubmitMessageRequest):
        try:
            record = await state.register_submit(
                worker_id=request.workerId,
                session_id=request.sessionId,
                content=request.content,
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return SubmitMessageResponse(ok=True, requestId=record.request_id, acceptedAt=record.accepted_at)

    @app.get("/web-console/workers/{worker_id}/events", tags=["WebConsole"])
    async def get_worker_events(worker_id: str):
        async def event_stream():
            try:
                request_id, history, is_terminal = await state.get_active_request_for_worker(worker_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=f"No active web request for worker {worker_id}") from exc

            for payload in history:
                yield _serialize_sse(payload)
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
                    yield _serialize_sse(payload)
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

    @app.get(
        "/web-console/messages/{request_id}/events",
        tags=["WebConsole"],
    )
    async def get_message_events(request_id: str):
        async def event_stream():
            try:
                history, is_terminal = await state.get_request_history(request_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=f"Request {request_id} not found") from exc

            for payload in history:
                yield _serialize_sse(payload)
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
                    yield _serialize_sse(payload)
            finally:
                await state.remove_subscriber(request_id, queue)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get(
        "/web-console/messages/{request_id}/result",
        response_model=PollResultResponse,
        tags=["WebConsole"],
    )
    async def get_message_result(request_id: str):
        try:
            return await state.get_request_result(request_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Request {request_id} not found") from exc

    @app.post("/api/worker/online", tags=["Worker"])
    async def worker_online(request: WorkerRequest) -> dict[str, bool]:
        state.mark_worker_online(worker_id=request.worker_id)
        return {"ok": True}

    @app.post("/api/worker/offline", tags=["Worker"])
    async def worker_offline(request: WorkerRequest) -> dict[str, bool]:
        state.mark_worker_offline(worker_id=request.worker_id)
        return {"ok": True}

    @app.post("/api/worker/poll", tags=["Worker"])
    async def worker_poll(request: WorkerRequest) -> dict[str, Any]:
        state.mark_worker_seen(worker_id=request.worker_id)
        record = await state.worker_dequeue_next(request.worker_id)
        if record is None:
            return {"messages": []}
        return {
            "messages": [
                {
                    "content": record.content,
                    "worker_id": request.worker_id,
                    "source": record.source,
                }
            ]
        }

    @app.get("/api/worker/stream", tags=["Worker"])
    async def worker_stream(worker_id: str) -> StreamingResponse:
        stream_version = state.register_stream(worker_id=worker_id)

        async def event_stream():
            yield _serialize_worker_sse("ready", {"worker_id": worker_id})
            while state.is_worker_online(worker_id) and state.is_current_stream(
                worker_id=worker_id,
                version=stream_version,
            ):
                record = await state.worker_dequeue_next(worker_id)
                if record is not None:
                    state.mark_worker_seen(worker_id=worker_id)
                    yield _serialize_worker_sse(
                        "message",
                        {
                            "content": record.content,
                            "source": record.source,
                        },
                    )
                    continue

                has_activity = await state.wait_for_stream_activity(
                    worker_id=worker_id,
                    timeout=state.stream_heartbeat_interval_seconds,
                )
                if not state.is_current_stream(worker_id=worker_id, version=stream_version):
                    yield _serialize_worker_sse(
                        "error",
                        {"code": "replaced", "message": "connection replaced by newer stream"},
                    )
                    break
                if not state.is_worker_online(worker_id):
                    break

                state.mark_worker_seen(worker_id=worker_id)
                if not has_activity:
                    yield _serialize_worker_sse("heartbeat", {"ts": time.time()})

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    @app.post("/api/worker/complete", tags=["Worker"])
    async def worker_complete(request: WorkerCompleteRequest) -> dict[str, bool]:
        ok = await state.worker_complete_next(
            worker_id=request.worker_id,
            final_content=request.final_content,
            source=request.source,
        )
        return {"ok": ok}

    @app.post("/api/worker/progress", tags=["Worker"])
    async def worker_progress(request: WorkerProgressRequest) -> dict[str, bool]:
        ok = await state.worker_append_progress(
            worker_id=request.worker_id,
            content=request.content,
            source=request.source,
        )
        return {"ok": ok}

    @app.get("/mock/online", tags=["WorkerDebug"])
    async def list_online() -> dict[str, Any]:
        return {
            "online_workers": {
                worker_id: record
                for worker_id, record in state._online_workers.items()
                if state.is_worker_online(worker_id)
            }
        }

    @app.get("/mock/messages", tags=["WorkerDebug"])
    async def list_pending_messages() -> dict[str, Any]:
        pending: list[dict[str, Any]] = []
        for worker_id, request_ids in state._pending_request_ids_by_worker.items():
            for request_id in request_ids:
                record = state._requests.get(request_id)
                if record is None:
                    continue
                pending.append(
                    {
                        "request_id": request_id,
                        "worker_id": worker_id,
                        "content": record.content,
                        "source": record.source,
                    }
                )
        return {"messages": pending}

    @app.get("/mock/completions", tags=["WorkerDebug"])
    async def list_completions() -> dict[str, Any]:
        completed = [
            {
                "request_id": record.request_id,
                "worker_id": record.worker_id,
                "final_content": record.final_content,
                "source": record.source,
                "finished_at": record.finished_at,
            }
            for record in state._requests.values()
            if record.status == "completed"
        ]
        return {"completions": completed}

    @app.post("/mock/messages", tags=["WorkerDebug"])
    async def enqueue_mock_message(request: SubmitMessageRequest):
        if not state.is_worker_online(request.workerId):
            return JSONResponse(
                status_code=409,
                content={"ok": False, "error": "no_online_worker", "worker_id": request.workerId},
            )
        try:
            record = await state.register_submit(
                worker_id=request.workerId,
                session_id=request.sessionId,
                content=request.content,
            )
        except ValueError as exc:
            return JSONResponse(status_code=409, content={"ok": False, "error": str(exc)})
        return {"ok": True, "requestId": record.request_id}
