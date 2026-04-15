"""A tiny mock gateway server for local worker development and tests."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass, field
import json
import time
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn


@dataclass
class MockWorkerMessage:
    """One fake worker message queued for delivery."""

    content: str
    worker_id: str


@dataclass
class MockGatewayState:
    """In-memory state for the mock gateway API."""

    queued_messages: list[MockWorkerMessage] = field(default_factory=list)
    completions: list[dict[str, Any]] = field(default_factory=list)
    online_workers: dict[str, dict[str, Any]] = field(default_factory=dict)
    worker_ttl_seconds: float = 30.0
    stream_heartbeat_interval_seconds: float = 0.1
    stream_versions: dict[str, int] = field(default_factory=dict)
    stream_notifications: dict[str, asyncio.Event] = field(default_factory=dict)

    def enqueue_message(self, *, worker_id: str, content: str) -> MockWorkerMessage:
        if not self.is_online(worker_id):
            raise LookupError(f"no_online_worker:{worker_id}")
        message = MockWorkerMessage(worker_id=worker_id, content=content)
        self.queued_messages.append(message)
        self.notify_stream(worker_id)
        return message

    def mark_online(self, *, worker_id: str) -> None:
        self.online_workers[worker_id] = {
            "last_seen_at": time.monotonic(),
            "online": True,
        }
        self.notify_stream(worker_id)

    def mark_seen(self, *, worker_id: str) -> None:
        record = self.online_workers.get(worker_id)
        if record is None:
            return
        record["last_seen_at"] = time.monotonic()

    def mark_offline(self, *, worker_id: str) -> None:
        record = self.online_workers.get(worker_id)
        if record is None:
            return
        record["online"] = False
        record["last_seen_at"] = time.monotonic()
        self.notify_stream(worker_id)

    def is_online(self, worker_id: str) -> bool:
        record = self.online_workers.get(worker_id)
        if record is None or not record.get("online", False):
            return False
        last_seen_at = float(record.get("last_seen_at", 0.0))
        if time.monotonic() - last_seen_at > self.worker_ttl_seconds:
            record["online"] = False
            return False
        return True

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


class WorkerRequest(BaseModel):
    worker_id: str


class CompleteRequest(BaseModel):
    worker_id: str
    final_content: str


class EnqueueRequest(BaseModel):
    worker_id: str
    content: str


def create_mock_gateway_app(state: MockGatewayState | None = None) -> FastAPI:
    """Create a FastAPI app that mimics the worker gateway contract."""
    state = state or MockGatewayState()
    app = FastAPI(title="Mock Worker Gateway")
    app.state.mock_gateway = state

    @app.post("/api/worker/poll")
    async def poll(request: WorkerRequest) -> dict[str, Any]:
        state.mark_seen(worker_id=request.worker_id)
        queued = state.dequeue_message(worker_id=request.worker_id)
        if queued is not None:
            return {"messages": [asdict(queued)]}
        return {"messages": []}

    @app.get("/api/worker/stream")
    async def stream(worker_id: str) -> StreamingResponse:
        stream_version = state.register_stream(worker_id=worker_id)

        async def event_stream():
            yield _encode_sse("ready", {"worker_id": worker_id})
            while state.is_online(worker_id) and state.is_current_stream(
                worker_id=worker_id,
                version=stream_version,
            ):
                queued = state.dequeue_message(worker_id=worker_id)
                if queued is not None:
                    state.mark_seen(worker_id=worker_id)
                    yield _encode_sse("message", {"content": queued.content})
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

    @app.post("/api/worker/online")
    async def online(request: WorkerRequest) -> dict[str, bool]:
        state.mark_online(worker_id=request.worker_id)
        return {"ok": True}

    @app.post("/api/worker/offline")
    async def offline(request: WorkerRequest) -> dict[str, bool]:
        state.mark_offline(worker_id=request.worker_id)
        return {"ok": True}

    @app.post("/api/worker/complete")
    async def complete(request: CompleteRequest) -> dict[str, bool]:
        state.completions.append(
            {
                "worker_id": request.worker_id,
                "final_content": request.final_content,
                "completed_at": time.time(),
            }
        )
        return {"ok": True}

    @app.post("/mock/messages")
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
        )
        return {
            "ok": True,
            "worker_id": message.worker_id,
        }

    @app.get("/mock/messages")
    async def list_messages() -> dict[str, Any]:
        return {"messages": [asdict(message) for message in state.queued_messages]}

    @app.get("/mock/completions")
    async def list_completions() -> dict[str, Any]:
        return {"completions": list(state.completions)}

    @app.get("/mock/online")
    async def list_online() -> dict[str, Any]:
        online = {
            worker_id: record
            for worker_id, record in state.online_workers.items()
            if state.is_online(worker_id)
        }
        return {"online_workers": online}

    return app


def _encode_sse(event: str, payload: dict[str, Any]) -> bytes:
    """Encode one SSE event frame."""
    data = json.dumps(payload, ensure_ascii=False)
    return f"event: {event}\ndata: {data}\n\n".encode("utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the mock gateway server."""
    parser = argparse.ArgumentParser(description="Mock worker gateway server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def cli_main() -> None:
    """Run the mock gateway server with uvicorn."""
    args = parse_args()
    uvicorn.run(create_mock_gateway_app(), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    cli_main()
