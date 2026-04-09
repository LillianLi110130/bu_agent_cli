"""A tiny mock gateway server for local worker development and tests."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import time
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
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
    sent_texts: list[dict[str, Any]] = field(default_factory=list)
    uploaded_attachments: list[dict[str, Any]] = field(default_factory=list)
    online_workers: dict[str, dict[str, Any]] = field(default_factory=dict)
    worker_ttl_seconds: float = 30.0
    inflight_messages: dict[str, list[MockWorkerMessage]] = field(default_factory=dict)

    def enqueue_message(self, *, worker_id: str, content: str) -> MockWorkerMessage:
        if not self.is_online(worker_id):
            raise LookupError(f"no_online_worker:{worker_id}")
        message = MockWorkerMessage(worker_id=worker_id, content=content)
        self.queued_messages.append(message)
        return message

    def mark_online(self, *, worker_id: str) -> None:
        self.online_workers[worker_id] = {
            "last_seen_at": time.monotonic(),
            "online": True,
        }

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

    def is_online(self, worker_id: str) -> bool:
        record = self.online_workers.get(worker_id)
        if record is None or not record.get("online", False):
            return False
        last_seen_at = float(record.get("last_seen_at", 0.0))
        if time.monotonic() - last_seen_at > self.worker_ttl_seconds:
            record["online"] = False
            return False
        return True


class WorkerRequest(BaseModel):
    worker_id: str


class CompleteRequest(BaseModel):
    worker_id: str
    final_content: str


class EnqueueRequest(BaseModel):
    worker_id: str
    content: str


class SendTextRequest(BaseModel):
    worker_id: str
    text: str


class UploadAttachmentRequest(BaseModel):
    worker_id: str
    file_name: str
    mime_type: str
    file_size: int
    file_content_base64: str


def create_mock_gateway_app(state: MockGatewayState | None = None) -> FastAPI:
    """Create a FastAPI app that mimics the worker gateway contract."""
    state = state or MockGatewayState()
    app = FastAPI(title="Mock Worker Gateway")
    app.state.mock_gateway = state

    @app.post("/api/worker/poll")
    async def poll(request: WorkerRequest) -> dict[str, Any]:
        state.mark_seen(worker_id=request.worker_id)
        for index, message in enumerate(list(state.queued_messages)):
            if message.worker_id != request.worker_id:
                continue
            queued = state.queued_messages.pop(index)
            inflight_queue = state.inflight_messages.setdefault(request.worker_id, [])
            inflight_queue.append(queued)
            return {"messages": [asdict(queued)]}
        return {"messages": []}

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
        inflight_queue = state.inflight_messages.get(request.worker_id, [])
        inflight = inflight_queue.pop(0) if inflight_queue else None
        if not inflight_queue:
            state.inflight_messages.pop(request.worker_id, None)
        state.completions.append(
            {
                "worker_id": request.worker_id,
                "input_content": inflight.content if inflight is not None else "",
                "final_content": request.final_content,
            }
        )
        return {"ok": True}

    @app.post("/api/worker/send_text")
    async def send_text(request: SendTextRequest) -> dict[str, bool]:
        state.sent_texts.append(
            {
                "worker_id": request.worker_id,
                "text": request.text,
            }
        )
        return {"ok": True}

    @app.post("/api/worker/upload_attachment")
    async def upload_attachment(request: UploadAttachmentRequest) -> dict[str, bool]:
        state.uploaded_attachments.append(
            {
                "worker_id": request.worker_id,
                "file_name": request.file_name,
                "mime_type": request.mime_type,
                "file_size": request.file_size,
                "file_content_base64": request.file_content_base64,
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
