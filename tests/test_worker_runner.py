from __future__ import annotations

import asyncio
import json
import shutil
import uuid
from pathlib import Path

import httpx
import pytest

from cli.im_bridge import FileBridgeStore, resolve_session_binding_id
from cli.worker.gateway_client import WorkerGatewayClient
from cli.worker.mock_server import MockGatewayState, create_mock_gateway_app
from cli.worker.runner import WorkerRunner


async def _wait_until(predicate, *, timeout: float = 2.0, interval: float = 0.01) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(interval)
    raise AssertionError("condition not reached before timeout")


@pytest.fixture
def workspace_root() -> Path:
    root = Path(".pytest_tmp") / f"worker-runner-{uuid.uuid4().hex}"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    try:
        yield root.resolve()
    finally:
        if root.exists():
            shutil.rmtree(root)


@pytest.mark.asyncio
async def test_worker_runner_bridges_remote_message_and_completes(workspace_root: Path):
    state = MockGatewayState()
    app = create_mock_gateway_app(state)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as http_client:
        gateway_client = WorkerGatewayClient(base_url="http://testserver", client=http_client)
        runner = WorkerRunner(
            worker_id="worker-1",
            gateway_client=gateway_client,
            model=None,
            root_dir=workspace_root,
            result_poll_interval_seconds=0.01,
        )
        task = asyncio.create_task(runner.run_forever())

        store = FileBridgeStore(
            workspace_root,
            session_binding_id=resolve_session_binding_id("worker-1"),
        )

        await _wait_until(lambda: state.is_online("worker-1"))
        state.enqueue_message(worker_id="worker-1", content="remote hello")
        await _wait_until(lambda: store.pending_count() == 1)
        claimed = store.claim_next_pending()
        assert claimed is not None
        assert claimed.content == "remote hello"
        store.complete_request(claimed, final_content="remote done")

        await _wait_until(lambda: len(state.completions) == 1)
        runner.stop()
        await asyncio.wait_for(task, timeout=1.0)

        assert state.is_online("worker-1") is False
        assert state.completions[0]["worker_id"] == "worker-1"
        assert state.completions[0]["input_content"] == "remote hello"
        assert state.completions[0]["final_content"] == "remote done"


@pytest.mark.asyncio
async def test_worker_runner_uses_local_request_id_for_remote_message(workspace_root: Path):
    state = MockGatewayState()
    app = create_mock_gateway_app(state)
    transport = httpx.ASGITransport(app=app)

    store = FileBridgeStore(
        workspace_root,
        session_binding_id=resolve_session_binding_id("worker-1"),
    )

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as http_client:
        gateway_client = WorkerGatewayClient(base_url="http://testserver", client=http_client)
        runner = WorkerRunner(
            worker_id="worker-1",
            gateway_client=gateway_client,
            model=None,
            root_dir=workspace_root,
            result_poll_interval_seconds=0.01,
        )
        task = asyncio.create_task(runner.run_forever())

        await _wait_until(lambda: state.is_online("worker-1"))
        state.enqueue_message(worker_id="worker-1", content="remote cached")

        await _wait_until(lambda: store.pending_count() == 1)
        pending_path = next(store.inbox_pending_dir.glob("*.json"))
        payload = json.loads(pending_path.read_text(encoding="utf-8"))
        request_id = str(payload["request_id"])
        claimed = store.claim_next_pending()
        assert claimed is not None
        store.complete_request(claimed, final_content="cached done")

        await _wait_until(lambda: len(state.completions) == 1)
        runner.stop()
        await asyncio.wait_for(task, timeout=1.0)

        assert request_id.startswith("req_")
        assert state.completions[0]["worker_id"] == "worker-1"
        assert state.completions[0]["input_content"] == "remote cached"
        assert state.completions[0]["final_content"] == "cached done"
        assert store.pending_count() == 0


@pytest.mark.asyncio
async def test_mock_gateway_rejects_messages_when_no_worker_is_online():
    state = MockGatewayState()
    app = create_mock_gateway_app(state)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as http_client:
        response = await http_client.post(
            "/mock/messages",
            json={
                "worker_id": "worker-1",
                "content": "hello",
            },
        )

    assert response.status_code == 409
    payload = response.json()
    assert payload["ok"] is False
    assert payload["error"] == "no_online_worker"


@pytest.mark.asyncio
async def test_mock_gateway_assigns_message_and_delivery_ids():
    state = MockGatewayState()
    state.mark_online(worker_id="worker-1")
    app = create_mock_gateway_app(state)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as http_client:
        response = await http_client.post(
            "/mock/messages",
            json={
                "worker_id": "worker-1",
                "content": "hello",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert state.queued_messages[0].worker_id == "worker-1"
