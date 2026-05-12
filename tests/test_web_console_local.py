from __future__ import annotations

import asyncio
from pathlib import Path
import shutil
import uuid

import httpx
import pytest

from cli.im_bridge import FileBridgeStore, resolve_session_binding_id
from cli.worker.mock_server import MockGatewayState, create_mock_gateway_app
from cli.worker.gateway_client import WorkerGatewayClient
from cli.worker.runner import WorkerRunner


async def _wait_until(predicate, *, timeout: float = 2.0, interval: float = 0.01) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(interval)
    raise AssertionError("condition not reached before timeout")


async def _wait_for_result_status(
    client: httpx.AsyncClient,
    request_id: str,
    expected_status: str,
    *,
    timeout: float = 2.0,
    interval: float = 0.02,
) -> dict:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        response = await client.get(f"/web-console/messages/{request_id}/result")
        payload = response.json()
        if payload["status"] == expected_status:
            return payload
        await asyncio.sleep(interval)
    raise AssertionError(f"request {request_id} did not reach status {expected_status} before timeout")


@pytest.fixture
def workspace_root() -> Path:
    root = Path(".pytest_tmp") / f"web-console-relay-{uuid.uuid4().hex}"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    try:
        yield root.resolve()
    finally:
        if root.exists():
            shutil.rmtree(root)


def test_build_app_registers_web_console_and_worker_routes() -> None:
    app = create_mock_gateway_app(MockGatewayState())

    route_paths = {route.path for route in app.routes}
    assert "/web-console/messages" in route_paths
    assert "/web-console/workers/{worker_id}/events" in route_paths
    assert "/web-console/workers/{worker_id}/sessions" in route_paths
    assert "/api/worker/poll" in route_paths
    assert "/api/worker/stream" in route_paths
    assert "/api/worker/complete" in route_paths
    assert "/api/worker/progress" in route_paths


@pytest.mark.asyncio
async def test_web_console_message_stays_pending_without_worker(workspace_root: Path) -> None:
    app = create_mock_gateway_app(MockGatewayState())
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        submit_response = await client.post(
            "/web-console/messages",
            json={
                "workerId": "worker-hk-01",
                "sessionId": "session-packaging",
                "content": "hello web console",
            },
        )
        assert submit_response.status_code == 200
        request_id = submit_response.json()["requestId"]

        await asyncio.sleep(0.05)

        result_response = await client.get(f"/web-console/messages/{request_id}/result")
        assert result_response.status_code == 200
        payload = result_response.json()
        assert payload["status"] == "submitted"
        assert payload["finalContent"] is None


@pytest.mark.asyncio
async def test_web_console_message_flows_through_worker_bridge(workspace_root: Path) -> None:
    app = create_mock_gateway_app(MockGatewayState())
    transport = httpx.ASGITransport(app=app)
    state = app.state.web_console_state

    store = FileBridgeStore(
        workspace_root,
        session_binding_id=resolve_session_binding_id("worker-hk-01"),
    )

    async with (
        httpx.AsyncClient(transport=transport, base_url="http://testserver") as web_client,
        httpx.AsyncClient(transport=transport, base_url="http://testserver") as worker_http_client,
    ):
        gateway_client = WorkerGatewayClient(base_url="http://testserver", client=worker_http_client)
        runner = WorkerRunner(
            worker_id="worker-hk-01",
            gateway_client=gateway_client,
            model=None,
            root_dir=workspace_root,
            gateway_transport="poll",
            result_poll_interval_seconds=0.01,
        )
        task = asyncio.create_task(runner.run_forever())

        try:
            submit_response = await web_client.post(
                "/web-console/messages",
                json={
                    "workerId": "worker-hk-01",
                    "sessionId": "session-packaging",
                    "content": "hello worker relay",
                },
            )
            assert submit_response.status_code == 200
            request_id = submit_response.json()["requestId"]

            await _wait_until(lambda: store.pending_count() == 1)
            claimed = store.claim_next_pending()
            assert claimed is not None
            assert claimed.content == "hello worker relay"
            assert claimed.source_meta["origin"] == "web"
            store.record_progress(claimed, "partial relay")

            await _wait_until(
                lambda: any(
                    event.get("type") == "progress" and event.get("content") == "partial relay"
                    for event in state._requests[request_id].events
                )
            )

            processing_payload = await _wait_for_result_status(web_client, request_id, "processing")
            assert processing_payload["finalContent"] is None

            store.complete_request(claimed, final_content="relay done")

            result_payload = await _wait_for_result_status(web_client, request_id, "completed")
            assert result_payload["finalContent"] == "relay done"

            worker_summary = await web_client.get("/web-console/workers/worker-hk-01")
            assert worker_summary.status_code == 200
            assert worker_summary.json()["isOnline"] is True
        finally:
            runner.stop()
            await asyncio.wait_for(task, timeout=1.0)


@pytest.mark.asyncio
async def test_worker_stream_cleanup_stops_current_web_request(workspace_root: Path) -> None:
    app = create_mock_gateway_app(MockGatewayState())
    transport = httpx.ASGITransport(app=app)
    state = app.state.web_console_state

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as web_client:
        submit_response = await web_client.post(
            "/web-console/messages",
            json={
                "workerId": "worker-hk-01",
                "sessionId": "session-packaging",
                "content": "disconnect me",
            },
        )
        request_id = submit_response.json()["requestId"]

        await state.stop_request_if_current_worker_stream_closed(
            worker_id="worker-hk-01",
            request_id=request_id,
        )

        await _wait_until(lambda: state._requests[request_id].status == "stopped")
        result_payload = await _wait_for_result_status(web_client, request_id, "stopped")
        assert result_payload["errorMessage"] == "stopped_by_web_disconnect"


@pytest.mark.asyncio
async def test_rejects_second_active_web_request_for_same_worker(workspace_root: Path) -> None:
    app = create_mock_gateway_app(MockGatewayState())
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as web_client:
        first_submit = await web_client.post(
            "/web-console/messages",
            json={
                "workerId": "worker-hk-01",
                "sessionId": "session-packaging",
                "content": "first request",
            },
        )
        assert first_submit.status_code == 200

        second_submit = await web_client.post(
            "/web-console/messages",
            json={
                "workerId": "worker-hk-01",
                "sessionId": "session-packaging",
                "content": "second request",
            },
        )
        assert second_submit.status_code == 409
        assert second_submit.json()["detail"] == "worker_busy"
