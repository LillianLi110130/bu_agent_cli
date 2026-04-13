from __future__ import annotations

import json
from pathlib import Path
import shutil
import uuid

import httpx
import pytest

from cli.worker.gateway_client import WorkerGatewayClient


def test_gateway_client_disables_env_proxy_for_loopback(monkeypatch):
    captured: dict[str, object] = {}

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    WorkerGatewayClient(base_url="http://127.0.0.1:8765")

    assert captured["trust_env"] is False


def test_gateway_client_keeps_env_proxy_for_remote_host(monkeypatch):
    captured: dict[str, object] = {}

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    WorkerGatewayClient(base_url="https://example.com")

    assert captured["trust_env"] is True


@pytest.fixture
def workspace_root() -> Path:
    root = Path(".pytest_tmp") / f"worker-gateway-client-{uuid.uuid4().hex}"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    try:
        yield root.resolve()
    finally:
        if root.exists():
            shutil.rmtree(root)


@pytest.mark.asyncio
async def test_gateway_client_refreshes_authorization_from_poll_response(workspace_root: Path):
    token_path = workspace_root / ".tg_agent" / "token.json"
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(
        json.dumps(
            {
                "authorization": "Bearer old-token",
                "user_id": "user-123",
            }
        ),
        encoding="utf-8",
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer old-token"
        return httpx.Response(
            200,
            headers={"Authorization": "Bearer new-token"},
            json={"messages": [{"content": "hello"}]},
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as http_client:
        client = WorkerGatewayClient(
            base_url="http://testserver",
            client=http_client,
            authorization="Bearer old-token",
            base_dir=workspace_root,
        )
        messages = await client.poll("worker-1")

    assert [message.content for message in messages] == ["hello"]
    assert client.authorization == "Bearer new-token"
    token_payload = json.loads(token_path.read_text(encoding="utf-8"))
    assert token_payload["authorization"] == "Bearer new-token"
    assert token_payload["user_id"] == "user-123"


@pytest.mark.asyncio
async def test_gateway_client_retries_once_after_refreshing_authorization(workspace_root: Path):
    token_path = workspace_root / ".tg_agent" / "token.json"
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(
        json.dumps(
            {
                "authorization": "Bearer old-token",
                "user_id": "user-123",
            }
        ),
        encoding="utf-8",
    )

    seen_authorizations: list[str | None] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen_authorizations.append(request.headers.get("Authorization"))
        if len(seen_authorizations) == 1:
            return httpx.Response(
                401,
                headers={"Authorization": "Bearer refreshed-token"},
                json={"detail": "token expired"},
            )
        return httpx.Response(
            200,
            headers={"Authorization": "Bearer refreshed-token"},
            json={"ok": True},
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as http_client:
        client = WorkerGatewayClient(
            base_url="http://testserver",
            client=http_client,
            authorization="Bearer old-token",
            base_dir=workspace_root,
        )
        ok = await client.complete(worker_id="worker-1", final_content="done")

    assert ok is True
    assert seen_authorizations == ["Bearer old-token", "Bearer refreshed-token"]
    assert client.authorization == "Bearer refreshed-token"
    token_payload = json.loads(token_path.read_text(encoding="utf-8"))
    assert token_payload["authorization"] == "Bearer refreshed-token"
