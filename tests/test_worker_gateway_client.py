from __future__ import annotations

import httpx

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
