from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient


def _load_module(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        pytest.fail(f"Expected module '{module_name}' to exist: {exc}")


class _FakeServer:
    def __init__(self, config) -> None:
        self.config = config
        self.should_exit = False
        self.force_exit = False
        self.serve_called = False

    async def serve(self) -> None:
        self.serve_called = True


def test_zhaohu_channel_healthz_returns_ok() -> None:
    queue_module = _load_module("bu_agent_sdk.bus.queue")
    zhaohu_module = _load_module("bu_agent_sdk.channels.zhaohu")

    bus = queue_module.MessageBus()
    config = SimpleNamespace(host="127.0.0.1", port=18080, webhook_path="/webhook/zhaohu")
    channel = zhaohu_module.ZhaohuChannel(config=config, bus=bus)

    with TestClient(channel._create_app()) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {
        "channel": "zhaohu",
        "status": "ok",
        "webhook_path": "/webhook/zhaohu",
    }


def test_zhaohu_channel_webhook_accepts_payload_and_publishes_inbound_message() -> None:
    queue_module = _load_module("bu_agent_sdk.bus.queue")
    zhaohu_module = _load_module("bu_agent_sdk.channels.zhaohu")

    bus = queue_module.MessageBus()
    config = SimpleNamespace(host="127.0.0.1", port=18080, webhook_path="/hooks/zhaohu")
    channel = zhaohu_module.ZhaohuChannel(config=config, bus=bus)

    payload = {
        "sender_id": "u123",
        "chat_id": "c456",
        "content": "hello zhaohu",
        "message_id": "m789",
        "session_key": "zhaohu:custom-session",
        "metadata": {"tenant": "demo"},
    }

    with TestClient(channel._create_app()) as client:
        response = client.post("/hooks/zhaohu", json=payload)

    assert response.status_code == 202
    assert response.json() == {
        "channel": "zhaohu",
        "status": "accepted",
        "message": "Zhaohu webhook payload accepted and queued.",
        "session_key": "zhaohu:custom-session",
    }
    assert bus.inbound_size == 1

    inbound = bus.inbound.get_nowait()
    assert inbound.channel == "zhaohu"
    assert inbound.sender_id == "u123"
    assert inbound.chat_id == "c456"
    assert inbound.content == "hello zhaohu"
    assert inbound.session_key_override == "zhaohu:custom-session"
    assert inbound.metadata["tenant"] == "demo"
    assert inbound.metadata["message_id"] == "m789"
    assert inbound.metadata["raw_payload"] == payload


def test_zhaohu_channel_webhook_rejects_invalid_payload() -> None:
    queue_module = _load_module("bu_agent_sdk.bus.queue")
    zhaohu_module = _load_module("bu_agent_sdk.channels.zhaohu")

    bus = queue_module.MessageBus()
    config = SimpleNamespace(host="127.0.0.1", port=18080, webhook_path="/hooks/zhaohu")
    channel = zhaohu_module.ZhaohuChannel(config=config, bus=bus)

    with TestClient(channel._create_app()) as client:
        response = client.post(
            "/hooks/zhaohu",
            json={"sender_id": "u123", "chat_id": "c456"},
        )

    assert response.status_code == 422
    assert bus.inbound_size == 0


@pytest.mark.asyncio
async def test_zhaohu_channel_start_builds_and_runs_uvicorn_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue_module = _load_module("bu_agent_sdk.bus.queue")
    zhaohu_module = _load_module("bu_agent_sdk.channels.zhaohu")

    bus = queue_module.MessageBus()
    config = SimpleNamespace(host="127.0.0.1", port=19090, webhook_path="/hooks/zhaohu")
    channel = zhaohu_module.ZhaohuChannel(config=config, bus=bus)

    captured: dict[str, object] = {}

    def _fake_config(app, host, port, log_level):
        captured["app"] = app
        captured["host"] = host
        captured["port"] = port
        captured["log_level"] = log_level
        return SimpleNamespace(app=app, host=host, port=port, log_level=log_level)

    server = _FakeServer(config=None)

    def _fake_server_factory(config):
        server.config = config
        return server

    monkeypatch.setattr(zhaohu_module.uvicorn, "Config", _fake_config)
    monkeypatch.setattr(zhaohu_module.uvicorn, "Server", _fake_server_factory)

    await channel.start()

    assert server.serve_called is True
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 19090
    assert captured["log_level"] == "info"


@pytest.mark.asyncio
async def test_zhaohu_channel_stop_signals_server_exit() -> None:
    queue_module = _load_module("bu_agent_sdk.bus.queue")
    zhaohu_module = _load_module("bu_agent_sdk.channels.zhaohu")

    bus = queue_module.MessageBus()
    config = SimpleNamespace(host="127.0.0.1", port=18080, webhook_path="/webhook/zhaohu")
    channel = zhaohu_module.ZhaohuChannel(config=config, bus=bus)
    channel._server = SimpleNamespace(should_exit=False, force_exit=False)
    channel._running = True

    await channel.stop()

    assert channel.is_running is False
    assert channel._server.should_exit is True
    assert channel._server.force_exit is False
