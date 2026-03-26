from __future__ import annotations

import importlib
from pathlib import Path

import pytest


def _load_module(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        pytest.fail(f"Expected module '{module_name}' to exist: {exc}")


def test_gateway_settings_reads_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_module = _load_module("agent_core.gateway.config")

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:abc")
    monkeypatch.setenv("TELEGRAM_ALLOW_FROM", "111,222")
    monkeypatch.setenv("GATEWAY_HEARTBEAT_ENABLED", "false")
    monkeypatch.setenv("GATEWAY_HEARTBEAT_INTERVAL_SECONDS", "120")
    monkeypatch.setenv("ZHAOHU_ENABLED", "true")
    monkeypatch.setenv("ZHAOHU_HOST", "127.0.0.1")
    monkeypatch.setenv("ZHAOHU_PORT", "19090")
    monkeypatch.setenv("ZHAOHU_WEBHOOK_PATH", "/hooks/zhaohu")

    settings = config_module.GatewaySettings.from_env(root_dir=tmp_path, model="glm")

    assert settings.telegram_bot_token == "123:abc"
    assert settings.telegram_allow_from == ["111", "222"]
    assert settings.heartbeat_enabled is False
    assert settings.heartbeat_interval_seconds == 120
    assert settings.zhaohu_enabled is True
    assert settings.zhaohu_host == "127.0.0.1"
    assert settings.zhaohu_port == 19090
    assert settings.zhaohu_webhook_path == "/hooks/zhaohu"
    assert settings.root_dir == tmp_path
    assert settings.model == "glm"


def test_gateway_settings_defaults_when_env_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_module = _load_module("agent_core.gateway.config")

    for key in [
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_ALLOW_FROM",
        "GATEWAY_HEARTBEAT_ENABLED",
        "GATEWAY_HEARTBEAT_INTERVAL_SECONDS",
        "ZHAOHU_ENABLED",
        "ZHAOHU_HOST",
        "ZHAOHU_PORT",
        "ZHAOHU_WEBHOOK_PATH",
    ]:
        monkeypatch.delenv(key, raising=False)

    settings = config_module.GatewaySettings.from_env(root_dir=tmp_path, model=None)

    assert settings.telegram_bot_token == ""
    assert settings.telegram_allow_from == []
    assert settings.heartbeat_enabled is True
    assert settings.heartbeat_interval_seconds == 1800
    assert settings.zhaohu_enabled is False
    assert settings.zhaohu_host == "0.0.0.0"
    assert settings.zhaohu_port == 18080
    assert settings.zhaohu_webhook_path == "/webhook/zhaohu"
    assert settings.model is None


def test_gateway_main_loads_dotenv_before_resolving_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    main_module = _load_module("agent_core.gateway.main")

    env_path = tmp_path / ".env"
    env_path.write_text(
        "TELEGRAM_BOT_TOKEN=dotenv-token\n"
        "TELEGRAM_ALLOW_FROM=111,222\n"
        "GATEWAY_HEARTBEAT_ENABLED=false\n"
        "ZHAOHU_ENABLED=true\n"
        "ZHAOHU_HOST=127.0.0.1\n"
        "ZHAOHU_PORT=19091\n"
        "ZHAOHU_WEBHOOK_PATH=/dotenv/zhaohu\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    for key in [
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_ALLOW_FROM",
        "GATEWAY_HEARTBEAT_ENABLED",
        "ZHAOHU_ENABLED",
        "ZHAOHU_HOST",
        "ZHAOHU_PORT",
        "ZHAOHU_WEBHOOK_PATH",
    ]:
        monkeypatch.delenv(key, raising=False)

    args = main_module.argparse.Namespace(
        root_dir=None,
        model="glm",
        heartbeat_interval_seconds=None,
        disable_heartbeat=False,
    )

    settings = main_module.load_settings_from_args(args)

    assert settings.telegram_bot_token == "dotenv-token"
    assert settings.telegram_allow_from == ["111", "222"]
    assert settings.heartbeat_enabled is False
    assert settings.zhaohu_enabled is True
    assert settings.zhaohu_host == "127.0.0.1"
    assert settings.zhaohu_port == 19091
    assert settings.zhaohu_webhook_path == "/dotenv/zhaohu"
    assert settings.root_dir == tmp_path.resolve()
    assert settings.model == "glm"


@pytest.mark.asyncio
async def test_run_gateway_registers_zhaohu_channel_when_enabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    main_module = _load_module("agent_core.gateway.main")

    registered_channels: list[object] = []

    class FakeChannelManager:
        def __init__(self, bus) -> None:
            self.bus = bus

        def register(self, channel) -> None:
            registered_channels.append(channel)

        async def start_all(self) -> None:
            return None

        async def stop_all(self) -> None:
            return None

    class FakeGatewayService:
        def __init__(self, dispatcher) -> None:
            self.dispatcher = dispatcher

        async def start(self) -> None:
            return None

        async def stop(self) -> None:
            return None

    class FakeHeartbeatService:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        async def start(self) -> None:
            return None

        async def stop(self) -> None:
            return None

    class FakeRuntimeManager:
        def __init__(self, runtime_factory) -> None:
            self.runtime_factory = runtime_factory

    class FakeZhaohuChannel:
        def __init__(self, config, bus) -> None:
            self.config = config
            self.bus = bus
            self.name = "zhaohu"

    class FakeStopEvent:
        async def wait(self) -> None:
            raise RuntimeError("stop gateway")

    monkeypatch.setattr(main_module, "ChannelManager", FakeChannelManager)
    monkeypatch.setattr(main_module, "GatewayService", FakeGatewayService)
    monkeypatch.setattr(main_module, "HeartbeatService", FakeHeartbeatService)
    monkeypatch.setattr(main_module, "RuntimeManager", FakeRuntimeManager)
    monkeypatch.setattr(main_module, "ZhaohuChannel", FakeZhaohuChannel)
    monkeypatch.setattr(main_module, "create_llm", lambda _model: object())
    monkeypatch.setattr(main_module, "create_agent", lambda model, root_dir: (model, root_dir))
    monkeypatch.setattr(main_module.asyncio, "Event", lambda: FakeStopEvent())

    settings = main_module.GatewaySettings(
        root_dir=tmp_path,
        model="glm",
        zhaohu_enabled=True,
        zhaohu_host="127.0.0.1",
        zhaohu_port=18080,
        zhaohu_webhook_path="/webhook/zhaohu",
        heartbeat_enabled=False,
    )

    with pytest.raises(RuntimeError, match="stop gateway"):
        await main_module.run_gateway(settings)

    assert len(registered_channels) == 1
    channel = registered_channels[0]
    assert isinstance(channel, FakeZhaohuChannel)
    assert channel.config.host == "127.0.0.1"
    assert channel.config.port == 18080
    assert channel.config.webhook_path == "/webhook/zhaohu"
