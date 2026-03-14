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
    config_module = _load_module("bu_agent_sdk.gateway.config")

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:abc")
    monkeypatch.setenv("TELEGRAM_ALLOW_FROM", "111,222")
    monkeypatch.setenv("GATEWAY_HEARTBEAT_ENABLED", "false")
    monkeypatch.setenv("GATEWAY_HEARTBEAT_INTERVAL_SECONDS", "120")

    settings = config_module.GatewaySettings.from_env(root_dir=tmp_path, model="glm")

    assert settings.telegram_bot_token == "123:abc"
    assert settings.telegram_allow_from == ["111", "222"]
    assert settings.heartbeat_enabled is False
    assert settings.heartbeat_interval_seconds == 120
    assert settings.root_dir == tmp_path
    assert settings.model == "glm"


def test_gateway_settings_defaults_when_env_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_module = _load_module("bu_agent_sdk.gateway.config")

    for key in [
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_ALLOW_FROM",
        "GATEWAY_HEARTBEAT_ENABLED",
        "GATEWAY_HEARTBEAT_INTERVAL_SECONDS",
    ]:
        monkeypatch.delenv(key, raising=False)

    settings = config_module.GatewaySettings.from_env(root_dir=tmp_path, model=None)

    assert settings.telegram_bot_token == ""
    assert settings.telegram_allow_from == []
    assert settings.heartbeat_enabled is True
    assert settings.heartbeat_interval_seconds == 1800
    assert settings.model is None


def test_gateway_main_loads_dotenv_before_resolving_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    main_module = _load_module("bu_agent_sdk.gateway.main")

    env_path = tmp_path / ".env"
    env_path.write_text(
        "TELEGRAM_BOT_TOKEN=dotenv-token\n"
        "TELEGRAM_ALLOW_FROM=111,222\n"
        "GATEWAY_HEARTBEAT_ENABLED=false\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    for key in [
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_ALLOW_FROM",
        "GATEWAY_HEARTBEAT_ENABLED",
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
    assert settings.root_dir == tmp_path.resolve()
    assert settings.model == "glm"
