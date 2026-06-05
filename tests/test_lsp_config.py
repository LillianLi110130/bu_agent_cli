from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_core.lsp.config import load_lsp_config, parse_lsp_config


def test_parse_lsp_config_defaults_enabled() -> None:
    config = parse_lsp_config({})

    assert config.enabled is True
    assert config.auto_start is True
    assert ".py" in config.servers["python"].extensions


def test_load_lsp_config_reads_settings_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    settings_dir = home / ".tg_agent"
    settings_dir.mkdir(parents=True)
    (settings_dir / "settings.json").write_text(
        json.dumps(
            {
                "lsp": {
                    "enabled": True,
                    "servers": {
                        "python": {
                            "command": "custom-pyright",
                            "extensions": ["py", ".pyi"],
                        }
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(home))

    config = load_lsp_config()

    assert config.enabled is True
    assert config.servers["python"].command == "custom-pyright"
    assert config.servers["python"].extensions == [".py", ".pyi"]


def test_parse_lsp_config_can_disable_lsp() -> None:
    config = parse_lsp_config({"enabled": False})

    assert config.enabled is False


def test_parse_lsp_config_rejects_non_object_lsp() -> None:
    with pytest.raises(ValueError, match="servers"):
        parse_lsp_config({"servers": []})


def test_parse_lsp_config_reads_server_disabled_env_settings_and_init_options() -> None:
    config = parse_lsp_config(
        {
            "servers": {
                "python": {
                    "disabled": True,
                    "env": {"PYRIGHT_PYTHON_FORCE_VERSION": 3},
                    "settings": {"python": {"analysis": {"typeCheckingMode": "basic"}}},
                    "initializationOptions": {"foo": "bar"},
                }
            }
        }
    )
    server = config.servers["python"]

    assert server.disabled is True
    assert server.env == {"PYRIGHT_PYTHON_FORCE_VERSION": "3"}
    assert server.settings == {"python": {"analysis": {"typeCheckingMode": "basic"}}}
    assert server.initialization_options == {"foo": "bar"}
