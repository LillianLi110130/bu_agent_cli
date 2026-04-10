from __future__ import annotations

import os
from pathlib import Path

import claude_code
from agent_core.runtime_paths import ensure_cli_runtime_state, load_runtime_env
from cli.worker.auth import load_auth_config


def test_load_runtime_env_prefers_workspace_file_over_home_without_overriding_shell(
    tmp_path: Path,
    monkeypatch,
) -> None:
    package_dir = tmp_path / "package"
    home_dir = tmp_path / "home"
    workspace_dir = tmp_path / "workspace"
    packaged_env = package_dir / ".env"
    home_env = home_dir / ".tg_agent" / ".env"
    workspace_env = workspace_dir / ".env"

    package_dir.mkdir()
    home_env.parent.mkdir(parents=True)
    workspace_dir.mkdir(parents=True)
    packaged_env.write_text(
        "TG_AGENT_TEST_PACKAGED_ONLY=1\nTG_AGENT_TEST_SHARED=packaged\n",
        encoding="utf-8",
    )
    home_env.write_text(
        "TG_AGENT_TEST_SHARED=home\nTG_AGENT_TEST_HOME_ONLY=1\n",
        encoding="utf-8",
    )
    workspace_env.write_text(
        "TG_AGENT_TEST_SHARED=workspace\nTG_AGENT_TEST_WORKSPACE_ONLY=1\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HOME", str(home_dir))
    monkeypatch.chdir(workspace_dir)
    monkeypatch.setattr("agent_core.runtime_paths.application_root", lambda: package_dir)
    monkeypatch.setenv("TG_AGENT_TEST_SHELL_ONLY", "shell")
    monkeypatch.setenv("TG_AGENT_TEST_SHARED", "shell")

    load_runtime_env()

    assert os.environ["TG_AGENT_TEST_SHARED"] == "shell"
    assert os.environ["TG_AGENT_TEST_PACKAGED_ONLY"] == "1"
    assert os.environ["TG_AGENT_TEST_HOME_ONLY"] == "1"
    assert os.environ["TG_AGENT_TEST_WORKSPACE_ONLY"] == "1"
    assert os.environ["TG_AGENT_TEST_SHELL_ONLY"] == "shell"


def test_load_auth_config_prefers_user_home_config_before_packaged(
    tmp_path: Path,
    monkeypatch,
) -> None:
    home_dir = tmp_path / "home"
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    (home_dir / ".tg_agent").mkdir(parents=True)
    (home_dir / ".tg_agent" / "tg_crab_worker.json").write_text(
        (
            "{\n"
            '  "enable_auth": false,\n'
            '  "gateway_base_url": "http://127.0.0.1:9988"\n'
            "}\n"
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("HOME", str(home_dir))

    config = load_auth_config(base_dir=workspace_dir)

    assert config.gateway_base_url == "http://127.0.0.1:9988"


def test_build_worker_process_command_uses_module_mode_for_source(monkeypatch) -> None:
    monkeypatch.setattr(claude_code, "is_frozen_app", lambda: False)

    command = claude_code._build_worker_process_command(
        worker_id="worker-1",
        gateway_base_url="http://127.0.0.1:8765",
        config_dir=Path("D:/config"),
        root_dir=Path("D:/workspace"),
        model="gpt-4o",
    )

    assert command[:3] == [claude_code.sys.executable, "-m", "cli.worker.main"]
    assert "--model" in command


def test_build_worker_process_command_uses_internal_flag_for_frozen(monkeypatch) -> None:
    monkeypatch.setattr(claude_code, "is_frozen_app", lambda: True)

    command = claude_code._build_worker_process_command(
        worker_id="worker-1",
        gateway_base_url="http://127.0.0.1:8765",
        config_dir=Path("D:/config"),
        root_dir=Path("D:/workspace"),
    )

    assert command[:2] == [claude_code.sys.executable, claude_code._INTERNAL_WORKER_FLAG]
    assert "-m" not in command


def test_cli_runtime_env_is_loaded_explicitly(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(claude_code, "ensure_cli_runtime_state", lambda: calls.append("bootstrapped"))
    monkeypatch.setattr(claude_code, "load_runtime_env", lambda: calls.append("loaded"))

    claude_code._load_cli_runtime_env()

    assert calls == ["bootstrapped", "loaded"]


def test_ensure_cli_runtime_state_creates_default_home_files(
    tmp_path: Path,
    monkeypatch,
) -> None:
    home_dir = tmp_path / "home"
    package_root = tmp_path / "package"
    package_root.mkdir()
    (package_root / ".env").write_text(
        "OPENAI_API_KEY=packaged-key\nCUSTOM_MODEL_KEY=custom-secret\nLLM_MODEL=GLM-5.1\n",
        encoding="utf-8",
    )
    (package_root / "tg_crab_worker.json").write_text(
        '{"enable_auth": false, "gateway_base_url": "http://127.0.0.1:8765"}\n',
        encoding="utf-8",
    )

    monkeypatch.setenv("HOME", str(home_dir))
    monkeypatch.delenv("TG_AGENT_HOME", raising=False)
    monkeypatch.setattr("agent_core.runtime_paths.application_root", lambda: package_root)
    monkeypatch.setattr(
        "agent_core.runtime_paths._load_model_preset_api_key_env_names",
        lambda: ["OPENAI_API_KEY", "CUSTOM_MODEL_KEY", "ANOTHER_MODEL_KEY"],
    )

    resolved_home = ensure_cli_runtime_state()

    env_file = resolved_home / ".env"
    worker_config = resolved_home / "tg_crab_worker.json"

    assert resolved_home == (home_dir / ".tg_agent").resolve()
    assert env_file.exists()
    env_text = env_file.read_text(encoding="utf-8")
    assert "OPENAI_API_KEY=packaged-key" in env_text
    assert "CUSTOM_MODEL_KEY=custom-secret" in env_text
    assert "LLM_MODEL=GLM-5.1" in env_text
    assert "ANOTHER_MODEL_KEY=" in env_text
    assert worker_config.exists()
    assert '"enable_auth": false' in worker_config.read_text(encoding="utf-8")


def test_ensure_cli_runtime_state_overwrites_existing_user_env_with_packaged_defaults(
    tmp_path: Path,
    monkeypatch,
) -> None:
    home_dir = tmp_path / "home"
    package_root = tmp_path / "package"
    package_root.mkdir()
    (package_root / ".env").write_text(
        "CUSTOM_MODEL_KEY=packaged-secret\nANOTHER_MODEL_KEY=from-package\n",
        encoding="utf-8",
    )

    user_env = home_dir / ".tg_agent" / ".env"
    user_env.parent.mkdir(parents=True, exist_ok=True)
    user_env.write_text(
        "OPENAI_API_KEY=user-key\nCUSTOM_MODEL_KEY=user-secret\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HOME", str(home_dir))
    monkeypatch.delenv("TG_AGENT_HOME", raising=False)
    monkeypatch.setattr("agent_core.runtime_paths.application_root", lambda: package_root)
    monkeypatch.setattr(
        "agent_core.runtime_paths._load_model_preset_api_key_env_names",
        lambda: ["OPENAI_API_KEY", "CUSTOM_MODEL_KEY", "ANOTHER_MODEL_KEY"],
    )

    ensure_cli_runtime_state()

    env_text = user_env.read_text(encoding="utf-8")
    assert "OPENAI_API_KEY=" in env_text
    assert "OPENAI_API_KEY=user-key" not in env_text
    assert "CUSTOM_MODEL_KEY=packaged-secret" in env_text
    assert "CUSTOM_MODEL_KEY=user-secret" not in env_text
    assert "ANOTHER_MODEL_KEY=from-package" in env_text
