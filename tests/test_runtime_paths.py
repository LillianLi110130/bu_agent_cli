from __future__ import annotations

import os
from pathlib import Path

import claude_code
from agent_core.runtime_paths import load_runtime_env
from cli.worker.auth import load_auth_config


def test_load_runtime_env_prefers_workspace_file_over_home_without_overriding_shell(
    tmp_path: Path,
    monkeypatch,
) -> None:
    home_dir = tmp_path / "home"
    workspace_dir = tmp_path / "workspace"
    home_env = home_dir / ".tg_agent" / ".env"
    workspace_env = workspace_dir / ".env"

    home_env.parent.mkdir(parents=True)
    workspace_dir.mkdir(parents=True)
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
    monkeypatch.setenv("TG_AGENT_TEST_SHELL_ONLY", "shell")
    monkeypatch.setenv("TG_AGENT_TEST_SHARED", "shell")

    load_runtime_env()

    assert os.environ["TG_AGENT_TEST_SHARED"] == "shell"
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
