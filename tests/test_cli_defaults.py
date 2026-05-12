from __future__ import annotations

import sys
import shutil
import uuid
from pathlib import Path

import pytest

import tg_crab_main


def test_parse_args_defaults_enable_local_bridge_and_im(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["tg_crab_main.py"])

    args = tg_crab_main.parse_args()

    assert args.local_bridge is True
    assert args.im_enable is True
    assert args.im_gateway_base_url == "http://127.0.0.1:8765"
    assert args.im_worker_id.startswith("worker-")


def test_parse_args_can_disable_im_and_local_bridge(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["tg_crab_main.py", "--no-im-enable", "--no-local-bridge"],
    )

    args = tg_crab_main.parse_args()

    assert args.im_enable is False
    assert args.local_bridge is False


def test_parse_args_root_dir_does_not_change_startup_config_dir(monkeypatch):
    root = Path(".pytest_tmp") / f"cli-defaults-{uuid.uuid4().hex}"
    if root.exists():
        shutil.rmtree(root)
    startup_dir = root / "startup"
    startup_dir.mkdir(parents=True)
    startup_dir_resolved = startup_dir.resolve()
    (startup_dir / "tg_crab_worker.json").write_text(
        (
            "{\n"
            '  "enable_auth": false,\n'
            '  "gateway_base_url": "http://127.0.0.1:9765"\n'
            "}\n"
        ),
        encoding="utf-8",
    )
    workspace_root = root / "workspace"

    try:
        monkeypatch.chdir(startup_dir)
        monkeypatch.setattr(
            sys,
            "argv",
            ["tg_crab_main.py", "--root-dir", str(workspace_root)],
        )

        args = tg_crab_main.parse_args()

        assert args.config_dir == startup_dir_resolved
        assert args.config_source_dir == startup_dir_resolved
        assert args.im_gateway_base_url == "http://127.0.0.1:9765"
    finally:
        if root.exists():
            shutil.rmtree(root)


def test_parse_args_falls_back_to_packaged_worker_config(monkeypatch):
    root = Path(".pytest_tmp") / f"cli-packaged-config-{uuid.uuid4().hex}"
    if root.exists():
        shutil.rmtree(root)
    startup_dir = root / "startup"
    home_dir = root / "home"
    startup_dir.mkdir(parents=True)
    home_dir.mkdir(parents=True)
    startup_dir_resolved = startup_dir.resolve()

    try:
        monkeypatch.setenv("HOME", str(home_dir.resolve()))
        monkeypatch.chdir(startup_dir)
        monkeypatch.setattr(sys, "argv", ["tg_crab_main.py"])

        args = tg_crab_main.parse_args()
        expected_gateway = tg_crab_main.load_auth_config(base_dir=startup_dir).gateway_base_url

        assert args.config_dir == startup_dir_resolved
        assert args.config_source_dir == Path(tg_crab_main._SCRIPT_DIR).resolve()
        assert args.im_gateway_base_url == expected_gateway
    finally:
        if root.exists():
            shutil.rmtree(root)


def test_parse_args_uses_user_default_workspace_when_root_dir_missing(monkeypatch):
    root = Path(".pytest_tmp") / f"cli-default-workspace-{uuid.uuid4().hex}"
    if root.exists():
        shutil.rmtree(root)
    startup_dir = root / "startup"
    startup_dir.mkdir(parents=True)
    startup_dir_resolved = startup_dir.resolve()
    home_dir = root / "home"
    configured_workspace = root / "configured-workspace"
    configured_workspace.mkdir(parents=True)
    configured_workspace_resolved = configured_workspace.resolve()
    (home_dir / ".tg_agent").mkdir(parents=True)
    (home_dir / ".tg_agent" / "settings.json").write_text(
        '{\n  "default_workspace": "%s"\n}\n' % configured_workspace_resolved.as_posix(),
        encoding="utf-8",
    )

    try:
        monkeypatch.setenv("HOME", str(home_dir.resolve()))
        monkeypatch.chdir(startup_dir)
        monkeypatch.setattr(sys, "argv", ["tg_crab_main.py"])

        args = tg_crab_main.parse_args()

        assert args.root_dir == configured_workspace_resolved
        assert args.config_dir == startup_dir_resolved
    finally:
        if root.exists():
            shutil.rmtree(root)


def test_parse_args_explicit_root_dir_overrides_user_default_workspace(monkeypatch):
    root = Path(".pytest_tmp") / f"cli-default-workspace-override-{uuid.uuid4().hex}"
    if root.exists():
        shutil.rmtree(root)
    startup_dir = root / "startup"
    startup_dir.mkdir(parents=True)
    startup_dir_resolved = startup_dir.resolve()
    home_dir = root / "home"
    configured_workspace = root / "configured-workspace"
    explicit_workspace = root / "explicit-workspace"
    configured_workspace.mkdir(parents=True)
    explicit_workspace.mkdir(parents=True)
    configured_workspace_resolved = configured_workspace.resolve()
    explicit_workspace_resolved = explicit_workspace.resolve()
    (home_dir / ".tg_agent").mkdir(parents=True)
    (home_dir / ".tg_agent" / "settings.json").write_text(
        '{\n  "default_workspace": "%s"\n}\n' % configured_workspace_resolved.as_posix(),
        encoding="utf-8",
    )

    try:
        monkeypatch.setenv("HOME", str(home_dir.resolve()))
        monkeypatch.chdir(startup_dir)
        monkeypatch.setattr(
            sys,
            "argv",
            ["tg_crab_main.py", "--root-dir", str(explicit_workspace_resolved)],
        )

        args = tg_crab_main.parse_args()

        assert args.root_dir == explicit_workspace_resolved
        assert args.config_dir == startup_dir_resolved
    finally:
        if root.exists():
            shutil.rmtree(root)


def test_top_level_console_outputs_plain_text():
    with tg_crab_main.console.capture() as capture:
        tg_crab_main.console.print("[yellow]再见！[/yellow]")

    assert capture.get() == "再见！\n"


def test_build_system_prompt_delegates_to_agent_factory(monkeypatch, tmp_path):
    class FakeSkillRegistry:
        def get_all(self):
            return []

    class FakeAgentRegistry:
        def list_callable_agents(self):
            return []

        def get_config(self, _name):
            return None

    captured: dict[str, object] = {}

    def fake_build_system_prompt(working_dir, *, skill_registry=None, agent_registry=None):
        captured["working_dir"] = working_dir
        captured["skill_registry"] = skill_registry
        captured["agent_registry"] = agent_registry
        return "shared prompt"

    monkeypatch.setattr(claude_code, "build_shared_system_prompt", fake_build_system_prompt)

    skill_registry = FakeSkillRegistry()
    agent_registry = FakeAgentRegistry()
    prompt = claude_code._build_system_prompt(
        tmp_path,
        skill_registry=skill_registry,
        agent_registry=agent_registry,
    )

    assert prompt == "shared prompt"
    assert captured["working_dir"] == tmp_path
    assert captured["skill_registry"] is skill_registry
    assert captured["agent_registry"] is agent_registry


@pytest.mark.asyncio
async def test_parent_process_marks_worker_offline(monkeypatch):
    calls: list[tuple[str, str]] = []

    class FakeGatewayClient:
        def __init__(self, *, base_url: str, client=None, authorization=None):
            self.base_url = base_url

        async def offline(self, *, worker_id: str) -> bool:
            calls.append((self.base_url, worker_id))
            return True

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr(tg_crab_main, "WorkerGatewayClient", FakeGatewayClient)

    await tg_crab_main._mark_worker_offline(
        worker_id="worker-1",
        gateway_base_url="http://127.0.0.1:8765",
    )

    assert calls == [("http://127.0.0.1:8765", "worker-1")]
