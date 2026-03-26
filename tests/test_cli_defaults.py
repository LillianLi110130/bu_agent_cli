from __future__ import annotations

import sys
import shutil
import uuid
from pathlib import Path

import pytest

import claude_code


def test_parse_args_defaults_enable_local_bridge_and_im(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["claude_code.py"])

    args = claude_code.parse_args()

    assert args.local_bridge is True
    assert args.im_enable is True
    assert args.im_gateway_base_url == "http://127.0.0.1:8765"
    assert args.im_worker_id.startswith("worker-")


def test_parse_args_can_disable_im_and_local_bridge(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["claude_code.py", "--no-im-enable", "--no-local-bridge"],
    )

    args = claude_code.parse_args()

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
            ["claude_code.py", "--root-dir", str(workspace_root)],
        )

        args = claude_code.parse_args()

        assert args.config_dir == startup_dir_resolved
        assert args.im_gateway_base_url == "http://127.0.0.1:9765"
    finally:
        if root.exists():
            shutil.rmtree(root)


def test_top_level_console_outputs_plain_text():
    with claude_code.console.capture() as capture:
        claude_code.console.print("[yellow]再见！[/yellow]")

    assert capture.get() == "再见！\n"


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

    monkeypatch.setattr(claude_code, "WorkerGatewayClient", FakeGatewayClient)

    await claude_code._mark_worker_offline(
        worker_id="worker-1",
        gateway_base_url="http://127.0.0.1:8765",
    )

    assert calls == [("http://127.0.0.1:8765", "worker-1")]
