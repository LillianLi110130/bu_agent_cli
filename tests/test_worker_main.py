from __future__ import annotations

import asyncio
import shutil
import sys
import uuid
from pathlib import Path

from cli.worker import auth
from cli.worker import main as worker_main


def test_async_main_uses_config_dir_for_auth(monkeypatch):
    root = Path(".pytest_tmp") / f"worker-main-{uuid.uuid4().hex}"
    if root.exists():
        shutil.rmtree(root)
    config_dir = root / "config"
    config_dir.mkdir(parents=True)
    workspace_root = root / "workspace"

    calls: list[tuple[str, object]] = []

    class FakeGatewayClient:
        def __init__(self, *, base_url: str, authorization: str | None = None):
            calls.append(("client", base_url, authorization))

    class FakeRunner:
        def __init__(self, *, worker_id, gateway_client, model, root_dir):
            calls.append(("runner", worker_id, model, root_dir, type(gateway_client).__name__))

        async def run_forever(self) -> None:
            calls.append(("run_forever",))

    def fake_load_auth_config(*, base_dir):
        calls.append(("load_auth_config", Path(base_dir)))
        return auth.WorkerAuthConfig(enable_auth=True)

    def fake_load_persisted_auth_result(*, base_dir):
        calls.append(("load_persisted_auth_result", Path(base_dir)))
        return auth.AuthBootstrapResult(
            authorization="Bearer test-token",
            user_id="worker-user",
        )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cli.worker.main",
            "--worker-id",
            "worker-1",
            "--gateway-base-url",
            "http://127.0.0.1:8765",
            "--config-dir",
            str(config_dir),
            "--root-dir",
            str(workspace_root),
        ],
    )
    monkeypatch.setattr(worker_main, "WorkerGatewayClient", FakeGatewayClient)
    monkeypatch.setattr(worker_main, "WorkerRunner", FakeRunner)
    monkeypatch.setattr(worker_main, "load_auth_config", fake_load_auth_config)
    monkeypatch.setattr(worker_main, "load_persisted_auth_result", fake_load_persisted_auth_result)

    try:
        asyncio.run(worker_main.async_main())

        assert calls == [
            ("load_auth_config", config_dir.resolve()),
            ("load_persisted_auth_result", config_dir.resolve()),
            ("client", "http://127.0.0.1:8765", "Bearer test-token"),
            ("runner", "worker-1", None, str(workspace_root), "FakeGatewayClient"),
            ("run_forever",),
        ]
    finally:
        if root.exists():
            shutil.rmtree(root)
