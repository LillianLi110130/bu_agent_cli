from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import shutil
import uuid

import httpx
import pytest

import claude_code
from cli.worker import auth
from cli.worker.mock_auth_server import create_mock_auth_app


@pytest.fixture
def workspace_root() -> Path:
    root = Path(".pytest_tmp") / f"worker-auth-{uuid.uuid4().hex}"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    try:
        yield root.resolve()
    finally:
        if root.exists():
            shutil.rmtree(root)


@pytest.mark.asyncio
async def test_fetch_authorization_extracts_authorization_and_userno():
    app = create_mock_auth_app()
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        result = await auth._fetch_authorization(  # noqa: SLF001
            http_client=client,
            server_host="http://testserver",
            code="mock-code",
        )

    assert result.authorization == "Bearer mock-token"
    assert result.user_id == "mock-user-123"


@pytest.mark.asyncio
async def test_fetch_authorization_rejects_unsuccessful_return_code():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Authorization": "Bearer mock-token"},
            json={
                "body": {"userNo": "mock-user-123"},
                "errorMsg": "login failed",
                "returnCode": "ERR0001",
            },
        )

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        with pytest.raises(ValueError, match="ERR0001"):
            await auth._fetch_authorization(  # noqa: SLF001
                http_client=client,
                server_host="http://testserver",
                code="mock-code",
            )


def test_persisted_auth_result_round_trip(workspace_root: Path):
    result = auth.AuthBootstrapResult(
        authorization="Bearer abc",
        user_id="user-123",
    )

    auth._persist_auth_result(workspace_root, result)  # noqa: SLF001
    loaded = auth.load_persisted_auth_result(workspace_root)

    assert loaded == result
    token_payload = json.loads(
        (workspace_root / ".tg_agent" / "token.json").read_text(encoding="utf-8")
    )
    assert token_payload["authorization"] == "Bearer abc"
    assert token_payload["user_id"] == "user-123"


def test_load_persisted_auth_result_falls_back_to_legacy_tg_crab_path(workspace_root: Path):
    legacy_path = workspace_root / ".tg_crab" / "token.json"
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.write_text(
        json.dumps(
            {
                "authorization": "Bearer legacy",
                "user_id": "legacy-user",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    loaded = auth.load_persisted_auth_result(workspace_root)

    assert loaded == auth.AuthBootstrapResult(
        authorization="Bearer legacy",
        user_id="legacy-user",
    )


def test_persist_updated_authorization_overwrites_token_and_preserves_user_id(workspace_root: Path):
    auth._persist_auth_result(  # noqa: SLF001
        workspace_root,
        auth.AuthBootstrapResult(
            authorization="Bearer old-token",
            user_id="user-123",
        ),
    )

    updated = auth.persist_updated_authorization(
        base_dir=workspace_root,
        authorization="Bearer new-token",
    )

    assert updated == auth.AuthBootstrapResult(
        authorization="Bearer new-token",
        user_id="user-123",
    )
    token_payload = json.loads(
        (workspace_root / ".tg_agent" / "token.json").read_text(encoding="utf-8")
    )
    assert token_payload["authorization"] == "Bearer new-token"
    assert token_payload["user_id"] == "user-123"


def test_parse_local_callback_url_requires_fixed_port_8088():
    parsed = auth._parse_local_callback_url("http://127.0.0.1:8088/callback")  # noqa: SLF001
    assert parsed.port == 8088

    with pytest.raises(ValueError, match="8088"):
        auth._parse_local_callback_url("http://127.0.0.1:9999/callback")  # noqa: SLF001


def test_load_auth_config_reads_gateway_base_url(workspace_root: Path):
    config_path = workspace_root / "tg_crab_worker.json"
    config_path.write_text(
        json.dumps(
            {
                "enable_auth": False,
                "gateway_base_url": "http://127.0.0.1:9765",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    config = auth.load_auth_config(workspace_root)

    assert config.gateway_base_url == "http://127.0.0.1:9765"


def test_load_auth_config_falls_back_to_package_install_dir(
    workspace_root: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    package_root = workspace_root / "installed-package"
    package_root.mkdir(parents=True, exist_ok=True)
    (package_root / "tg_crab_worker.json").write_text(
        json.dumps(
            {
                "enable_auth": False,
                "gateway_base_url": "http://127.0.0.1:8866",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(auth, "_get_package_config_dir", lambda: package_root)

    config = auth.load_auth_config(workspace_root)

    assert config.gateway_base_url == "http://127.0.0.1:8866"


def test_claude_code_authentication_overrides_worker_id(
    workspace_root: Path,
    monkeypatch,
):
    config_root = workspace_root / "config"
    config_root.mkdir(parents=True, exist_ok=True)
    config = auth.WorkerAuthConfig(enable_auth=True)
    calls: list[Path] = []

    def fake_load_auth_config(base_dir):
        calls.append(Path(base_dir))
        return config

    async def fake_authenticate_startup(*, config, base_dir, client=None):
        assert config.enable_auth is True
        assert Path(base_dir) == config_root
        return auth.AuthBootstrapResult(
            authorization="Bearer mock-token",
            user_id="mock-user-123",
        )

    monkeypatch.setattr(claude_code, "load_auth_config", fake_load_auth_config)
    monkeypatch.setattr(claude_code, "authenticate_startup", fake_authenticate_startup)

    args = argparse.Namespace(
        root_dir=str(workspace_root),
        config_dir=config_root,
        im_worker_id="worker-old",
    )

    asyncio.run(claude_code._authenticate_worker_startup(args))

    assert calls == [config_root]
    assert args.im_worker_id == "mock-user-123"


@pytest.mark.asyncio
async def test_mock_auth_server_authorize_and_login():
    app = create_mock_auth_app()
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        authorize = await client.get(
            "/oauth/authorize",
            params={
                "client_id": "client-123",
                "response_type": "code",
                "redirect_uri": "http://127.0.0.1:8088/callback",
            },
            follow_redirects=False,
        )
        assert authorize.status_code == 302
        assert authorize.headers["location"] == "http://127.0.0.1:8088/callback?code=mock-code"

        login = await client.get("/user-privilege/login", params={"code": "mock-code"})
        assert login.status_code == 200
        assert login.headers["Authorization"] == "Bearer mock-token"
        payload = login.json()
        assert payload["returnCode"] == "SUC0000"
        assert payload["body"]["userNo"] == "mock-user-123"
        assert payload["body"]["ystId"] == "mock-yst-123"
