"""Authentication bootstrap for tg_crab worker."""

from __future__ import annotations

import asyncio
import errno
import json
import logging
import queue
import threading
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse, urlunparse

import httpx

logger = logging.getLogger("cli.worker.auth")

DEFAULT_AUTH_CALLBACK_URL = "http://127.0.0.1:8765/callback"
AUTH_CALLBACK_TIMEOUT_SECONDS = 180.0


@dataclass(slots=True)
class WorkerAuthConfig:
    """Authentication-related worker configuration."""

    enable_auth: bool = False
    auth_host: str | None = None
    server_host: str | None = None
    client_id: str | None = None
    redirect_url: str | None = None


@dataclass(slots=True)
class _CallbackPayload:
    """Result returned by the local callback server."""

    code: str | None = None
    error: str | None = None


class _AuthCallbackHTTPServer(ThreadingHTTPServer):
    """One-shot localhost callback server for auth code capture."""

    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, server_address: tuple[str, int], expected_path: str) -> None:
        super().__init__(server_address, _AuthCallbackRequestHandler)
        self.expected_path = expected_path or "/"
        self.result_queue: queue.Queue[_CallbackPayload] = queue.Queue(maxsize=1)


class _AuthCallbackRequestHandler(BaseHTTPRequestHandler):
    """Capture the callback request and return a small HTML page."""

    server: _AuthCallbackHTTPServer

    def do_GET(self) -> None:  # noqa: N802
        parsed_request = urlparse(self.path)
        if parsed_request.path != self.server.expected_path:
            self.send_error(404, "Not Found")
            return

        params = parse_qs(parsed_request.query)
        error_message = _first_query_value(params, "error_description") or _first_query_value(
            params, "error"
        )
        if error_message:
            self._publish_result(
                _CallbackPayload(error=f"Auth callback returned error: {error_message}")
            )
            self._write_html(
                status_code=400,
                body="<html><body><h1>认证失败</h1><p>请返回命令行查看错误详情。</p></body></html>",
            )
            return

        code = _first_query_value(params, "code")
        if not code:
            self._publish_result(
                _CallbackPayload(error="Auth callback missing code query parameter")
            )
            self._write_html(
                status_code=400,
                body="<html><body><h1>认证失败</h1><p>回调中缺少 code，请返回命令行重试。</p></body></html>",
            )
            return

        self._publish_result(_CallbackPayload(code=code))
        self._write_html(
            status_code=200,
            body="<html><body><h1>认证完成</h1><p>可以关闭此页面并返回终端。</p></body></html>",
        )

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        logger.info(f"Local auth callback server: {format % args}")

    def _publish_result(self, payload: _CallbackPayload) -> None:
        try:
            self.server.result_queue.put_nowait(payload)
        except queue.Full:
            logger.debug("Local auth callback result already captured, ignoring duplicate request")

    def _write_html(self, status_code: int, body: str) -> None:
        encoded_body = body.encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded_body)))
        self.end_headers()
        self.wfile.write(encoded_body)


@dataclass(slots=True)
class _LocalAuthCallbackServer:
    """Running local callback server and its runtime metadata."""

    server: _AuthCallbackHTTPServer
    thread: threading.Thread
    callback_url: str

    def wait_for_payload(self, timeout_seconds: float) -> _CallbackPayload:
        """Wait for the callback payload from the browser redirect."""
        try:
            return self.server.result_queue.get(timeout=timeout_seconds)
        except queue.Empty as exc:
            timeout_text = int(timeout_seconds)
            raise TimeoutError(
                f"Timed out waiting for auth callback after {timeout_text} seconds"
            ) from exc

    def close(self) -> None:
        """Stop the local callback server and join its background thread."""
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=1.0)


def load_auth_config(base_dir: Path | str | None = None) -> WorkerAuthConfig:
    """Load worker auth config from tg_crab_worker.json."""
    resolved_base_dir = Path(base_dir or Path.cwd())
    config_path = resolved_base_dir / "tg_crab_worker.json"
    if not config_path.exists():
        logger.info(f"Worker auth config file not found, using defaults: {config_path}")
        return WorkerAuthConfig()

    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid worker auth config JSON: {config_path}") from exc

    return WorkerAuthConfig(
        enable_auth=bool(payload.get("enable_auth", False)),
        auth_host=_normalize_optional_string(payload.get("auth_host")),
        server_host=_normalize_optional_string(payload.get("server_host")),
        client_id=_normalize_optional_string(payload.get("client_id")),
        redirect_url=_normalize_optional_string(payload.get("redirect_url")),
    )


async def authenticate_if_enabled(
    config: WorkerAuthConfig,
    base_dir: Path | str | None = None,
    client: httpx.AsyncClient | None = None,
) -> str | None:
    """Authenticate and persist the Authorization token when enabled."""
    if not config.enable_auth:
        logger.info("Worker auth disabled, skipping auth bootstrap")
        return None

    _validate_enabled_config(config)

    resolved_base_dir = Path(base_dir or Path.cwd())
    owns_client = client is None
    http_client = client or httpx.AsyncClient(
        timeout=httpx.Timeout(connect=5.0, read=10.0, write=10.0, pool=5.0)
    )

    try:
        code = await _fetch_code(http_client=http_client, config=config)
        authorization = await _fetch_authorization(
            http_client=http_client,
            server_host=config.server_host,
            code=code,
        )
    finally:
        if owns_client:
            await http_client.aclose()

    _persist_authorization(base_dir=resolved_base_dir, authorization=authorization)
    logger.info("Worker auth bootstrap succeeded")
    return authorization


async def _fetch_code(http_client: httpx.AsyncClient, config: WorkerAuthConfig) -> str:
    """Open the browser auth flow and wait for the localhost callback code."""
    callback_server = _start_local_callback_server(config.redirect_url)
    try:
        response = await http_client.get(
            config.auth_host,
            params={
                "client_id": config.client_id,
                "response_type": "code",
                "redirect_url": callback_server.callback_url,
            },
            follow_redirects=False,
        )
        if response.status_code != 302:
            raise ValueError(f"Expected 302 from auth_host, got {response.status_code}")

        location = response.headers.get("Location")
        if not location:
            raise ValueError("Auth response missing Location header")

        _open_browser(location)
        payload = await asyncio.to_thread(
            callback_server.wait_for_payload,
            AUTH_CALLBACK_TIMEOUT_SECONDS,
        )
    finally:
        callback_server.close()

    if payload.error:
        raise ValueError(payload.error)
    if not payload.code:
        raise ValueError("Auth callback missing code query parameter")
    return payload.code


async def _fetch_authorization(
    http_client: httpx.AsyncClient,
    server_host: str | None,
    code: str,
) -> str:
    """Fetch Authorization header from server login endpoint."""
    login_url = f"{server_host.rstrip('/')}/login"
    response = await http_client.get(login_url, params={"code": code})
    response.raise_for_status()

    authorization = response.headers.get("Authorization")
    if not authorization:
        raise ValueError("Login response missing Authorization header")
    return authorization


def _persist_authorization(base_dir: Path, authorization: str) -> None:
    """Persist the authorization token to .tg_crab/token.json."""
    token_path = base_dir / ".tg_crab" / "token.json"
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(
        json.dumps({"authorization": authorization}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _validate_enabled_config(config: WorkerAuthConfig) -> None:
    """Validate required config fields for enabled auth."""
    required_fields = {
        "auth_host": config.auth_host,
        "server_host": config.server_host,
        "client_id": config.client_id,
    }
    missing_fields = [name for name, value in required_fields.items() if not value]
    if missing_fields:
        missing_fields_text = ", ".join(missing_fields)
        raise ValueError(f"Missing required auth config fields: {missing_fields_text}")


def _normalize_optional_string(value: object) -> str | None:
    """Normalize optional string config values."""
    if value is None:
        return None
    normalized_value = str(value).strip()
    if not normalized_value:
        return None
    return normalized_value


def _open_browser(url: str) -> bool:
    """Open the system browser for the interactive auth step."""
    try:
        opened = bool(webbrowser.open(url, new=2, autoraise=True))
    except Exception as exc:  # pragma: no cover - depends on runtime desktop environment
        logger.warning(
            f"Failed to open browser automatically: {exc}. Open this URL manually: {url}"
        )
        return False

    if not opened:
        logger.warning(f"Browser did not open automatically. Open this URL manually: {url}")
        return False

    logger.info(f"Opened browser for auth URL: {url}")
    return True


def _start_local_callback_server(preferred_callback_url: str | None) -> _LocalAuthCallbackServer:
    """Start a localhost callback server, falling back to a random port if needed."""
    parsed_callback_url = _parse_local_callback_url(
        preferred_callback_url or DEFAULT_AUTH_CALLBACK_URL
    )

    try:
        server, callback_url = _create_callback_server(
            parsed_callback_url, parsed_callback_url.port
        )
    except OSError as exc:
        if exc.errno != errno.EADDRINUSE:
            raise
        logger.warning(
            f"Preferred auth callback port {parsed_callback_url.port} is busy, falling back to a random port"
        )
        server, callback_url = _create_callback_server(parsed_callback_url, 0)

    thread = threading.Thread(
        target=server.serve_forever,
        name="worker-auth-callback",
        daemon=True,
    )
    thread.start()
    logger.info(f"Started local auth callback server at {callback_url}")
    return _LocalAuthCallbackServer(server=server, thread=thread, callback_url=callback_url)


def _parse_local_callback_url(callback_url: str):
    """Validate and normalize the configured localhost callback URL."""
    parsed_callback_url = urlparse(callback_url)
    if parsed_callback_url.scheme != "http":
        raise ValueError("redirect_url must use http for the local callback server")

    if parsed_callback_url.hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError("redirect_url must point to localhost or 127.0.0.1")

    if parsed_callback_url.port is None:
        raise ValueError("redirect_url must include an explicit local callback port")

    normalized_path = parsed_callback_url.path or "/"
    return parsed_callback_url._replace(path=normalized_path, fragment="")


def _create_callback_server(parsed_callback_url, port: int) -> tuple[_AuthCallbackHTTPServer, str]:
    """Bind the local callback server and return the actual callback URL."""
    host = parsed_callback_url.hostname or "127.0.0.1"
    server = _AuthCallbackHTTPServer((host, port), parsed_callback_url.path)
    actual_port = int(server.server_address[1])
    callback_url = urlunparse(
        (
            parsed_callback_url.scheme,
            f"{host}:{actual_port}",
            parsed_callback_url.path,
            "",
            parsed_callback_url.query,
            "",
        )
    )
    return server, callback_url


def _first_query_value(params: dict[str, list[str]], key: str) -> str | None:
    """Return the first non-empty value for one query parameter."""
    values = params.get(key)
    if not values:
        return None

    first_value = values[0].strip()
    if not first_value:
        return None
    return first_value
