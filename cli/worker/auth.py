"""Authentication bootstrap helpers for worker startup."""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

from agent_core.runtime_paths import application_root, tg_agent_home

logger = logging.getLogger("cli.worker.auth")

DEFAULT_AUTH_CALLBACK_URL = "http://127.0.0.1:8088/callback"
AUTH_CALLBACK_TIMEOUT_SECONDS = 180.0
_PACKAGE_CONFIG_FILE_NAME = "tg_crab_worker.json"


@dataclass(slots=True)
class WorkerAuthConfig:
    """Authentication-related worker configuration."""

    enable_auth: bool = False
    auth_host: str | None = None
    server_host: str | None = None
    gateway_base_url: str | None = None
    client_id: str | None = None
    redirect_url: str | None = None


@dataclass(slots=True)
class AuthBootstrapResult:
    """Authentication material required by the worker runtime."""

    authorization: str
    user_id: str


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
                body=(
                    "<html><body><h1>Login failed</h1>"
                    "<p>The authorization callback reported an error.</p></body></html>"
                ),
            )
            return

        code = _first_query_value(params, "code")
        if not code:
            self._publish_result(
                _CallbackPayload(error="Auth callback missing code query parameter")
            )
            self._write_html(
                status_code=400,
                body=(
                    "<html><body><h1>Login failed</h1>"
                    "<p>The authorization callback did not include a code.</p></body></html>"
                ),
            )
            return

        self._publish_result(_CallbackPayload(code=code))
        self._write_html(
            status_code=200,
            body=(
                "<html><body><h1>Login complete</h1>"
                "<p>You can close this page and return to the terminal.</p></body></html>"
            ),
        )

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        logger.info(f"Local auth callback server: {format % args}")

    def _publish_result(self, payload: _CallbackPayload) -> None:
        try:
            self.server.result_queue.put_nowait(payload)
        except queue.Full:
            logger.debug("Local auth callback result already captured, ignoring duplicate")

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
    config_path = _resolve_auth_config_path(base_dir)
    if config_path is None:
        logger.info(
            "Worker auth config file not found, using defaults. "
            f"Searched: {_format_auth_config_search_paths(base_dir)}"
        )
        return WorkerAuthConfig()

    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid worker auth config JSON: {config_path}") from exc

    return WorkerAuthConfig(
        enable_auth=bool(payload.get("enable_auth", False)),
        auth_host=_normalize_optional_string(payload.get("auth_host")),
        server_host=_normalize_optional_string(payload.get("server_host")),
        gateway_base_url=_normalize_optional_string(payload.get("gateway_base_url")),
        client_id=_normalize_optional_string(payload.get("client_id")),
        redirect_url=_normalize_optional_string(payload.get("redirect_url")),
    )


def _resolve_auth_config_path(base_dir: Path | str | None = None) -> Path | None:
    """Return the first existing auth config path."""
    for config_path in _iter_auth_config_paths(base_dir):
        if config_path.exists():
            return config_path
    return None


def _iter_auth_config_paths(base_dir: Path | str | None = None) -> list[Path]:
    """Return auth config search paths in priority order."""
    resolved_base_dir = Path(base_dir or Path.cwd()).resolve()
    search_paths = [resolved_base_dir / _PACKAGE_CONFIG_FILE_NAME]

    home_config_path = tg_agent_home() / _PACKAGE_CONFIG_FILE_NAME
    if home_config_path not in search_paths:
        search_paths.append(home_config_path)

    package_config_path = _get_package_config_dir() / _PACKAGE_CONFIG_FILE_NAME
    if package_config_path not in search_paths:
        search_paths.append(package_config_path)

    return search_paths


def _format_auth_config_search_paths(base_dir: Path | str | None = None) -> str:
    return ", ".join(str(path) for path in _iter_auth_config_paths(base_dir))


def _get_package_config_dir() -> Path:
    """Return the installed tg-agent package root directory."""
    return application_root()


async def authenticate_startup(
    config: WorkerAuthConfig,
    base_dir: Path | str | None = None,
    client: httpx.AsyncClient | None = None,
) -> AuthBootstrapResult:
    """Perform interactive auth and persist the resulting authorization state."""
    _validate_enabled_config(config)

    resolved_base_dir = Path(base_dir or Path.cwd())
    owns_client = client is None
    http_client = client or httpx.AsyncClient(
        timeout=httpx.Timeout(connect=5.0, read=10.0, write=10.0, pool=5.0),
        trust_env=_should_trust_env(config),
    )

    try:
        code = await _fetch_code(config=config)
        auth_result = await _fetch_authorization(
            http_client=http_client,
            server_host=config.server_host,
            code=code,
        )
    finally:
        if owns_client:
            await http_client.aclose()

    _persist_auth_result(base_dir=resolved_base_dir, auth_result=auth_result)
    logger.info("Worker auth bootstrap succeeded")
    return auth_result


async def authenticate_if_enabled(
    config: WorkerAuthConfig,
    base_dir: Path | str | None = None,
    client: httpx.AsyncClient | None = None,
) -> AuthBootstrapResult | None:
    """Backward-compatible wrapper that only authenticates when enabled."""
    if not config.enable_auth:
        logger.info("Worker auth disabled, skipping auth bootstrap")
        return None
    return await authenticate_startup(config=config, base_dir=base_dir, client=client)


def load_persisted_auth_result(base_dir: Path | str | None = None) -> AuthBootstrapResult | None:
    """Load the persisted authorization state from disk."""
    token_path = _resolve_auth_result_path(base_dir)
    if token_path is None:
        return None

    payload = json.loads(token_path.read_text(encoding="utf-8"))
    authorization = _normalize_optional_string(payload.get("authorization"))
    user_id = _normalize_optional_string(payload.get("user_id"))
    if not authorization or not user_id:
        raise ValueError(f"Invalid persisted auth result: {token_path}")
    return AuthBootstrapResult(authorization=authorization, user_id=user_id)


async def _fetch_code(config: WorkerAuthConfig) -> str:
    """Open the browser auth flow and wait for the localhost callback code."""
    callback_server = _start_local_callback_server(config.redirect_url)
    try:
        auth_url = _build_auth_url(config=config, redirect_uri=callback_server.callback_url)
        _open_browser(auth_url)
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
) -> AuthBootstrapResult:
    """Fetch Authorization header and worker user id from the login endpoint."""
    if not server_host:
        raise ValueError("server_host is required")

    login_url = f"{server_host.rstrip('/')}/user-privilege/login"
    response = await http_client.get(login_url, params={"code": code})
    response.raise_for_status()

    authorization = _normalize_optional_string(response.headers.get("Authorization"))
    if not authorization:
        raise ValueError("Login response missing Authorization header")

    payload = response.json()
    return_code = _normalize_optional_string(payload.get("returnCode"))
    if return_code != "SUC0000":
        error_message = _normalize_optional_string(payload.get("errorMsg")) or "unknown login error"
        raise ValueError(f"Login response returned failure code {return_code!r}: {error_message}")

    body = payload.get("body")
    if not isinstance(body, dict):
        raise ValueError("Login response missing body object")

    user_id = _normalize_optional_string(body.get("userNo"))
    if not user_id:
        raise ValueError("Login response missing body.userNo in JSON body")

    return AuthBootstrapResult(authorization=authorization, user_id=user_id)


def _persist_auth_result(base_dir: Path, auth_result: AuthBootstrapResult) -> None:
    """Persist the authorization state to .tg_agent/token.json."""
    token_path = base_dir / ".tg_agent" / "token.json"
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(
        json.dumps(
            {
                "authorization": auth_result.authorization,
                "user_id": auth_result.user_id,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _resolve_auth_result_path(base_dir: Path | str | None) -> Path | None:
    """Return the preferred auth result path, with legacy fallback support."""
    resolved_base_dir = Path(base_dir or Path.cwd())
    preferred_path = resolved_base_dir / ".tg_agent" / "token.json"
    if preferred_path.exists():
        return preferred_path

    legacy_path = resolved_base_dir / ".tg_crab" / "token.json"
    if legacy_path.exists():
        return legacy_path
    return None


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


def _build_auth_url(config: WorkerAuthConfig, redirect_uri: str) -> str:
    """Build the SSO authorization URL."""
    if not config.auth_host:
        raise ValueError("auth_host is required")
    params = {
        "client_id": config.client_id,
        "response_type": "code",
        "redirect_uri": redirect_uri,
    }
    return f"{config.auth_host}?{urlencode(params)}"


def _open_browser(url: str) -> bool:
    """Open the system browser for the interactive auth step."""
    try:
        opened = bool(webbrowser.open(url, new=2, autoraise=True))
    except Exception as exc:  # pragma: no cover - depends on desktop environment
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
    """Start the fixed localhost callback server on port 8088."""
    parsed_callback_url = _parse_local_callback_url(
        preferred_callback_url or DEFAULT_AUTH_CALLBACK_URL
    )
    server, callback_url = _create_callback_server(parsed_callback_url)
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

    if parsed_callback_url.port != 8088:
        raise ValueError("redirect_url must use local callback port 8088")

    normalized_path = parsed_callback_url.path or "/callback"
    return parsed_callback_url._replace(path=normalized_path, fragment="")


def _create_callback_server(parsed_callback_url) -> tuple[_AuthCallbackHTTPServer, str]:
    """Bind the local callback server and return the actual callback URL."""
    host = parsed_callback_url.hostname or "127.0.0.1"
    server = _AuthCallbackHTTPServer((host, 8088), parsed_callback_url.path)
    callback_url = f"http://{host}:8088{parsed_callback_url.path}"
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


def _should_trust_env(config: WorkerAuthConfig) -> bool:
    """Return False when both auth endpoints target loopback addresses."""
    hosts = [
        (urlparse(config.auth_host or "").hostname or "").strip().lower(),
        (urlparse(config.server_host or "").hostname or "").strip().lower(),
    ]
    meaningful_hosts = [host for host in hosts if host]
    if meaningful_hosts and all(
        host in {"127.0.0.1", "localhost", "::1"} for host in meaningful_hosts
    ):
        return False
    return True
