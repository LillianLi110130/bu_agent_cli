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

_LOGIN_SUCCESS_HTML = """\
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>登录成功</title>
<style>
  *{margin:0;padding:0;box-sizing:border-box}
  body{
    min-height:100vh;display:flex;align-items:center;justify-content:center;
    background:linear-gradient(135deg,#0f172a 0%,#1e293b 50%,#0f172a 100%);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
    color:#f8fafc;overflow:hidden;
  }
  .card{
    text-align:center;padding:48px 56px;border-radius:24px;
    background:rgba(255,255,255,.06);backdrop-filter:blur(20px);
    border:1px solid rgba(255,255,255,.1);
    box-shadow:0 25px 60px rgba(0,0,0,.4);
    animation:fadeUp .6s ease-out both;
    max-width:420px;
  }
  @keyframes fadeUp{
    from{opacity:0;transform:translateY(24px) scale(.96)}
    to{opacity:1;transform:translateY(0) scale(1)}
  }
  .icon{
    width:80px;height:80px;margin:0 auto 28px;
    border-radius:50%;
    background:linear-gradient(135deg,#22c55e,#16a34a);
    display:flex;align-items:center;justify-content:center;
    box-shadow:0 0 0 8px rgba(34,197,94,.15),0 8px 32px rgba(34,197,94,.25);
    animation:popIn .5s .2s ease-out both;
  }
  @keyframes popIn{
    from{transform:scale(0)}
    60%{transform:scale(1.15)}
    to{transform:scale(1)}
  }
  .icon svg{width:40px;height:40px;stroke:#fff;stroke-width:3;fill:none;stroke-linecap:round;stroke-linejoin=round}
  h1{font-size:26px;font-weight:700;margin-bottom:12px;letter-spacing:-.02em}
  p{font-size:15px;color:#94a3b8;line-height:1.6;margin-bottom:32px}
  .btn{
    display:inline-block;padding:12px 32px;border-radius:12px;
    background:linear-gradient(135deg,#3b82f6,#2563eb);
    color:#fff;font-size:14px;font-weight:600;
    border:none;cursor:pointer;
    box-shadow:0 4px 16px rgba(37,99,235,.35);
    transition:transform .15s,box-shadow .15s;
  }
  .btn:hover{transform:translateY(-1px);box-shadow:0 6px 24px rgba(37,99,235,.45)}
  .btn:active{transform:translateY(0)}
  .hint{margin-top:20px;font-size:12px;color:#475569}
  .particles{position:fixed;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:-1}
  .particle{
    position:absolute;border-radius:50%;
    background:rgba(34,197,94,.25);
    animation:float linear infinite;
  }
  @keyframes float{
    0%{transform:translateY(100vh) scale(0);opacity:0}
    10%{opacity:1}
    90%{opacity:1}
    100%{transform:translateY(-10vh) scale(1);opacity:0}
  }
</style>
</head>
<body>
<div class="particles" id="particles"></div>
<div class="card">
  <div class="icon">
    <svg viewBox="0 0 24 24"><polyline points="20 6 9 17 4 12"/></svg>
  </div>
  <h1>🦀 小螃蟹已登录成功</h1>
  <p>身份验证已完成。<br>您可以关闭此页面并返回终端。</p>
  <button class="btn" onclick="window.close()">关闭窗口</button>
  <div class="hint">或手动关闭此标签页</div>
</div>
<script>
(function(){
  var c=document.getElementById('particles');
  for(var i=0;i<12;i++){
    var p=document.createElement('div');
    p.className='particle';
    var s=Math.random()*6+3;
    p.style.cssText='width:'+s+'px;height:'+s+'px;left:'+Math.random()*100+'%;animation-duration:'+(Math.random()*6+4)+'s;animation-delay:'+(Math.random()*4)+'s';
    c.appendChild(p);
  }
})();
</script>
</body>
</html>
"""

_LOGIN_ERROR_HTML = """\
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>登录失败</title>
<style>
  *{margin:0;padding:0;box-sizing:border-box}
  body{
    min-height:100vh;display:flex;align-items:center;justify-content:center;
    background:linear-gradient(135deg,#1c1917 0%,#292524 50%,#1c1917 100%);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
    color:#f8fafc;overflow:hidden;
  }
  .card{
    text-align:center;padding:48px 56px;border-radius:24px;
    background:rgba(255,255,255,.06);backdrop-filter:blur(20px);
    border:1px solid rgba(255,255,255,.1);
    box-shadow:0 25px 60px rgba(0,0,0,.4);
    animation:fadeUp .6s ease-out both;
    max-width:460px;
  }
  @keyframes fadeUp{
    from{opacity:0;transform:translateY(24px) scale(.96)}
    to{opacity:1;transform:translateY(0) scale(1)}
  }
  .icon{
    width:80px;height:80px;margin:0 auto 28px;
    border-radius:50%;
    background:linear-gradient(135deg,#ef4444,#dc2626);
    display:flex;align-items:center;justify-content:center;
    box-shadow:0 0 0 8px rgba(239,68,68,.15),0 8px 32px rgba(239,68,68,.25);
    animation:popIn .5s .2s ease-out both;
  }
  @keyframes popIn{
    from{transform:scale(0)}
    60%{transform:scale(1.15)}
    to{transform:scale(1)}
  }
  .icon svg{width:40px;height:40px;stroke:#fff;stroke-width:3;fill:none;stroke-linecap:round;stroke-linejoin=round}
  h1{font-size:26px;font-weight:700;margin-bottom:12px;letter-spacing:-.02em}
  p{font-size:15px;color:#94a3b8;line-height:1.6;margin-bottom:8px}
  .error-msg{
    margin:16px 0 28px;padding:14px 20px;border-radius:12px;
    background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.2);
    font-size:13px;color:#fca5a5;word-break:break-word;text-align:left;
  }
  .btn{
    display:inline-block;padding:12px 32px;border-radius:12px;
    background:linear-gradient(135deg,#6b7280,#4b5563);
    color:#fff;font-size:14px;font-weight:600;
    border:none;cursor:pointer;
    box-shadow:0 4px 16px rgba(75,85,99,.35);
    transition:transform .15s,box-shadow .15s;
  }
  .btn:hover{transform:translateY(-1px);box-shadow:0 6px 24px rgba(75,85,99,.45)}
  .btn:active{transform:translateY(0)}
  .hint{margin-top:20px;font-size:12px;color:#475569}
</style>
</head>
<body>
<div class="card">
  <div class="icon">
    <svg viewBox="0 0 24 24"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
  </div>
  <h1>登录失败</h1>
  <p>身份验证未能完成。</p>
  <div class="error-msg">{error}</div>
  <button class="btn" onclick="window.close()">关闭窗口</button>
  <div class="hint">或手动关闭此标签页</div>
</div>
</body>
</html>
"""

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
                body=_LOGIN_ERROR_HTML.format(error=error_message),
            )
            return

        code = _first_query_value(params, "code")
        if not code:
            self._publish_result(
                _CallbackPayload(error="Auth callback missing code query parameter")
            )
            self._write_html(
                status_code=400,
                body=_LOGIN_ERROR_HTML.format(
                    error="The authorization callback did not include a code."
                ),
            )
            return

        self._publish_result(_CallbackPayload(code=code))
        self._write_html(
            status_code=200,
            body=_LOGIN_SUCCESS_HTML,
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
    """Return the installed Crab CLI package root directory."""
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

    _persist_auth_result(auth_result=auth_result)
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

    user_id = _normalize_optional_string(body.get("ystId"))
    if not user_id:
        raise ValueError("Login response missing body.ystId in JSON body")

    return AuthBootstrapResult(authorization=authorization, user_id=user_id)


def _persist_auth_result(auth_result: AuthBootstrapResult) -> None:
    """Persist the authorization state to the fixed user-level token path."""
    token_path = tg_agent_home() / "token.json"
    _write_auth_result(token_path=token_path, auth_result=auth_result)


def persist_updated_authorization(
    *,
    base_dir: Path | str | None,
    authorization: str,
) -> AuthBootstrapResult | None:
    """Persist a refreshed Authorization token while preserving the current user id."""
    normalized_authorization = _normalize_optional_string(authorization)
    if not normalized_authorization:
        return None

    existing_result = load_persisted_auth_result(base_dir)
    if existing_result is None:
        logger.warning("Skipping Authorization refresh persistence because no token file exists")
        return None

    if existing_result.authorization == normalized_authorization:
        return existing_result

    updated_result = AuthBootstrapResult(
        authorization=normalized_authorization,
        user_id=existing_result.user_id,
    )
    _persist_auth_result(updated_result)
    logger.info("Persisted refreshed Authorization token to user token.json")
    return updated_result


def _write_auth_result(*, token_path: Path, auth_result: AuthBootstrapResult) -> None:
    """Write one auth result payload to the target token path."""
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
    preferred_path = tg_agent_home() / "token.json"
    if preferred_path.exists():
        return preferred_path

    legacy_search_roots: list[Path] = []
    if base_dir is not None:
        legacy_search_roots.append(Path(base_dir).resolve())
    legacy_search_roots.append(tg_agent_home())

    seen: set[Path] = set()
    for search_root in legacy_search_roots:
        if search_root in seen:
            continue
        seen.add(search_root)

        legacy_workspace_path = search_root / ".tg_agent" / "token.json"
        if legacy_workspace_path.exists():
            return legacy_workspace_path

        legacy_crab_path = search_root / ".tg_crab" / "token.json"
        if legacy_crab_path.exists():
            return legacy_crab_path
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
