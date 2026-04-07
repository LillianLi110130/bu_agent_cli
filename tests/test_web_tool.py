from __future__ import annotations

import httpx
from pathlib import Path
import pytest
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import ALL_TOOLS
from tools.web import web_fetch


class _FakeResponse:
    def __init__(self, *, status_code: int, text: str, content_type: str) -> None:
        self.status_code = status_code
        self.text = text
        self.headers = {"content-type": content_type}


class _FakeAsyncClient:
    def __init__(self, response: _FakeResponse | None = None, error: Exception | None = None) -> None:
        self._response = response
        self._error = error

    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def get(self, url: str, *, headers: dict[str, str], follow_redirects: bool) -> _FakeResponse:
        assert url == "https://example.com"
        assert follow_redirects is True
        assert "User-Agent" in headers
        if self._error is not None:
            raise self._error
        assert self._response is not None
        return self._response


@pytest.mark.asyncio
async def test_web_fetch_returns_plain_text(monkeypatch: pytest.MonkeyPatch) -> None:
    response = _FakeResponse(
        status_code=200,
        text="hello from server",
        content_type="text/plain; charset=utf-8",
    )
    monkeypatch.setattr("tools.web.httpx.AsyncClient", lambda **_: _FakeAsyncClient(response=response))

    result = await web_fetch.execute(url="https://example.com")

    assert result == "hello from server"


@pytest.mark.asyncio
async def test_web_fetch_extracts_text_from_html(monkeypatch: pytest.MonkeyPatch) -> None:
    response = _FakeResponse(
        status_code=200,
        text=(
            "<html><head><title>Demo</title><style>.x{color:red}</style></head>"
            "<body><main><h1>Hello</h1><p>World</p><script>ignore()</script></main></body></html>"
        ),
        content_type="text/html; charset=utf-8",
    )
    monkeypatch.setattr("tools.web.httpx.AsyncClient", lambda **_: _FakeAsyncClient(response=response))

    result = await web_fetch.execute(url="https://example.com")

    assert "Hello" in result
    assert "World" in result
    assert "ignore()" not in result
    assert ".x{color:red}" not in result


@pytest.mark.asyncio
async def test_web_fetch_reports_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    response = _FakeResponse(
        status_code=404,
        text="missing",
        content_type="text/plain; charset=utf-8",
    )
    monkeypatch.setattr("tools.web.httpx.AsyncClient", lambda **_: _FakeAsyncClient(response=response))

    result = await web_fetch.execute(url="https://example.com")

    assert result == "Error: Failed to fetch URL. Status: 404."


@pytest.mark.asyncio
async def test_web_fetch_reports_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    error = httpx.TimeoutException("timed out")
    monkeypatch.setattr("tools.web.httpx.AsyncClient", lambda **_: _FakeAsyncClient(error=error))

    result = await web_fetch.execute(url="https://example.com")

    assert result == "Error: Failed to fetch URL: request timed out."


def test_web_fetch_is_registered() -> None:
    assert web_fetch in ALL_TOOLS
    assert web_fetch.name == "WebFetch"
