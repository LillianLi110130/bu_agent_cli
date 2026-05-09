from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import httpx
import pytest

from agent_core.llm.gateway.chat import ChatGateway
from agent_core.llm.messages import UserMessage
from agent_core.server.llm_gateway import LLMGatewayService
from agent_core.server.models import LLMQueryRequest
from agent_core.server.route_config import GatewayRoute, load_gateway_routes
from cli.worker.auth import AuthBootstrapResult
from config.model_config import resolve_model_config


def _make_temp_file() -> Path:
    return Path.cwd() / f"test_gateway_routes_{uuid4().hex}.json"


def test_resolve_model_config_supports_gateway_provider_without_dedicated_key_env() -> None:
    presets = {
        "coding": {
            "provider": "gateway",
            "model": "coding-default",
            "base_url": "https://gateway.example.com",
            "vision": False,
        }
    }

    resolved = resolve_model_config("coding", presets=presets)

    assert resolved.provider == "gateway"
    assert resolved.model == "coding-default"
    assert resolved.base_url == "https://gateway.example.com"
    assert resolved.api_key is None


@pytest.mark.asyncio
async def test_chat_gateway_aggregates_sse_events(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured_auth: str | None = None
    persisted_updates: list[tuple[Path, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_auth
        captured_auth = request.headers.get("Authorization")
        payload = json.loads(request.content.decode("utf-8"))
        assert payload["model"] == "coding-default"
        assert payload["messages"][0]["role"] == "user"

        body = (
            'data: {"type":"text","content":"hello "}\n\n'
            'data: {"type":"text","content":"world"}\n\n'
            'data: {"type":"tool_call","tool":"read","args":{"path":"README.md"},'
            '"tool_call_id":"call_1","display_name":"read"}\n\n'
            'data: {"type":"usage","usage":{"prompt_tokens":10,"prompt_cached_tokens":0,'
            '"prompt_cache_creation_tokens":null,"prompt_image_tokens":null,'
            '"completion_tokens":5,"total_tokens":15}}\n\n'
            'data: {"type":"done","stop_reason":"tool_calls"}\n\n'
            ': done\n\n'
        )
        return httpx.Response(
            200,
            text=body,
            headers={
                "content-type": "text/event-stream",
                "Authorization": "Bearer refreshed-token",
            },
        )

    monkeypatch.setattr(
        "agent_core.llm.gateway.chat.load_persisted_auth_result",
        lambda base_dir=None: AuthBootstrapResult(
            authorization="Bearer shared-login-token",
            user_id="u-1",
        ),
    )
    monkeypatch.setattr(
        "agent_core.llm.gateway.chat.persist_updated_authorization",
        lambda *, base_dir, authorization: persisted_updates.append(
            (Path(base_dir).resolve(), authorization)
        ),
    )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        llm = ChatGateway(
            model="coding-default",
            base_url="https://gateway.example.com",
            http_client=client,
            base_dir=tmp_path,
        )
        response = await llm.ainvoke([UserMessage(content="hi")])

    assert captured_auth == "Bearer shared-login-token"
    assert response.content == "hello world"
    assert response.stop_reason == "tool_calls"
    assert response.usage is not None
    assert response.usage.total_tokens == 15
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].function.name == "read"
    assert json.loads(response.tool_calls[0].function.arguments) == {"path": "README.md"}
    assert persisted_updates == [(tmp_path.resolve(), "Bearer refreshed-token")]


@pytest.mark.asyncio
async def test_chat_gateway_falls_back_to_explicit_token_when_no_persisted_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_auth: str | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_auth
        captured_auth = request.headers.get("Authorization")
        return httpx.Response(
            200,
            text='data: {"type":"done","stop_reason":"stop"}\n\n: done\n\n',
            headers={"content-type": "text/event-stream"},
        )

    monkeypatch.setattr(
        "agent_core.llm.gateway.chat.load_persisted_auth_result",
        lambda base_dir=None: None,
    )
    monkeypatch.setattr(
        "agent_core.llm.gateway.chat.persist_updated_authorization",
        lambda **_: None,
    )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        llm = ChatGateway(
            model="coding-default",
            api_key="explicit-token",
            base_url="https://gateway.example.com",
            http_client=client,
        )
        await llm.ainvoke([UserMessage(content="hi")])

    assert captured_auth == "Bearer explicit-token"


def test_load_gateway_routes_reads_alias_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = _make_temp_file()
    config_path.write_text(
        json.dumps(
            {
                "routes": {
                    "coding-default": {
                        "provider": "openai",
                        "upstream_model": "GLM-5.1",
                        "base_url": "https://example.invalid/v1",
                        "api_key_env": "OPENAI_API_KEY",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    try:
        routes = load_gateway_routes(config_path)

        assert routes["coding-default"].provider == "openai"
        assert routes["coding-default"].upstream_model == "GLM-5.1"
        assert routes["coding-default"].base_url == "https://example.invalid/v1"
        assert routes["coding-default"].api_key_env == "OPENAI_API_KEY"
    finally:
        config_path.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_llm_gateway_service_routes_alias_to_upstream_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, str | None] = {}

    class FakeChatOpenAI:
        def __init__(self, *, model: str, api_key: str | None, base_url: str | None, **_: object):
            captured["model"] = model
            captured["api_key"] = api_key
            captured["base_url"] = base_url

        async def astream(self, **_: object):
            yield type("Chunk", (), {"delta": "ok", "thinking": None, "tool_calls": [], "usage": None, "stop_reason": "stop"})()

        async def close(self) -> None:
            return None

    monkeypatch.setenv("OPENAI_API_KEY", "upstream-key")
    monkeypatch.setattr("agent_core.server.llm_gateway.ChatOpenAI", FakeChatOpenAI)

    service = LLMGatewayService(
        routes={
            "coding-default": GatewayRoute(
                alias="coding-default",
                provider="openai",
                upstream_model="GLM-5.1",
                base_url="https://gateway-upstream.example.com/v1",
                api_key_env="OPENAI_API_KEY",
            )
        }
    )

    request = LLMQueryRequest(
        model="coding-default",
        messages=[UserMessage(content="hi")],
    )
    events = [event async for event in service.query_stream(request)]

    assert captured["model"] == "GLM-5.1"
    assert captured["api_key"] == "upstream-key"
    assert captured["base_url"] == "https://gateway-upstream.example.com/v1"
    assert events[-1].type == "done"
