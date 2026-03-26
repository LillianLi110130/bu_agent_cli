from __future__ import annotations

import pytest

from agent_core.server.client import AgentClient


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class RecordingAsyncClient:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)
        self.posts: list[tuple[str, dict]] = []

    async def post(self, url: str, json: dict, timeout=None):
        self.posts.append((url, json))
        return FakeResponse(self.responses.pop(0))

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_agent_client_create_session_sends_user_id() -> None:
    client = AgentClient(base_url="http://localhost:8000", user_id="user-1")
    client._client = RecordingAsyncClient(
        responses=[{"session_id": "session-1", "created_at": "2026-03-18T10:00:00+00:00"}]
    )

    session_id = await client.create_session()

    assert session_id == "session-1"
    assert client._client.posts == [("http://localhost:8000/sessions", {"user_id": "user-1"})]


@pytest.mark.asyncio
async def test_agent_client_query_sends_user_id_without_session_id_and_updates_session() -> None:
    client = AgentClient(base_url="http://localhost:8000", user_id="user-1")
    client._client = RecordingAsyncClient(
        responses=[
            {
                "session_id": "session-1",
                "response": "echo:hello",
                "usage": {
                    "total_tokens": 0,
                    "total_prompt_tokens": 0,
                    "total_completion_tokens": 0,
                    "by_model": {},
                },
                "skill_used": None,
            }
        ]
    )

    response = await client.query("hello")

    assert response.session_id == "session-1"
    assert client.session_id == "session-1"
    assert client._client.posts == [
        (
            "http://localhost:8000/agent/query",
            {
                "message": "hello",
                "session_id": None,
                "user_id": "user-1",
                "stream": False,
                "skill": None,
            },
        )
    ]


@pytest.mark.asyncio
async def test_agent_client_query_requires_user_id_or_existing_session() -> None:
    client = AgentClient(base_url="http://localhost:8000")

    with pytest.raises(RuntimeError, match="user_id is required"):
        await client.query("hello")
