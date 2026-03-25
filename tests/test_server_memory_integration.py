from __future__ import annotations

from typing import AsyncIterator
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from bu_agent_sdk.llm.messages import AssistantMessage, BaseMessage, DeveloperMessage, UserMessage
from bu_agent_sdk.server.app import create_app
from bu_agent_sdk.server.session import AgentSession, SessionManager
from bu_agent_sdk.tokens.views import UsageSummary


class DummyAgent:
    def __init__(self) -> None:
        self.system_prompt = "system prompt"
        self.messages: list[BaseMessage] = []
        self.loaded_history: list[BaseMessage] = []
        self.context_snapshots: list[list[tuple[str, str]]] = []

    def _snapshot_context(self) -> None:
        snapshot: list[tuple[str, str]] = []
        for message in self.messages:
            content = getattr(message, "content", "")
            if not isinstance(content, str):
                content = str(content)
            snapshot.append((message.role, content))
        self.context_snapshots.append(snapshot)

    async def query(self, message: str) -> str:
        self._snapshot_context()
        self.messages.append(UserMessage(content=message))
        reply = f"echo:{message}"
        self.messages.append(AssistantMessage(content=reply))
        return reply

    async def query_stream(self, message: str) -> AsyncIterator[object]:
        raise NotImplementedError

    async def query_stream_delta(self, message: str) -> AsyncIterator[object]:
        raise NotImplementedError

    async def get_usage(self) -> UsageSummary:
        return UsageSummary(
            total_prompt_tokens=0,
            total_prompt_cached_tokens=0,
            total_completion_tokens=0,
            total_tokens=0,
            entry_count=0,
            by_model={},
        )

    def clear_history(self) -> None:
        self.messages = []

    def load_history(self, messages: list[BaseMessage]) -> None:
        self.loaded_history = list(messages)
        self.messages = list(messages)


class StreamingDummyAgent(DummyAgent):
    async def query_stream(self, message: str) -> AsyncIterator[object]:
        from bu_agent_sdk.agent.events import FinalResponseEvent, TextEvent

        self._snapshot_context()
        yield TextEvent(content=f"partial:{message}")
        yield FinalResponseEvent(content=f"echo:{message}")


class StreamingDeltaDummyAgent(DummyAgent):
    async def query_stream_delta(self, message: str) -> AsyncIterator[object]:
        from bu_agent_sdk.agent.events import FinalResponseEvent, TextDeltaEvent

        self._snapshot_context()
        yield TextDeltaEvent(delta=f"partial:{message}")
        yield FinalResponseEvent(content=f"echo:{message}")


class DummyMemoryService:
    def __init__(self) -> None:
        self.load_calls: list[tuple[str, str]] = []
        self.user_memory_calls: list[str] = []
        self.append_calls: list[tuple[str, str, str, str]] = []
        self.user_memories: list[str] = ["memory one", "memory two"]

    async def load_history(self, session_id: str, user_id: str) -> list[BaseMessage]:
        self.load_calls.append((session_id, user_id))
        return [
            UserMessage(content="persisted user"),
            AssistantMessage(content="persisted assistant"),
        ]

    async def load_user_memory_context(self, user_id: str) -> DeveloperMessage | None:
        self.user_memory_calls.append(user_id)
        if not self.user_memories:
            return None
        content = "User memory context:\n" + "\n".join(
            f"- {memory}" for memory in self.user_memories
        )
        return DeveloperMessage(content=content)

    async def append_round(
        self,
        session_id: str,
        user_id: str,
        user_message: str,
        assistant_message: str,
    ) -> None:
        self.append_calls.append((session_id, user_id, user_message, assistant_message))


@pytest.fixture
def test_client() -> TestClient:
    app = create_app(agent_factory=DummyAgent, config=None)
    return TestClient(app)


def test_query_requires_user_id_when_session_id_missing(test_client: TestClient) -> None:
    with test_client as client:
        response = client.post("/agent/query", json={"message": "hello"})

    assert response.status_code == 400
    assert response.json()["detail"] == "user_id is required when session_id is missing"


def test_query_requires_user_id_for_new_session_id(test_client: TestClient) -> None:
    with test_client as client:
        response = client.post(
            "/agent/query",
            json={"message": "hello", "session_id": "session-1"},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "user_id is required for a new or unbound session"


def test_query_requires_user_id_for_unbound_session(test_client: TestClient) -> None:
    with test_client as client:
        create_response = client.post("/sessions", json={})
        session_id = create_response.json()["session_id"]
        query_response = client.post(
            "/agent/query",
            json={"message": "hello", "session_id": session_id},
        )

    assert create_response.status_code == 200
    assert query_response.status_code == 400
    assert query_response.json()["detail"] == "user_id is required for a new or unbound session"


def test_query_accepts_bound_session_without_repeating_user_id() -> None:
    app = create_app(agent_factory=DummyAgent, config=None)

    with TestClient(app) as client:
        create_response = client.post("/sessions", json={"user_id": "user-1"})
        session_id = create_response.json()["session_id"]
        query_response = client.post(
            "/agent/query",
            json={"message": "hello", "session_id": session_id},
        )

    assert create_response.status_code == 200
    assert query_response.status_code == 200


def test_query_rejects_mismatched_user_id_on_bound_session() -> None:
    app = create_app(agent_factory=DummyAgent, config=None)

    with TestClient(app) as client:
        create_response = client.post("/sessions", json={"user_id": "user-1"})
        session_id = create_response.json()["session_id"]
        query_response = client.post(
            "/agent/query",
            json={"message": "hello", "session_id": session_id, "user_id": "user-2"},
        )

    assert create_response.status_code == 200
    assert query_response.status_code == 400
    assert query_response.json()["detail"] == "session user_id mismatch"


def test_create_session_binds_user_id_and_exposes_it_in_session_info() -> None:
    app = create_app(agent_factory=DummyAgent, config=None)

    with TestClient(app) as client:
        create_response = client.post("/sessions", json={"user_id": "user-1"})
        session_id = create_response.json()["session_id"]
        info_response = client.get(f"/sessions/{session_id}")

    assert create_response.status_code == 200
    assert info_response.status_code == 200
    assert info_response.json()["user_id"] == "user-1"


@pytest.mark.asyncio
async def test_session_manager_binds_user_id_and_rejects_mismatch() -> None:
    manager = SessionManager(agent_factory=DummyAgent)

    session = await manager.get_or_create_session(session_id="session-1", user_id="user-1")

    assert session.user_id == "user-1"

    same_session = await manager.get_or_create_session(session_id="session-1", user_id="user-1")

    assert same_session is session

    with pytest.raises(ValueError, match="session user_id mismatch"):
        await manager.get_or_create_session(session_id="session-1", user_id="user-2")


def test_tg_mem_memory_can_be_instantiated_in_mysql_only_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    from tg_mem.memory import main as tg_mem_main

    monkeypatch.setattr(tg_mem_main, "_create_mysql_manager", lambda config: object())

    memory = tg_mem_main.Memory()

    assert memory.db is not None
    assert memory._is_vector_store_enabled() is False


@pytest.mark.asyncio
async def test_agent_session_loads_history_only_once_from_memory_service() -> None:
    session = AgentSession(
        session_id="session-1",
        agent=DummyAgent(),
        user_id="user-1",
        memory_service=DummyMemoryService(),
    )

    first_response = await session.query("hello")
    second_response = await session.query("again")

    assert first_response == "echo:hello"
    assert second_response == "echo:again"
    assert session.agent.loaded_history[0].role == "system"
    assert session.agent.loaded_history[1].content == "persisted user"
    assert session.agent.loaded_history[2].content == "persisted assistant"
    assert session.memory_service.load_calls == [("session-1", "user-1")]
    assert session.memory_service.user_memory_calls == ["user-1", "user-1"]
    assert any(role == "developer" for role, _ in session.agent.context_snapshots[0])
    assert all(message.role != "developer" for message in session.agent.messages)


@pytest.mark.asyncio
async def test_tg_mem_memory_service_loads_user_and_assistant_history_only() -> None:
    from bu_agent_sdk.server.memory_service import TgMemMemoryService

    memory = SimpleNamespace(
        db=SimpleNamespace(
            get_conversation_records=MagicMock(
                return_value=[
                    {"role": "user", "content": "persisted user"},
                    {"role": "assistant", "content": "persisted assistant"},
                    {"role": "tool", "content": "ignored tool"},
                ]
            )
        )
    )
    service = TgMemMemoryService(memory=memory)

    messages = await service.load_history(session_id="session-1", user_id="user-1")

    memory.db.get_conversation_records.assert_called_once_with(session_id="session-1")
    assert [message.role for message in messages] == ["user", "assistant"]
    assert [message.content for message in messages] == ["persisted user", "persisted assistant"]


@pytest.mark.asyncio
async def test_tg_mem_memory_service_loads_all_user_memories_with_char_limit() -> None:
    from bu_agent_sdk.server.memory_service import TgMemMemoryService

    memory = SimpleNamespace(
        db=SimpleNamespace(
            list_memory_records=MagicMock(
                return_value=[
                    {"memory_data": "first memory"},
                    {"memory_data": "second memory"},
                ]
            )
        )
    )
    service = TgMemMemoryService(memory=memory, max_memory_context_chars=60)

    message = await service.load_user_memory_context(user_id="user-1")

    memory.db.list_memory_records.assert_called_once_with(user_id="user-1", status="ACTE", limit=None)
    assert message is not None
    assert message.role == "developer"
    assert "User memory context:" in message.content
    assert "first memory" in message.content
    assert len(message.content) <= 60


@pytest.mark.asyncio
async def test_tg_mem_memory_service_appends_single_round_with_infer_true() -> None:
    from bu_agent_sdk.server.memory_service import TgMemMemoryService

    memory = SimpleNamespace(add=MagicMock(return_value={"results": []}))
    service = TgMemMemoryService(memory=memory)

    await service.append_round(
        session_id="session-1",
        user_id="user-1",
        user_message="hello",
        assistant_message="hi",
    )

    memory.add.assert_called_once_with(
        messages=[
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ],
        user_id="user-1",
        session_id="session-1",
        infer=True,
    )


def test_query_persists_single_round_after_success() -> None:
    memory_service = DummyMemoryService()
    app = create_app(agent_factory=DummyAgent, config=None, memory_service=memory_service)

    with TestClient(app) as client:
        response = client.post(
            "/agent/query",
            json={"message": "hello", "user_id": "user-1"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert memory_service.append_calls == [
        (payload["session_id"], "user-1", "hello", "echo:hello")
    ]


def test_query_stream_persists_only_after_final_event() -> None:
    memory_service = DummyMemoryService()
    app = create_app(agent_factory=StreamingDummyAgent, config=None, memory_service=memory_service)

    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/agent/query-stream",
            json={"message": "hello", "user_id": "user-1"},
        ) as response:
            chunks = list(response.iter_text())

    assert response.status_code == 200
    assert any('"type": "final"' in chunk for chunk in chunks)
    assert len(memory_service.append_calls) == 1
    session_id, user_id, user_message, assistant_message = memory_service.append_calls[0]
    assert session_id
    assert user_id == "user-1"
    assert user_message == "hello"
    assert assistant_message == "echo:hello"


def test_query_stream_delta_persists_only_after_final_event() -> None:
    memory_service = DummyMemoryService()
    app = create_app(agent_factory=StreamingDeltaDummyAgent, config=None, memory_service=memory_service)

    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/agent/query-stream-delta",
            json={"message": "hello", "user_id": "user-1"},
        ) as response:
            chunks = list(response.iter_text())

    assert response.status_code == 200
    assert any('"type": "final"' in chunk for chunk in chunks)
    assert len(memory_service.append_calls) == 1
    session_id, user_id, user_message, assistant_message = memory_service.append_calls[0]
    assert session_id
    assert user_id == "user-1"
    assert user_message == "hello"
    assert assistant_message == "echo:hello"


@pytest.mark.asyncio
async def test_clear_history_triggers_reload_on_next_query() -> None:
    memory_service = DummyMemoryService()
    session = AgentSession(
        session_id="session-1",
        agent=DummyAgent(),
        user_id="user-1",
        memory_service=memory_service,
    )

    await session.query("hello")
    await session.clear_history()
    await session.query("again")

    assert memory_service.load_calls == [
        ("session-1", "user-1"),
        ("session-1", "user-1"),
    ]


def test_build_memory_service_from_env_returns_none_without_db_uri(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from bu_agent_sdk.server.memory_service import build_memory_service_from_env

    monkeypatch.delenv("TG_MEM_MYSQL_DB_URI", raising=False)
    monkeypatch.delenv("TG_MEM_ENABLED", raising=False)

    assert build_memory_service_from_env() is None


def test_build_memory_service_from_env_creates_tg_mem_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from bu_agent_sdk.server.memory_service import TgMemMemoryService, build_memory_service_from_env

    recorded_config: dict[str, object] = {}

    class FakeMemory:
        @classmethod
        def from_config(cls, config: dict[str, object]):
            recorded_config.update(config)
            return SimpleNamespace(name="fake-memory")

    monkeypatch.setenv("TG_MEM_MYSQL_DB_URI", "mysql://root:@localhost:3306/mem0")
    monkeypatch.setenv("TG_MEM_LLM_MODEL", "glm-test")
    monkeypatch.setenv("TG_MEM_OPENAI_BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("TG_MEM_OPENAI_API_KEY", "secret")
    monkeypatch.setattr("tg_mem.Memory", FakeMemory)

    service = build_memory_service_from_env()

    assert isinstance(service, TgMemMemoryService)
    assert service.memory.name == "fake-memory"
    assert recorded_config == {
        "vector_store": {"provider": "none", "config": None},
        "mysql": {"db_uri": "mysql://root:@localhost:3306/mem0"},
        "llm": {
            "provider": "openai_like",
            "config": {
                "model": "glm-test",
                "api_key": "secret",
                "openai_base_url": "https://example.com/v1",
            },
        },
    }


def test_create_app_uses_env_memory_service_when_not_explicitly_provided(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory_service = DummyMemoryService()

    monkeypatch.setattr(
        "bu_agent_sdk.server.memory_service.build_memory_service_from_env",
        lambda: memory_service,
    )

    app = create_app(agent_factory=DummyAgent, config=None)

    with TestClient(app) as client:
        response = client.post(
            "/agent/query",
            json={"message": "hello", "user_id": "user-1"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert memory_service.append_calls == [
        (payload["session_id"], "user-1", "hello", "echo:hello")
    ]
