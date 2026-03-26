from __future__ import annotations

import asyncio
import importlib
from pathlib import Path

import pytest


class DummyAgent:
    def __init__(self, response: str = "ok", *, raises: Exception | None = None) -> None:
        self.response = response
        self.raises = raises
        self.received_messages: list[str] = []

    async def query(self, message: str) -> str:
        self.received_messages.append(message)
        if self.raises is not None:
            raise self.raises
        return self.response


class FakeRuntime:
    def __init__(self, agent, context) -> None:
        self.agent = agent
        self.context = context
        self.lock = asyncio.Lock()
        self.workspace_instruction_state = object()
        self.touched = 0

    def touch(self) -> None:
        self.touched += 1


class FakeRuntimeManager:
    def __init__(self, runtime) -> None:
        self.runtime = runtime
        self.requested_keys: list[str] = []
        self.cleared_keys: list[str] = []

    async def get_or_create_runtime(self, session_key: str):
        self.requested_keys.append(session_key)
        return self.runtime

    async def clear_runtime(self, session_key: str) -> bool:
        self.cleared_keys.append(session_key)
        return True


def _load_module(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        pytest.fail(f"Expected module '{module_name}' to exist: {exc}")


@pytest.mark.asyncio
async def test_dispatcher_routes_user_message_to_runtime_and_returns_outbound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bus_events = _load_module("agent_core.bus.events")
    dispatcher_module = _load_module("agent_core.gateway.dispatcher")
    sandbox_module = _load_module("tools.sandbox")

    monkeypatch.setattr(
        dispatcher_module,
        "sync_workspace_agents_md",
        lambda agent, workspace_dir, state: state,
    )

    runtime = FakeRuntime(
        agent=DummyAgent(response="hello back"),
        context=sandbox_module.SandboxContext.create(tmp_path),
    )
    runtime_manager = FakeRuntimeManager(runtime)
    dispatcher = dispatcher_module.GatewayDispatcher(
        bus=None,
        runtime_manager=runtime_manager,
    )

    inbound = bus_events.InboundMessage(
        channel="telegram",
        sender_id="user-1",
        chat_id="chat-1",
        content="hello",
        metadata={"message_id": 99},
    )

    outbound = await dispatcher.dispatch(inbound)

    assert outbound is not None
    assert outbound.channel == "telegram"
    assert outbound.chat_id == "chat-1"
    assert outbound.content == "hello back"
    assert outbound.reply_to_message_id == 99
    assert runtime.agent.received_messages == ["hello"]
    assert runtime_manager.requested_keys == ["telegram:chat-1"]
    assert dispatcher.last_active_private_chat == ("telegram", "chat-1")


@pytest.mark.asyncio
async def test_dispatcher_handles_new_command_without_calling_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bus_events = _load_module("agent_core.bus.events")
    dispatcher_module = _load_module("agent_core.gateway.dispatcher")
    sandbox_module = _load_module("tools.sandbox")

    monkeypatch.setattr(
        dispatcher_module,
        "sync_workspace_agents_md",
        lambda agent, workspace_dir, state: state,
    )

    runtime = FakeRuntime(
        agent=DummyAgent(response="should not be used"),
        context=sandbox_module.SandboxContext.create(tmp_path),
    )
    runtime_manager = FakeRuntimeManager(runtime)
    dispatcher = dispatcher_module.GatewayDispatcher(bus=None, runtime_manager=runtime_manager)

    inbound = bus_events.InboundMessage(
        channel="telegram",
        sender_id="user-1",
        chat_id="chat-1",
        content="/new",
    )

    outbound = await dispatcher.dispatch(inbound)

    assert outbound is not None
    assert "new session" in outbound.content.lower()
    assert runtime.agent.received_messages == []
    assert runtime_manager.cleared_keys == ["telegram:chat-1"]
    assert runtime_manager.requested_keys == []


@pytest.mark.asyncio
async def test_dispatcher_returns_friendly_error_when_agent_query_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bus_events = _load_module("agent_core.bus.events")
    dispatcher_module = _load_module("agent_core.gateway.dispatcher")
    sandbox_module = _load_module("tools.sandbox")

    monkeypatch.setattr(
        dispatcher_module,
        "sync_workspace_agents_md",
        lambda agent, workspace_dir, state: state,
    )

    runtime = FakeRuntime(
        agent=DummyAgent(raises=RuntimeError("boom")),
        context=sandbox_module.SandboxContext.create(tmp_path),
    )
    runtime_manager = FakeRuntimeManager(runtime)
    dispatcher = dispatcher_module.GatewayDispatcher(bus=None, runtime_manager=runtime_manager)

    inbound = bus_events.InboundMessage(
        channel="telegram",
        sender_id="user-1",
        chat_id="chat-1",
        content="hello",
    )

    outbound = await dispatcher.dispatch(inbound)

    assert outbound is not None
    assert "sorry" in outbound.content.lower()
    assert runtime_manager.requested_keys == ["telegram:chat-1"]


@pytest.mark.asyncio
async def test_dispatcher_does_not_update_last_active_chat_for_heartbeat_origin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bus_events = _load_module("agent_core.bus.events")
    dispatcher_module = _load_module("agent_core.gateway.dispatcher")
    sandbox_module = _load_module("tools.sandbox")

    monkeypatch.setattr(
        dispatcher_module,
        "sync_workspace_agents_md",
        lambda agent, workspace_dir, state: state,
    )

    runtime = FakeRuntime(
        agent=DummyAgent(response="done"),
        context=sandbox_module.SandboxContext.create(tmp_path),
    )
    runtime_manager = FakeRuntimeManager(runtime)
    dispatcher = dispatcher_module.GatewayDispatcher(bus=None, runtime_manager=runtime_manager)
    dispatcher.last_active_private_chat = ("telegram", "chat-user")

    inbound = bus_events.InboundMessage(
        channel="telegram",
        sender_id="system",
        chat_id="chat-heartbeat",
        content="check heartbeat",
        origin="heartbeat",
        session_key_override="heartbeat",
    )

    outbound = await dispatcher.dispatch(inbound)

    assert outbound is not None
    assert dispatcher.last_active_private_chat == ("telegram", "chat-user")
    assert runtime_manager.requested_keys == ["heartbeat"]


@pytest.mark.asyncio
async def test_gateway_service_starts_dispatch_loop_and_publishes_outbound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bus_events = _load_module("agent_core.bus.events")
    queue_module = _load_module("agent_core.bus.queue")
    dispatcher_module = _load_module("agent_core.gateway.dispatcher")
    service_module = _load_module("agent_core.gateway.service")
    sandbox_module = _load_module("tools.sandbox")

    monkeypatch.setattr(
        dispatcher_module,
        "sync_workspace_agents_md",
        lambda agent, workspace_dir, state: state,
    )

    bus = queue_module.MessageBus()
    runtime = FakeRuntime(
        agent=DummyAgent(response="loop ok"),
        context=sandbox_module.SandboxContext.create(tmp_path),
    )
    runtime_manager = FakeRuntimeManager(runtime)
    dispatcher = dispatcher_module.GatewayDispatcher(bus=bus, runtime_manager=runtime_manager)
    service = service_module.GatewayService(dispatcher=dispatcher)

    await service.start()
    await bus.publish_inbound(
        bus_events.InboundMessage(
            channel="telegram",
            sender_id="user-1",
            chat_id="chat-1",
            content="hello",
        )
    )

    outbound = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
    await service.stop()

    assert outbound.content == "loop ok"
    assert runtime_manager.requested_keys == ["telegram:chat-1"]
