from __future__ import annotations

import asyncio
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.sandbox import SandboxContext


class DummyAgent:
    def __init__(self, response: str) -> None:
        self.response = response
        self.received_messages: list[str] = []

    async def query(self, message: str) -> str:
        self.received_messages.append(message)
        return self.response


class FakeToolFunction:
    def __init__(self, name: str, arguments: dict) -> None:
        self.name = name
        self.arguments = arguments


class FakeToolCall:
    def __init__(self, name: str, arguments: dict) -> None:
        self.function = FakeToolFunction(name=name, arguments=arguments)


class FakeCompletion:
    def __init__(self, tool_calls: list[FakeToolCall] | None = None) -> None:
        self.tool_calls = tool_calls or []
        self.content = None

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


class FakeLLM:
    def __init__(self, responses: list[FakeCompletion]) -> None:
        self.responses = list(responses)

    @property
    def provider(self) -> str:
        return "fake"

    @property
    def name(self) -> str:
        return "fake-model"

    model = "fake-model"

    async def ainvoke(self, messages, tools=None, tool_choice=None, **kwargs):
        return self.responses.pop(0)

    async def astream(self, messages, tools=None, tool_choice=None, **kwargs):
        if False:
            yield None


def _load_module(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        pytest.fail(f"Expected module '{module_name}' to exist: {exc}")


@pytest.mark.asyncio
async def test_gateway_pipeline_delivers_user_reply_to_channel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bus_events = _load_module("agent_core.bus.events")
    queue_module = _load_module("agent_core.bus.queue")
    channels_base = _load_module("agent_core.channels.base")
    manager_module = _load_module("agent_core.channels.manager")
    dispatcher_module = _load_module("agent_core.gateway.dispatcher")
    service_module = _load_module("agent_core.gateway.service")
    runtime_manager_module = _load_module("agent_core.runtime.manager")

    monkeypatch.setattr(
        dispatcher_module,
        "sync_workspace_agents_md",
        lambda agent, workspace_dir, state: state,
    )

    class FakeChannel(channels_base.BaseChannel):
        name = "telegram"
        display_name = "Telegram"

        def __init__(self, config, bus):
            super().__init__(config, bus)
            self.sent = []
            self._stop_event = asyncio.Event()

        async def start(self) -> None:
            self._running = True
            await self._stop_event.wait()

        async def stop(self) -> None:
            self._running = False
            self._stop_event.set()

        async def send(self, msg) -> None:
            self.sent.append(msg)

    bus = queue_module.MessageBus()
    runtime_manager = runtime_manager_module.RuntimeManager(
        runtime_factory=lambda: (
            DummyAgent(response="hello back"),
            SandboxContext.create(tmp_path),
        )
    )
    dispatcher = dispatcher_module.GatewayDispatcher(bus=bus, runtime_manager=runtime_manager)
    gateway_service = service_module.GatewayService(dispatcher=dispatcher)
    channel_manager = manager_module.ChannelManager(bus)
    channel = FakeChannel(SimpleNamespace(allow_from=["*"]), bus)
    channel_manager.register(channel)

    await gateway_service.start()
    await channel_manager.start_all()
    await bus.publish_inbound(
        bus_events.InboundMessage(
            channel="telegram",
            sender_id="user-1",
            chat_id="chat-1",
            content="hello",
        )
    )

    for _ in range(50):
        if channel.sent:
            break
        await asyncio.sleep(0.01)

    await channel_manager.stop_all()
    await gateway_service.stop()

    assert len(channel.sent) == 1
    assert channel.sent[0].content == "hello back"


@pytest.mark.asyncio
async def test_heartbeat_pipeline_delivers_result_to_last_active_channel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue_module = _load_module("agent_core.bus.queue")
    channels_base = _load_module("agent_core.channels.base")
    manager_module = _load_module("agent_core.channels.manager")
    dispatcher_module = _load_module("agent_core.gateway.dispatcher")
    service_module = _load_module("agent_core.gateway.service")
    heartbeat_module = _load_module("agent_core.heartbeat.service")
    runtime_manager_module = _load_module("agent_core.runtime.manager")

    monkeypatch.setattr(
        dispatcher_module,
        "sync_workspace_agents_md",
        lambda agent, workspace_dir, state: state,
    )

    class FakeChannel(channels_base.BaseChannel):
        name = "telegram"
        display_name = "Telegram"

        def __init__(self, config, bus):
            super().__init__(config, bus)
            self.sent = []
            self._stop_event = asyncio.Event()

        async def start(self) -> None:
            self._running = True
            await self._stop_event.wait()

        async def stop(self) -> None:
            self._running = False
            self._stop_event.set()

        async def send(self, msg) -> None:
            self.sent.append(msg)

    (tmp_path / "HEARTBEAT.md").write_text("- [ ] do thing", encoding="utf-8")

    bus = queue_module.MessageBus()
    runtime_manager = runtime_manager_module.RuntimeManager(
        runtime_factory=lambda: (
            DummyAgent(response="heartbeat done"),
            SandboxContext.create(tmp_path),
        )
    )
    dispatcher = dispatcher_module.GatewayDispatcher(bus=bus, runtime_manager=runtime_manager)
    dispatcher.last_active_private_chat = ("telegram", "chat-1")
    gateway_service = service_module.GatewayService(dispatcher=dispatcher)
    channel_manager = manager_module.ChannelManager(bus)
    channel = FakeChannel(SimpleNamespace(allow_from=["*"]), bus)
    channel_manager.register(channel)
    heartbeat = heartbeat_module.HeartbeatService(
        workspace=tmp_path,
        bus=bus,
        llm=FakeLLM(
            [
                FakeCompletion(
                    [FakeToolCall("heartbeat", {"action": "run", "tasks": "check open tasks"})]
                )
            ]
        ),
        get_delivery_target=lambda: dispatcher.last_active_private_chat,
        interval_seconds=3600,
        enabled=True,
    )

    await gateway_service.start()
    await channel_manager.start_all()
    published = await heartbeat.trigger_now()

    for _ in range(50):
        if channel.sent:
            break
        await asyncio.sleep(0.01)

    await channel_manager.stop_all()
    await gateway_service.stop()

    assert published is True
    assert len(channel.sent) == 1
    assert channel.sent[0].content == "heartbeat done"
