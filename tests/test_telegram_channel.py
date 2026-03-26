from __future__ import annotations

import asyncio
import importlib
from types import SimpleNamespace

import pytest


def _load_module(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        pytest.fail(f"Expected module '{module_name}' to exist: {exc}")


class _FakeBot:
    def __init__(self) -> None:
        self.sent_messages: list[dict] = []
        self.sent_actions: list[dict] = []

    async def get_me(self):
        return SimpleNamespace(id=999, username="bu_agent_test")

    async def send_message(self, **kwargs) -> None:
        self.sent_messages.append(kwargs)

    async def send_chat_action(self, **kwargs) -> None:
        self.sent_actions.append(kwargs)


class _FakeUpdater:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False

    async def start_polling(self, **kwargs) -> None:
        self.started = True
        self.kwargs = kwargs

    async def stop(self) -> None:
        self.stopped = True


class _FakeApp:
    def __init__(self) -> None:
        self.bot = _FakeBot()
        self.updater = _FakeUpdater()
        self.handlers = []
        self.initialized = False
        self.started = False
        self.stopped = False
        self.shutdown_called = False

    def add_handler(self, handler) -> None:
        self.handlers.append(handler)

    async def initialize(self) -> None:
        self.initialized = True

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def shutdown(self) -> None:
        self.shutdown_called = True


class _FakeBuilder:
    def __init__(self, app: _FakeApp) -> None:
        self.app = app
        self.token_value = None
        self.request_value = None
        self.get_updates_request_value = None

    def token(self, token: str):
        self.token_value = token
        return self

    def request(self, request):
        self.request_value = request
        return self

    def get_updates_request(self, request):
        self.get_updates_request_value = request
        return self

    def build(self):
        return self.app


def _make_update(*, chat_type: str = "private", text: str = "hello", user_id: int = 123):
    user = SimpleNamespace(id=user_id, username="alice", first_name="Alice")
    message = SimpleNamespace(
        chat=SimpleNamespace(type=chat_type),
        chat_id=456,
        text=text,
        message_id=42,
    )
    return SimpleNamespace(message=message, effective_user=user)


@pytest.mark.asyncio
async def test_telegram_channel_forwards_private_message_to_bus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue_module = _load_module("agent_core.bus.queue")
    telegram_module = _load_module("agent_core.channels.telegram")

    bus = queue_module.MessageBus()
    config = SimpleNamespace(token="123:abc", allow_from=["*"], proxy=None)
    channel = telegram_module.TelegramChannel(config, bus)
    channel._app = _FakeApp()
    channel._start_typing = lambda _chat_id: None

    await channel._on_message(_make_update(text="hello"), None)
    inbound = await bus.consume_inbound()

    assert inbound.channel == "telegram"
    assert inbound.chat_id == "456"
    assert inbound.sender_id == "123"
    assert inbound.content == "hello"
    assert inbound.metadata["message_id"] == 42


@pytest.mark.asyncio
async def test_telegram_channel_ignores_non_private_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue_module = _load_module("agent_core.bus.queue")
    telegram_module = _load_module("agent_core.channels.telegram")

    bus = queue_module.MessageBus()
    config = SimpleNamespace(token="123:abc", allow_from=["*"], proxy=None)
    channel = telegram_module.TelegramChannel(config, bus)
    channel._app = _FakeApp()
    channel._start_typing = lambda _chat_id: None

    await channel._on_message(_make_update(chat_type="group", text="hello"), None)

    assert bus.inbound_size == 0


@pytest.mark.asyncio
async def test_telegram_channel_rejects_sender_not_in_allowlist() -> None:
    queue_module = _load_module("agent_core.bus.queue")
    telegram_module = _load_module("agent_core.channels.telegram")

    bus = queue_module.MessageBus()
    config = SimpleNamespace(token="123:abc", allow_from=["999"], proxy=None)
    channel = telegram_module.TelegramChannel(config, bus)
    channel._app = _FakeApp()
    channel._start_typing = lambda _chat_id: None

    await channel._on_message(_make_update(text="hello", user_id=123), None)

    assert bus.inbound_size == 0


@pytest.mark.asyncio
async def test_telegram_channel_send_splits_long_messages_and_stops_typing() -> None:
    bus_events = _load_module("agent_core.bus.events")
    telegram_module = _load_module("agent_core.channels.telegram")

    config = SimpleNamespace(token="123:abc", allow_from=["*"], proxy=None)
    channel = telegram_module.TelegramChannel(config, bus=None)
    channel._app = _FakeApp()
    stopped_chat_ids: list[str] = []
    channel._stop_typing = lambda chat_id: stopped_chat_ids.append(chat_id)

    long_text = "x" * (telegram_module.TELEGRAM_MAX_MESSAGE_LEN + 10)
    message = bus_events.OutboundMessage(channel="telegram", chat_id="456", content=long_text)

    await channel.send(message)

    assert len(channel._app.bot.sent_messages) == 2
    assert stopped_chat_ids == ["456"]


@pytest.mark.asyncio
async def test_channel_manager_dispatches_outbound_messages() -> None:
    bus_events = _load_module("agent_core.bus.events")
    queue_module = _load_module("agent_core.bus.queue")
    channels_base = _load_module("agent_core.channels.base")
    manager_module = _load_module("agent_core.channels.manager")

    class FakeChannel(channels_base.BaseChannel):
        name = "fake"
        display_name = "Fake"

        def __init__(self, config, bus):
            super().__init__(config, bus)
            self.started = 0
            self.stopped = 0
            self.sent: list[object] = []
            self._stop_event = asyncio.Event()

        async def start(self) -> None:
            self._running = True
            self.started += 1
            await self._stop_event.wait()

        async def stop(self) -> None:
            self._running = False
            self.stopped += 1
            self._stop_event.set()

        async def send(self, msg) -> None:
            self.sent.append(msg)

    bus = queue_module.MessageBus()
    manager = manager_module.ChannelManager(bus)
    channel = FakeChannel(SimpleNamespace(allow_from=["*"]), bus)
    manager.register(channel)

    await manager.start_all()
    await bus.publish_outbound(
        bus_events.OutboundMessage(channel="fake", chat_id="1", content="hello")
    )

    for _ in range(50):
        if channel.sent:
            break
        await asyncio.sleep(0.01)

    await manager.stop_all()

    assert channel.started == 1
    assert channel.stopped == 1
    assert len(channel.sent) == 1


@pytest.mark.asyncio
async def test_telegram_channel_start_builds_application(monkeypatch: pytest.MonkeyPatch) -> None:
    queue_module = _load_module("agent_core.bus.queue")
    telegram_module = _load_module("agent_core.channels.telegram")

    bus = queue_module.MessageBus()
    config = SimpleNamespace(token="123:abc", allow_from=["*"], proxy=None)
    channel = telegram_module.TelegramChannel(config, bus)
    app = _FakeApp()
    builder = _FakeBuilder(app)

    monkeypatch.setattr(
        telegram_module,
        "Application",
        SimpleNamespace(builder=lambda: builder),
    )
    monkeypatch.setattr(
        telegram_module,
        "HTTPXRequest",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )

    async def _fake_sleep(_seconds: float) -> None:
        channel._running = False

    monkeypatch.setattr(telegram_module.asyncio, "sleep", _fake_sleep)

    await channel.start()

    assert channel._app is app
    assert app.initialized is True
    assert app.started is True
    assert app.updater.started is True
    assert len(app.handlers) == 1
