from __future__ import annotations

import importlib
from pathlib import Path

import pytest


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
        self.calls = 0

    @property
    def provider(self) -> str:
        return "fake"

    @property
    def name(self) -> str:
        return "fake-model"

    model = "fake-model"

    async def ainvoke(self, messages, tools=None, tool_choice=None, **kwargs):
        self.calls += 1
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
async def test_heartbeat_trigger_now_publishes_inbound_when_decision_is_run(tmp_path: Path) -> None:
    queue_module = _load_module("bu_agent_sdk.bus.queue")
    heartbeat_module = _load_module("bu_agent_sdk.heartbeat.service")

    (tmp_path / "HEARTBEAT.md").write_text("- [ ] check tasks", encoding="utf-8")
    llm = FakeLLM(
        [
            FakeCompletion(
                [FakeToolCall("heartbeat", {"action": "run", "tasks": "check open tasks"})]
            )
        ]
    )
    bus = queue_module.MessageBus()
    service = heartbeat_module.HeartbeatService(
        workspace=tmp_path,
        bus=bus,
        llm=llm,
        get_delivery_target=lambda: ("telegram", "chat-1"),
        interval_seconds=3600,
        enabled=True,
    )

    published = await service.trigger_now()
    inbound = await bus.consume_inbound()

    assert published is True
    assert inbound.origin == "heartbeat"
    assert inbound.session_key == "heartbeat"
    assert inbound.content == "check open tasks"
    assert inbound.chat_id == "chat-1"


@pytest.mark.asyncio
async def test_heartbeat_trigger_now_skips_when_decision_is_skip(tmp_path: Path) -> None:
    queue_module = _load_module("bu_agent_sdk.bus.queue")
    heartbeat_module = _load_module("bu_agent_sdk.heartbeat.service")

    (tmp_path / "HEARTBEAT.md").write_text("nothing urgent", encoding="utf-8")
    llm = FakeLLM([FakeCompletion([FakeToolCall("heartbeat", {"action": "skip"})])])
    bus = queue_module.MessageBus()
    service = heartbeat_module.HeartbeatService(
        workspace=tmp_path,
        bus=bus,
        llm=llm,
        get_delivery_target=lambda: ("telegram", "chat-1"),
        interval_seconds=3600,
        enabled=True,
    )

    published = await service.trigger_now()

    assert published is False
    assert bus.inbound_size == 0


@pytest.mark.asyncio
async def test_heartbeat_trigger_now_skips_without_delivery_target(tmp_path: Path) -> None:
    queue_module = _load_module("bu_agent_sdk.bus.queue")
    heartbeat_module = _load_module("bu_agent_sdk.heartbeat.service")

    (tmp_path / "HEARTBEAT.md").write_text("- [ ] check tasks", encoding="utf-8")
    llm = FakeLLM(
        [
            FakeCompletion(
                [FakeToolCall("heartbeat", {"action": "run", "tasks": "check open tasks"})]
            )
        ]
    )
    bus = queue_module.MessageBus()
    service = heartbeat_module.HeartbeatService(
        workspace=tmp_path,
        bus=bus,
        llm=llm,
        get_delivery_target=lambda: None,
        interval_seconds=3600,
        enabled=True,
    )

    published = await service.trigger_now()

    assert published is False
    assert bus.inbound_size == 0


@pytest.mark.asyncio
async def test_heartbeat_start_is_idempotent(tmp_path: Path) -> None:
    queue_module = _load_module("bu_agent_sdk.bus.queue")
    heartbeat_module = _load_module("bu_agent_sdk.heartbeat.service")

    llm = FakeLLM([])
    bus = queue_module.MessageBus()
    service = heartbeat_module.HeartbeatService(
        workspace=tmp_path,
        bus=bus,
        llm=llm,
        get_delivery_target=lambda: None,
        interval_seconds=9999,
        enabled=True,
    )

    await service.start()
    first_task = service._task
    await service.start()
    await service.stop()

    assert service._task is None
    assert first_task is not None
