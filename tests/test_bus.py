import importlib
from datetime import datetime

import pytest


def _load_module(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        pytest.fail(f"Expected module '{module_name}' to exist: {exc}")


@pytest.mark.asyncio
async def test_message_bus_round_trips_inbound_and_outbound_messages() -> None:
    events = _load_module("bu_agent_sdk.bus.events")
    queue_module = _load_module("bu_agent_sdk.bus.queue")

    bus = queue_module.MessageBus()

    inbound = events.InboundMessage(
        channel="telegram",
        sender_id="user-1",
        chat_id="chat-1",
        content="hello",
        timestamp=datetime.now(),
        metadata={"message_id": 1},
        origin="user",
    )
    outbound = events.OutboundMessage(
        channel="telegram",
        chat_id="chat-1",
        content="world",
        reply_to_message_id=1,
        metadata={"sent": True},
    )

    await bus.publish_inbound(inbound)
    await bus.publish_outbound(outbound)

    consumed_inbound = await bus.consume_inbound()
    consumed_outbound = await bus.consume_outbound()

    assert consumed_inbound == inbound
    assert consumed_outbound == outbound
    assert bus.inbound_size == 0
    assert bus.outbound_size == 0


@pytest.mark.asyncio
async def test_inbound_message_prefers_session_key_override() -> None:
    events = _load_module("bu_agent_sdk.bus.events")

    overridden = events.InboundMessage(
        channel="telegram",
        sender_id="user-1",
        chat_id="chat-1",
        content="hello",
        session_key_override="heartbeat",
    )
    defaulted = events.InboundMessage(
        channel="telegram",
        sender_id="user-1",
        chat_id="chat-1",
        content="hello",
    )

    assert overridden.session_key == "heartbeat"
    assert defaulted.session_key == "telegram:chat-1"
