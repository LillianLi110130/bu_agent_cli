"""Async in-memory message bus for gateway integrations."""

from __future__ import annotations

import asyncio

from bu_agent_sdk.bus.events import InboundMessage, OutboundMessage


class MessageBus:
    """Simple async queue-based bus that decouples channels from the gateway."""

    def __init__(self) -> None:
        self.inbound: asyncio.Queue[InboundMessage] = asyncio.Queue()
        self.outbound: asyncio.Queue[OutboundMessage] = asyncio.Queue()

    async def publish_inbound(self, message: InboundMessage) -> None:
        """Publish an inbound message from a chat channel."""
        await self.inbound.put(message)

    async def consume_inbound(self) -> InboundMessage:
        """Consume the next inbound message."""
        return await self.inbound.get()

    async def publish_outbound(self, message: OutboundMessage) -> None:
        """Publish an outbound message to be sent by a chat channel."""
        await self.outbound.put(message)

    async def consume_outbound(self) -> OutboundMessage:
        """Consume the next outbound message."""
        return await self.outbound.get()

    @property
    def inbound_size(self) -> int:
        """Return the number of pending inbound messages."""
        return self.inbound.qsize()

    @property
    def outbound_size(self) -> int:
        """Return the number of pending outbound messages."""
        return self.outbound.qsize()
