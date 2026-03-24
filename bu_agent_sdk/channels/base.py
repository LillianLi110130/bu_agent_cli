"""Base channel abstractions for IM gateway integrations."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from bu_agent_sdk.bus.events import InboundMessage, OutboundMessage
from bu_agent_sdk.bus.queue import MessageBus

logger = logging.getLogger("bu_agent_sdk.channels")


class BaseChannel(ABC):
    """Abstract base class for chat platform integrations."""

    name: str = "base"
    display_name: str = "Base"

    def __init__(self, config: Any, bus: MessageBus):
        self.config = config
        self.bus = bus
        self._running = False

    @abstractmethod
    async def start(self) -> None:
        """Start the channel and begin receiving messages."""

    @abstractmethod
    async def stop(self) -> None:
        """Stop the channel and release its resources."""

    @abstractmethod
    async def send(self, msg: OutboundMessage) -> None:
        """Send an outbound message through this channel."""

    def is_allowed(self, sender_id: str) -> bool:
        """Return whether the sender is allowed to use this channel."""
        allow_list = getattr(self.config, "allow_from", []) or []
        if not allow_list:
            logger.warning(f"{self.name}: allow_from is empty; denying sender {sender_id}")
            return False
        if "*" in allow_list:
            return True
        return str(sender_id) in {str(item) for item in allow_list}

    async def _handle_message(
        self,
        sender_id: str,
        chat_id: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        session_key: str | None = None,
        origin: str = "user",
    ) -> None:
        """Validate and publish an inbound message to the shared bus."""
        if not self.is_allowed(sender_id):
            logger.warning(f"{self.name}: access denied for sender {sender_id}")
            return

        inbound = InboundMessage(
            channel=self.name,
            sender_id=str(sender_id),
            chat_id=str(chat_id),
            content=content,
            metadata=metadata or {},
            session_key_override=session_key,
            origin=origin,
        )
        await self.bus.publish_inbound(inbound)

    @property
    def is_running(self) -> bool:
        """Return whether the channel has been started."""
        return self._running
