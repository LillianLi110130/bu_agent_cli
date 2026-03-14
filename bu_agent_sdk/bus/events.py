"""Event models for the IM gateway message bus."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal


@dataclass
class InboundMessage:
    """Message received from an external chat channel."""

    channel: str
    sender_id: str
    chat_id: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    origin: Literal["user", "heartbeat", "system"] = "user"
    session_key_override: str | None = None

    @property
    def session_key(self) -> str:
        """Return the effective session key for this inbound message."""
        return self.session_key_override or f"{self.channel}:{self.chat_id}"


@dataclass
class OutboundMessage:
    """Message that should be delivered to an external chat channel."""

    channel: str
    chat_id: str
    content: str
    reply_to_message_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    media: list[str] = field(default_factory=list)
