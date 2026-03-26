"""Async message bus primitives for IM gateway integrations."""

from agent_core.bus.events import InboundMessage, OutboundMessage
from agent_core.bus.queue import MessageBus

__all__ = ["InboundMessage", "OutboundMessage", "MessageBus"]
