"""Channel manager for starting integrations and dispatching outbound messages."""

from __future__ import annotations

import asyncio
import logging

from agent_core.bus.queue import MessageBus
from agent_core.channels.base import BaseChannel

logger = logging.getLogger("agent_core.channels.manager")


class ChannelManager:
    """Manage channel lifecycles and outbound message dispatch."""

    def __init__(self, bus: MessageBus):
        self.bus = bus
        self.channels: dict[str, BaseChannel] = {}
        self._channel_tasks: dict[str, asyncio.Task] = {}
        self._dispatch_task: asyncio.Task | None = None

    def register(self, channel: BaseChannel) -> None:
        """Register a channel instance by its name."""
        self.channels[channel.name] = channel

    async def start_all(self) -> None:
        """Start all registered channels and the outbound dispatcher."""
        if self._dispatch_task is None or self._dispatch_task.done():
            self._dispatch_task = asyncio.create_task(self._dispatch_outbound())

        for name, channel in self.channels.items():
            task = self._channel_tasks.get(name)
            if task is not None and not task.done():
                continue
            self._channel_tasks[name] = asyncio.create_task(channel.start())

    async def stop_all(self) -> None:
        """Stop all channels and the outbound dispatcher."""
        if self._dispatch_task is not None:
            self._dispatch_task.cancel()
            try:
                await self._dispatch_task
            except asyncio.CancelledError:
                pass
            self._dispatch_task = None

        for name, channel in self.channels.items():
            try:
                await channel.stop()
            except Exception as exc:
                logger.error(f"Failed to stop channel {name}: {exc}")

        for name, task in list(self._channel_tasks.items()):
            if task.done():
                continue
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.error(f"Channel task for {name} exited with error: {exc}")
        self._channel_tasks.clear()

    async def _dispatch_outbound(self) -> None:
        """Continuously route outbound messages to their target channels."""
        while True:
            try:
                message = await self.bus.consume_outbound()
                channel = self.channels.get(message.channel)
                if channel is None:
                    logger.warning(f"Unknown outbound channel: {message.channel}")
                    continue
                await channel.send(message)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"Outbound dispatch error: {exc}")

    @property
    def enabled_channels(self) -> list[str]:
        """Return the names of registered channels."""
        return list(self.channels.keys())
