"""Lifecycle wrapper around the gateway dispatcher loop."""

from __future__ import annotations

import asyncio

from bu_agent_sdk.gateway.dispatcher import GatewayDispatcher


class GatewayService:
    """Manage the background task that runs the gateway dispatcher."""

    def __init__(self, dispatcher: GatewayDispatcher) -> None:
        self.dispatcher = dispatcher
        self._dispatch_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the dispatcher loop if it is not already running."""
        if self._dispatch_task is not None and not self._dispatch_task.done():
            return
        self._dispatch_task = asyncio.create_task(self.dispatcher.run())

    async def stop(self) -> None:
        """Stop the dispatcher loop and wait for the background task to finish."""
        self.dispatcher.stop()
        if self._dispatch_task is None:
            return
        self._dispatch_task.cancel()
        try:
            await self._dispatch_task
        except asyncio.CancelledError:
            pass
        self._dispatch_task = None
