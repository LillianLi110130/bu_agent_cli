"""Inbound message dispatcher for gateway-managed agent sessions."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from bu_agent_sdk.bootstrap.session_bootstrap import sync_workspace_agents_md
from bu_agent_sdk.bus.events import InboundMessage, OutboundMessage

if TYPE_CHECKING:
    from bu_agent_sdk.bus.queue import MessageBus
    from bu_agent_sdk.runtime.manager import RuntimeManager

logger = logging.getLogger("bu_agent_sdk.gateway.dispatcher")

_DEFAULT_ERROR_MESSAGE = "Sorry, I encountered an error while processing your message."
_START_MESSAGE = (
    "Hello! I'm BU Agent.\n\n"
    "Use /new to start a fresh conversation.\n"
    "Use /help to see available commands."
)
_HELP_MESSAGE = (
    "Available commands:\n" "/new - Start a new conversation\n" "/help - Show this help message"
)
_NEW_SESSION_MESSAGE = "New session started."


class GatewayDispatcher:
    """Route inbound messages to session runtimes and publish outbound replies."""

    def __init__(
        self,
        bus: "MessageBus | None",
        runtime_manager: "RuntimeManager",
        error_message: str = _DEFAULT_ERROR_MESSAGE,
    ) -> None:
        self.bus = bus
        self.runtime_manager = runtime_manager
        self.error_message = error_message
        self.last_active_private_chat: tuple[str, str] | None = None
        self._running = False

    async def dispatch(self, message: InboundMessage) -> OutboundMessage | None:
        """Dispatch a single inbound message and return its outbound response."""
        if message.origin == "user":
            self.last_active_private_chat = (message.channel, message.chat_id)

        command = message.content.strip().lower()
        if command == "/start":
            return self._build_outbound(message, _START_MESSAGE)
        if command == "/help":
            return self._build_outbound(message, _HELP_MESSAGE)
        if command == "/new":
            await self.runtime_manager.clear_runtime(message.session_key)
            return self._build_outbound(message, _NEW_SESSION_MESSAGE)

        try:
            runtime = await self.runtime_manager.get_or_create_runtime(message.session_key)
            async with runtime.lock:
                runtime.touch()
                runtime.workspace_instruction_state = sync_workspace_agents_md(
                    agent=runtime.agent,
                    workspace_dir=runtime.context.working_dir,
                    state=runtime.workspace_instruction_state,
                )
                response_text = await runtime.agent.query(message.content)
            return self._build_outbound(message, response_text)
        except Exception as exc:
            logger.error(f"Failed to dispatch message for session {message.session_key}: {exc}")
            return self._build_outbound(message, self.error_message)

    async def dispatch_once(self) -> None:
        """Consume one inbound message from the bus and publish its response."""
        if self.bus is None:
            raise RuntimeError("GatewayDispatcher requires a bus to run its dispatch loop")

        inbound = await self.bus.consume_inbound()
        outbound = await self.dispatch(inbound)
        if outbound is not None:
            await self.bus.publish_outbound(outbound)

    async def run(self) -> None:
        """Run the continuous dispatch loop until stopped or cancelled."""
        self._running = True
        while self._running:
            try:
                await self.dispatch_once()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"Gateway dispatcher loop error: {exc}")

    def stop(self) -> None:
        """Stop the continuous dispatch loop."""
        self._running = False

    @staticmethod
    def _build_outbound(message: InboundMessage, content: str) -> OutboundMessage:
        reply_to_message_id = message.metadata.get("message_id")
        if not isinstance(reply_to_message_id, int):
            reply_to_message_id = None
        return OutboundMessage(
            channel=message.channel,
            chat_id=message.chat_id,
            content=content,
            reply_to_message_id=reply_to_message_id,
        )
