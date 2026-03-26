"""Telegram private chat channel integration."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from telegram import Update
from telegram.ext import Application, ContextTypes, MessageHandler, filters
from telegram.request import HTTPXRequest

from agent_core.bus.events import OutboundMessage
from agent_core.bus.queue import MessageBus
from agent_core.channels.base import BaseChannel

logger = logging.getLogger("agent_core.channels.telegram")

TELEGRAM_MAX_MESSAGE_LEN = 4000


def _split_message(text: str, limit: int = TELEGRAM_MAX_MESSAGE_LEN) -> list[str]:
    """Split long Telegram messages into chunks that fit the API limit."""
    if len(text) <= limit:
        return [text]
    return [text[index : index + limit] for index in range(0, len(text), limit)]


class TelegramChannel(BaseChannel):
    """Telegram channel using long polling for private chats only."""

    name = "telegram"
    display_name = "Telegram"

    def __init__(self, config: Any, bus: MessageBus):
        super().__init__(config, bus)
        self._app: Application | None = None
        self._typing_tasks: dict[str, asyncio.Task] = {}

    async def start(self) -> None:
        """Start the Telegram bot in polling mode."""
        token = getattr(self.config, "token", "")
        if not token:
            logger.error("Telegram bot token not configured")
            return

        self._running = True
        request = HTTPXRequest(
            connection_pool_size=16,
            pool_timeout=5.0,
            connect_timeout=30.0,
            read_timeout=30.0,
            proxy=getattr(self.config, "proxy", None),
        )
        builder = Application.builder().token(token).request(request).get_updates_request(request)
        self._app = builder.build()
        self._app.add_handler(MessageHandler(filters.TEXT, self._on_message))

        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(
            allowed_updates=["message"],
            drop_pending_updates=True,
        )

        while self._running:
            await asyncio.sleep(1)

    async def stop(self) -> None:
        """Stop the Telegram bot and cancel typing indicators."""
        self._running = False

        for chat_id in list(self._typing_tasks):
            self._stop_typing(chat_id)

        if self._app is None:
            return

        await self._app.updater.stop()
        await self._app.stop()
        await self._app.shutdown()
        self._app = None

    async def send(self, msg: OutboundMessage) -> None:
        """Send a plain-text Telegram message, splitting if necessary."""
        if self._app is None:
            logger.warning("Telegram bot is not running")
            return

        if not msg.metadata.get("_progress", False):
            self._stop_typing(msg.chat_id)

        chat_id = int(msg.chat_id)
        for chunk in _split_message(msg.content, TELEGRAM_MAX_MESSAGE_LEN):
            await self._app.bot.send_message(chat_id=chat_id, text=chunk)

    async def _on_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Forward private chat messages into the shared bus."""
        del context
        if not update.message or not update.effective_user:
            return

        message = update.message
        user = update.effective_user
        if getattr(message.chat, "type", None) != "private":
            return

        text = message.text or ""
        if not text.strip():
            return

        chat_id = str(message.chat_id)
        self._start_typing(chat_id)
        await self._handle_message(
            sender_id=str(user.id),
            chat_id=chat_id,
            content=text,
            metadata={"message_id": message.message_id},
        )

    def _start_typing(self, chat_id: str) -> None:
        """Start the background typing indicator for one chat."""
        self._stop_typing(chat_id)
        self._typing_tasks[chat_id] = asyncio.create_task(self._typing_loop(chat_id))

    def _stop_typing(self, chat_id: str) -> None:
        """Stop the background typing indicator for one chat."""
        task = self._typing_tasks.pop(chat_id, None)
        if task is not None and not task.done():
            task.cancel()

    async def _typing_loop(self, chat_id: str) -> None:
        """Periodically send typing actions until cancelled."""
        try:
            while self._app is not None:
                await self._app.bot.send_chat_action(chat_id=int(chat_id), action="typing")
                await asyncio.sleep(4)
        except asyncio.CancelledError:
            pass
