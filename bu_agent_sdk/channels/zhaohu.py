"""Zhaohu webhook channel skeleton using FastAPI."""

from __future__ import annotations

import logging
from typing import Any

import uvicorn
from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from bu_agent_sdk.bus.events import InboundMessage, OutboundMessage
from bu_agent_sdk.bus.queue import MessageBus
from bu_agent_sdk.channels.base import BaseChannel

logger = logging.getLogger("bu_agent_sdk.channels.zhaohu")


class ZhaohuWebhookRequest(BaseModel):
    """Minimal normalized webhook payload accepted by the Zhaohu skeleton."""

    sender_id: str
    chat_id: str
    content: str
    message_id: str | None = None
    session_key: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ZhaohuChannel(BaseChannel):
    """Webhook-style Zhaohu channel with a minimal FastAPI server skeleton."""

    name = "zhaohu"
    display_name = "Zhaohu"

    def __init__(self, config: Any, bus: MessageBus):
        super().__init__(config, bus)
        self.host = getattr(config, "host", "0.0.0.0")
        self.port = int(getattr(config, "port", 18080))
        self.webhook_path = getattr(config, "webhook_path", "/webhook/zhaohu")
        if not self.webhook_path.startswith("/"):
            self.webhook_path = f"/{self.webhook_path}"
        self._app: FastAPI | None = None
        self._server: uvicorn.Server | None = None

    async def start(self) -> None:
        """Start the embedded FastAPI server for receiving webhooks."""
        if self._running:
            logger.warning("Zhaohu channel is already running")
            return

        self._running = True
        self._app = self._create_app()
        logger.info(f"Starting Zhaohu webhook server on {self.host}:{self.port}{self.webhook_path}")
        config = uvicorn.Config(
            app=self._app,
            host=self.host,
            port=self.port,
            log_level="info",
        )
        self._server = uvicorn.Server(config)
        try:
            await self._server.serve()
        finally:
            self._running = False

    async def stop(self) -> None:
        """Request the embedded FastAPI server to stop."""
        self._running = False
        if self._server is None:
            return

        logger.info("Stopping Zhaohu webhook server")
        self._server.should_exit = True
        self._server.force_exit = False

    async def send(self, msg: OutboundMessage) -> None:
        """Placeholder outbound sender for future Zhaohu integrations."""
        logger.info(f"Zhaohu outbound send is not implemented yet for chat_id={msg.chat_id}")

    def _build_inbound_message(self, payload: ZhaohuWebhookRequest) -> InboundMessage:
        """Normalize a webhook payload into the shared inbound bus event."""
        metadata = dict(payload.metadata)
        if payload.message_id is not None:
            metadata["message_id"] = payload.message_id
        metadata["raw_payload"] = payload.model_dump()
        return InboundMessage(
            channel=self.name,
            sender_id=payload.sender_id,
            chat_id=payload.chat_id,
            content=payload.content,
            metadata=metadata,
            session_key_override=payload.session_key,
        )

    async def _publish_webhook_payload(self, payload: ZhaohuWebhookRequest) -> InboundMessage:
        """Publish the normalized webhook payload to the shared message bus."""
        inbound = self._build_inbound_message(payload)
        await self.bus.publish_inbound(inbound)
        return inbound

    def _create_app(self) -> FastAPI:
        """Build the minimal FastAPI app for health checks and webhook intake skeleton."""
        app = FastAPI(
            title="Zhaohu Channel",
            docs_url=None,
            redoc_url=None,
            openapi_url=None,
        )

        @app.get("/healthz")
        async def healthz() -> dict[str, str]:
            return {
                "channel": self.name,
                "status": "ok",
                "webhook_path": self.webhook_path,
            }

        @app.post(self.webhook_path)
        async def webhook(payload: ZhaohuWebhookRequest) -> JSONResponse:
            logger.info(f"Received Zhaohu webhook payload on {self.webhook_path}")
            inbound = await self._publish_webhook_payload(payload)
            return JSONResponse(
                status_code=status.HTTP_202_ACCEPTED,
                content={
                    "channel": self.name,
                    "status": "accepted",
                    "message": "Zhaohu webhook payload accepted and queued.",
                    "session_key": inbound.session_key,
                },
            )

        return app
