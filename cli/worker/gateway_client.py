"""HTTP client for the gateway worker API."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

logger = logging.getLogger("cli.worker.gateway_client")


@dataclass(slots=True)
class WorkerMessage:
    """One message returned by the gateway poll API."""

    message_id: int
    delivery_id: str
    epoch: int
    content: str


class WorkerGatewayClient:
    """Minimal async client for the worker HTTP protocol."""

    def __init__(
        self,
        base_url: str,
        client: httpx.AsyncClient | None = None,
        authorization: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.authorization = authorization
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(connect=5.0, read=35.0, write=10.0, pool=5.0),
        )

    async def poll(self, session_key: str, worker_id: str) -> list[WorkerMessage]:
        """Long poll for at most one message for the bound session."""
        logger.info(f"Polling gateway for session_key={session_key} worker_id={worker_id}")
        response = await self._client.post(
            "/api/worker/poll",
            json={
                "session_key": session_key,
                "worker_id": worker_id,
            },
            headers=self._build_headers(),
        )
        response.raise_for_status()
        payload = response.json()
        messages = payload.get("messages", [])
        return [
            WorkerMessage(
                message_id=int(message["message_id"]),
                delivery_id=str(message["delivery_id"]),
                epoch=int(message["epoch"]),
                content=str(message["content"]),
            )
            for message in messages
        ]

    async def renew(self, session_key: str, worker_id: str, delivery_id: str) -> bool:
        """Renew the current in-flight delivery lease."""
        logger.info(
            f"Renewing delivery for session_key={session_key} worker_id={worker_id} delivery_id={delivery_id}"
        )
        response = await self._client.post(
            "/api/worker/renew",
            json={
                "session_key": session_key,
                "worker_id": worker_id,
                "delivery_id": delivery_id,
            },
            headers=self._build_headers(),
        )
        response.raise_for_status()
        return bool(response.json().get("ok", False))

    async def complete(
        self,
        session_key: str,
        worker_id: str,
        delivery_id: str,
        final_content: str,
    ) -> bool:
        """Submit the final response for one delivery."""
        logger.info(
            f"Completing delivery for session_key={session_key} worker_id={worker_id} delivery_id={delivery_id}"
        )
        response = await self._client.post(
            "/api/worker/complete",
            json={
                "session_key": session_key,
                "worker_id": worker_id,
                "delivery_id": delivery_id,
                "final_content": final_content,
            },
            headers=self._build_headers(),
        )
        response.raise_for_status()
        return bool(response.json().get("ok", False))

    def _build_headers(self) -> dict[str, str] | None:
        """Build optional request headers for worker protocol calls."""
        if not self.authorization:
            return None
        return {"Authorization": self.authorization}

    async def aclose(self) -> None:
        """Close the underlying HTTP client when owned by this instance."""
        if not self._owns_client:
            return
        await self._client.aclose()
