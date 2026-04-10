"""HTTP client for the gateway worker API."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

logger = logging.getLogger("cli.worker.gateway_client")


@dataclass(slots=True)
class WorkerMessage:
    """One message returned by the gateway poll API."""

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
        trust_env = not _is_loopback_base_url(self.base_url)
        self._client = client or httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(connect=5.0, read=35.0, write=10.0, pool=5.0),
            trust_env=trust_env,
        )

    async def poll(self, worker_id: str) -> list[WorkerMessage]:
        """Long poll for at most one message for the bound worker."""
        logger.info(f"Polling gateway for worker_id={worker_id}")
        response = await self._client.post(
            "/api/worker/poll",
            json={
                "worker_id": worker_id,
            },
            headers=self._build_headers(),
        )
        response.raise_for_status()
        payload = response.json()
        messages = payload.get("messages", [])
        return [WorkerMessage(content=str(message["content"])) for message in messages]

    async def complete(
        self,
        worker_id: str,
        final_content: str,
    ) -> bool:
        """Submit the final response for one delivery."""
        logger.info(f"Completing delivery for worker_id={worker_id}")
        response = await self._client.post(
            "/api/worker/complete",
            json={
                "worker_id": worker_id,
                "final_content": final_content,
            },
            headers=self._build_headers(),
        )
        response.raise_for_status()
        return bool(response.json().get("ok", False))

    async def online(self, worker_id: str) -> bool:
        """Mark the worker as online."""
        # todo: 改成通知实际的server端
        response = await self._client.post(
            "/api/worker/online",
            json={
                "worker_id": worker_id,
            },
            headers=self._build_headers(),
        )
        response.raise_for_status()
        return bool(response.json().get("ok", False))

    async def send_text(self, worker_id: str, text: str) -> bool:
        """Send one proactive text message through the gateway."""
        logger.info(f"Sending proactive text for worker_id={worker_id}")
        response = await self._client.post(
            "/api/worker/send_text",
            json={
                "worker_id": worker_id,
                "text": text,
            },
            headers=self._build_headers(),
        )
        response.raise_for_status()
        return bool(response.json().get("ok", False))

    async def upload_attachment(
        self,
        *,
        worker_id: str,
        file_name: str,
        mime_type: str,
        file_size: int,
        file_bytes: bytes,
    ) -> bool:
        """Upload one proactive attachment through the gateway."""
        logger.info(f"Uploading proactive attachment for worker_id={worker_id}, file_name={file_name}")
        response = await self._client.post(
            "/api/worker/upload_attachment",
            data={
                "worker_id": worker_id,
                "mime_type": mime_type,
                "file_size": file_size,
            },
            files={
                "file": (file_name, file_bytes, mime_type),
            },
            headers=self._build_headers(),
        )
        response.raise_for_status()
        return bool(response.json().get("ok", False))

    async def offline(self, worker_id: str) -> bool:
        """Mark the worker as offline."""
        response = await self._client.post(
            "/api/worker/offline",
            json={
                "worker_id": worker_id,
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


def _is_loopback_base_url(base_url: str) -> bool:
    """Return True when *base_url* targets the local machine."""
    hostname = (urlparse(base_url).hostname or "").strip().lower()
    return hostname in {"127.0.0.1", "localhost", "::1"}
