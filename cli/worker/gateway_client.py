"""HTTP client for the gateway worker API."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator
from urllib.parse import urlparse

import httpx

from cli.worker.auth import persist_updated_authorization

logger = logging.getLogger("cli.worker.gateway_client")


@dataclass(slots=True)
class WorkerMessage:
    """One message returned by the gateway poll API."""

    content: str


@dataclass(slots=True)
class WorkerStreamEvent:
    """One SSE event emitted by the gateway worker stream."""

    event: str
    message: WorkerMessage | None = None
    payload: dict[str, Any] | None = None


class WorkerGatewayClient:
    """Minimal async client for the worker HTTP protocol."""

    def __init__(
        self,
        base_url: str,
        client: httpx.AsyncClient | None = None,
        authorization: str | None = None,
        base_dir: str | Path | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.authorization = authorization
        self.base_dir = Path(base_dir).resolve() if base_dir is not None else None
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
        response = await self._post_with_auth_refresh(
            "/api/worker/poll",
            {
                "worker_id": worker_id,
            },
        )
        payload = response.json()
        messages = payload.get("messages", [])
        return [WorkerMessage(content=str(message["content"])) for message in messages]

    async def stream_events(self, worker_id: str) -> AsyncIterator[WorkerStreamEvent]:
        """Consume SSE events for the bound worker."""
        logger.info(f"Opening gateway SSE stream for worker_id={worker_id}")
        request_authorization = self.authorization

        for attempt in range(2):
            async with self._client.stream(
                "GET",
                "/api/worker/stream",
                params={"worker_id": worker_id},
                headers=self._build_headers(
                    request_authorization if attempt == 0 else None,
                ),
            ) as response:
                authorization_changed = self._refresh_authorization_from_response(
                    response,
                    request_authorization=request_authorization if attempt == 0 else self.authorization,
                )
                if response.is_error:
                    if attempt == 0 and authorization_changed:
                        logger.warning(
                            "Gateway stream request failed with refreshed Authorization header, "
                            f"retrying once: status={response.status_code}"
                        )
                        request_authorization = self.authorization
                        continue
                    response.raise_for_status()

                async for event in self._iter_stream_events(response):
                    yield event
                return

    async def complete(
        self,
        worker_id: str,
        final_content: str,
    ) -> bool:
        """Submit the final response for one delivery."""
        logger.info(f"Completing delivery for worker_id={worker_id}")
        response = await self._post_with_auth_refresh(
            "/api/worker/complete",
            {
                "worker_id": worker_id,
                "final_content": final_content,
            },
        )
        return bool(response.json().get("ok", False))

    async def online(self, worker_id: str) -> bool:
        """Mark the worker as online."""
        response = await self._post_with_auth_refresh(
            "/api/worker/online",
            {
                "worker_id": worker_id,
            },
        )
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
        response = await self._post_with_auth_refresh(
            "/api/worker/offline",
            {
                "worker_id": worker_id,
            },
        )
        return bool(response.json().get("ok", False))

    def _build_headers(self, authorization: str | None = None) -> dict[str, str] | None:
        """Build optional request headers for worker protocol calls."""
        resolved_authorization = authorization if authorization is not None else self.authorization
        if not resolved_authorization:
            return None
        return {"Authorization": resolved_authorization}

    async def _post_with_auth_refresh(
        self,
        path: str,
        payload: dict[str, object],
    ) -> httpx.Response:
        """POST JSON, refresh local Authorization from response headers, and retry once."""
        request_authorization = self.authorization
        response = await self._client.post(
            path,
            json=payload,
            headers=self._build_headers(request_authorization),
        )
        authorization_changed = self._refresh_authorization_from_response(
            response,
            request_authorization=request_authorization,
        )
        if response.is_error and authorization_changed:
            logger.warning(
                "Gateway request failed with refreshed Authorization header, retrying once: "
                f"path={path}, status={response.status_code}"
            )
            response = await self._client.post(
                path,
                json=payload,
                headers=self._build_headers(),
            )
            self._refresh_authorization_from_response(
                response,
                request_authorization=self.authorization,
            )
        response.raise_for_status()
        return response

    def _refresh_authorization_from_response(
        self,
        response: httpx.Response,
        *,
        request_authorization: str | None,
    ) -> bool:
        """Update in-memory and persisted Authorization when the response provides a new one."""
        response_authorization = _normalize_header_value(response.headers.get("Authorization"))
        if not response_authorization:
            return False
        if response_authorization == self.authorization:
            return False
        if self.authorization is not None and self.authorization != request_authorization:
            logger.info("Skipping stale Authorization refresh because a newer token is already active")
            return False

        self.authorization = response_authorization
        if self.base_dir is not None:
            persist_updated_authorization(
                base_dir=self.base_dir,
                authorization=response_authorization,
            )
        logger.info("Updated worker Authorization from gateway response header")
        return True

    async def aclose(self) -> None:
        """Close the underlying HTTP client when owned by this instance."""
        if not self._owns_client:
            return
        await self._client.aclose()

    async def _iter_stream_events(
        self,
        response: httpx.Response,
    ) -> AsyncIterator[WorkerStreamEvent]:
        """Parse an SSE response into structured events."""
        event_name = "message"
        data_lines: list[str] = []

        async for line in response.aiter_lines():
            if line == "":
                event = _build_stream_event(event_name, data_lines)
                if event is not None:
                    yield event
                event_name = "message"
                data_lines = []
                continue
            if line.startswith(":"):
                continue

            field, separator, value = line.partition(":")
            if not separator:
                continue
            if value.startswith(" "):
                value = value[1:]

            if field == "event":
                event_name = value or "message"
            elif field == "data":
                data_lines.append(value)

        event = _build_stream_event(event_name, data_lines)
        if event is not None:
            yield event


def _is_loopback_base_url(base_url: str) -> bool:
    """Return True when *base_url* targets the local machine."""
    hostname = (urlparse(base_url).hostname or "").strip().lower()
    return hostname in {"127.0.0.1", "localhost", "::1"}


def _normalize_header_value(value: str | None) -> str | None:
    """Normalize an optional response header string."""
    if value is None:
        return None
    normalized_value = value.strip()
    if not normalized_value:
        return None
    return normalized_value


def _build_stream_event(
    event_name: str,
    data_lines: list[str],
) -> WorkerStreamEvent | None:
    """Build one structured stream event from raw SSE lines."""
    payload: dict[str, Any] | None = None
    raw_data = "\n".join(data_lines)
    if raw_data:
        try:
            decoded = json.loads(raw_data)
        except json.JSONDecodeError:
            decoded = {"data": raw_data}
        if isinstance(decoded, dict):
            payload = decoded
        else:
            payload = {"data": decoded}

    if event_name == "message":
        if payload is None or "content" not in payload:
            return None
        return WorkerStreamEvent(
            event="message",
            message=WorkerMessage(content=str(payload["content"])),
            payload=payload,
        )
    return WorkerStreamEvent(
        event=event_name,
        payload=payload,
    )
