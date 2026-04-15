"""Main worker loop for bridging gateway deliveries into the local file queue."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path
from typing import Any

import httpx

from cli.im_bridge import FileBridgeStore, resolve_session_binding_id

logger = logging.getLogger("cli.worker.runner")


class WorkerRunner:
    """Run the minimal worker lifecycle for one fixed worker binding."""

    def __init__(
        self,
        worker_id: str,
        gateway_client: Any,
        model: str | None,
        root_dir: str | Path | None,
        gateway_transport: str = "sse",
        stream_max_session_seconds: float = 20 * 60,
        result_poll_interval_seconds: float = 0.5,
        empty_poll_sleep_seconds: float = 0.1,
    ) -> None:
        self.worker_id = worker_id
        self.gateway_client = gateway_client
        self.model = model
        self.root_dir = root_dir
        self.gateway_transport = gateway_transport
        self.stream_max_session_seconds = stream_max_session_seconds
        self.result_poll_interval_seconds = result_poll_interval_seconds
        self.empty_poll_sleep_seconds = empty_poll_sleep_seconds
        resolved_root_dir = Path(root_dir).resolve() if root_dir is not None else Path.cwd().resolve()
        self.bridge_store = FileBridgeStore(
            root_dir=resolved_root_dir,
            session_binding_id=resolve_session_binding_id(worker_id),
        )
        self.bridge_store.initialize()
        self._stop_event = asyncio.Event()
        self._completion_tasks: set[asyncio.Task[Any]] = set()

    def stop(self) -> None:
        """Request the worker loop to stop."""
        self._stop_event.set()

    async def run_forever(self) -> None:
        """Run the worker loop until explicitly stopped."""
        if not await self.gateway_client.online(worker_id=self.worker_id):
            raise RuntimeError(
                f"Failed to mark worker online for worker_id={self.worker_id}"
            )

        try:
            if self.gateway_transport == "sse":
                await self._run_stream_loop()
            else:
                await self._run_poll_loop()
        finally:
            self._stop_event.set()
            if self._completion_tasks:
                await asyncio.gather(*list(self._completion_tasks), return_exceptions=True)
            await self.gateway_client.offline(worker_id=self.worker_id)

    async def _run_poll_loop(self) -> None:
        """Consume remote messages via long polling."""
        while not self._stop_event.is_set():
            try:
                messages = await self.gateway_client.poll(worker_id=self.worker_id)
            except httpx.ReadTimeout as exc:
                logger.warning(
                    f"Worker poll timed out for worker_id={self.worker_id}, continuing next poll: {exc}"
                )
                continue
            if not messages:
                await asyncio.sleep(self.empty_poll_sleep_seconds)
                continue

            for message in messages:
                self._schedule_message_processing(message)

    async def _run_stream_loop(self) -> None:
        """Consume remote messages via a persistent SSE stream."""
        while not self._stop_event.is_set():
            stream_iter = self.gateway_client.stream_events(worker_id=self.worker_id)
            stream_session_deadline = self._get_stream_session_deadline()
            reconnect_delay_seconds = self.empty_poll_sleep_seconds
            try:
                while not self._stop_event.is_set():
                    event = await self._next_stream_event(
                        stream_iter=stream_iter,
                        deadline=stream_session_deadline,
                    )
                    if self._stop_event.is_set():
                        break
                    if event.event != "message" or event.message is None:
                        continue
                    self._schedule_message_processing(event.message)
            except StopAsyncIteration:
                logger.info(
                    f"Worker stream closed normally for worker_id={self.worker_id}, reconnecting"
                )
            except asyncio.TimeoutError:
                logger.info(
                    "Worker stream session reached reconnect deadline for "
                    f"worker_id={self.worker_id}, reconnecting"
                )
                reconnect_delay_seconds = 0.0
            except (httpx.HTTPError, httpx.ReadTimeout) as exc:
                logger.warning(
                    f"Worker stream failed for worker_id={self.worker_id}, reconnecting: {exc}"
                )
            finally:
                aclose = getattr(stream_iter, "aclose", None)
                if callable(aclose):
                    with contextlib.suppress(Exception):
                        await aclose()
            if not self._stop_event.is_set():
                await asyncio.sleep(reconnect_delay_seconds)

    def _schedule_message_processing(self, message: Any) -> None:
        """Process one remote message in the background while intake continues."""
        task = asyncio.create_task(self._process_message(message))
        self._completion_tasks.add(task)
        task.add_done_callback(self._completion_tasks.discard)

    def _get_stream_session_deadline(self) -> float | None:
        """Return the deadline for the current SSE session, if enabled."""
        if self.stream_max_session_seconds <= 0:
            return None
        return asyncio.get_running_loop().time() + self.stream_max_session_seconds

    async def _next_stream_event(
        self,
        *,
        stream_iter: Any,
        deadline: float | None,
    ) -> Any:
        """Read the next SSE event, proactively rotating the stream near its deadline."""
        if deadline is None:
            return await anext(stream_iter)

        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise asyncio.TimeoutError
        return await asyncio.wait_for(anext(stream_iter), timeout=remaining)

    async def _process_message(self, message: Any) -> None:
        """Process one polled message."""
        request = self.bridge_store.enqueue_text(
            message.content,
            source="remote",
            source_meta={
                "worker_id": self.worker_id,
            },
            remote_response_required=True,
        )

        try:
            result = await self._wait_for_result(request.request_id)
        except Exception:
            return

        ok = await self.gateway_client.complete(
            worker_id=self.worker_id,
            final_content=result.final_content,
        )
        if not ok:
            logger.warning(
                f"Worker complete returned ok=false for worker_id={self.worker_id}"
            )

    async def _wait_for_result(self, request_id: str):
        """Wait until the local CLI writes a result for *request_id*."""
        while not self._stop_event.is_set():
            result = self.bridge_store.find_result(request_id)
            if result is not None:
                return result
            await asyncio.sleep(self.result_poll_interval_seconds)
        raise RuntimeError(f"Worker stopped while waiting for bridge result: {request_id}")
