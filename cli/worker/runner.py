"""Main worker loop for polling, renewing, and completing messages."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import httpx

from bu_agent_sdk.bootstrap.session_bootstrap import (
    WorkspaceInstructionState,
    sync_workspace_agents_md,
)
from cli.worker.runtime_factory import create_worker_runtime

logger = logging.getLogger("cli.worker.runner")


class WorkerRunner:
    """Run the minimal worker lifecycle for one fixed session key."""

    def __init__(
        self,
        session_key: str,
        worker_id: str,
        gateway_client: Any,
        model: str | None,
        root_dir: str | Path | None,
        renew_interval_seconds: float = 30.0,
    ) -> None:
        self.session_key = session_key
        self.worker_id = worker_id
        self.gateway_client = gateway_client
        self.model = model
        self.root_dir = root_dir
        self.renew_interval_seconds = renew_interval_seconds
        self.current_epoch: int | None = None
        self.agent: Any | None = None
        self.context: Any | None = None
        self._workspace_instruction_state = WorkspaceInstructionState()
        self._renew_task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()

    def stop(self) -> None:
        """Request the worker loop to stop."""
        self._stop_event.set()

    async def run_forever(self) -> None:
        """Run the worker loop until explicitly stopped."""
        while not self._stop_event.is_set():
            try:
                messages = await self.gateway_client.poll(
                    session_key=self.session_key,
                    worker_id=self.worker_id,
                )
            except httpx.ReadTimeout as exc:
                logger.warning(
                    f"Worker poll timed out for session_key={self.session_key} worker_id={self.worker_id}, continuing next poll: {exc}"
                )
                continue
            if not messages:
                continue

            message = messages[0]
            await self._process_message(message)

    async def _process_message(self, message: Any) -> None:
        """Process one polled message."""
        await self._ensure_runtime(message.epoch)
        if self.agent is None or self.context is None:
            raise RuntimeError("Worker runtime is not initialized")

        self._workspace_instruction_state = sync_workspace_agents_md(
            agent=self.agent,
            workspace_dir=self.context.working_dir,
            state=self._workspace_instruction_state,
        )
        self._renew_task = asyncio.create_task(self._run_renew_loop(message.delivery_id))

        try:
            final_content = await self.agent.query(message.content)
        except Exception as exc:
            logger.exception(
                f"Worker query failed for session_key={self.session_key} delivery_id={message.delivery_id}: {exc}"
            )
            await self._stop_renew_task()
            return

        await self._stop_renew_task()
        ok = await self.gateway_client.complete(
            session_key=self.session_key,
            worker_id=self.worker_id,
            delivery_id=message.delivery_id,
            final_content=final_content,
        )
        if not ok:
            logger.warning(
                f"Worker complete returned ok=false for session_key={self.session_key} delivery_id={message.delivery_id}"
            )

    async def _ensure_runtime(self, epoch: int) -> None:
        """Create or rebuild the local runtime when epoch changes."""
        if self.agent is not None and self.context is not None and self.current_epoch == epoch:
            return

        self.agent, self.context = create_worker_runtime(model=self.model, root_dir=self.root_dir)
        self.current_epoch = epoch
        self._workspace_instruction_state = WorkspaceInstructionState()
        logger.info(
            f"Initialized worker runtime for session_key={self.session_key} epoch={self.current_epoch}"
        )

    async def _run_renew_loop(self, delivery_id: str) -> None:
        """Renew the current delivery until cancelled."""
        try:
            while not self._stop_event.is_set():
                await asyncio.sleep(self.renew_interval_seconds)
                ok = await self.gateway_client.renew(
                    session_key=self.session_key,
                    worker_id=self.worker_id,
                    delivery_id=delivery_id,
                )
                if not ok:
                    logger.warning(
                        f"Worker renew returned ok=false for session_key={self.session_key} delivery_id={delivery_id}"
                    )
        except asyncio.CancelledError:
            logger.debug(
                f"Stopped renew loop for session_key={self.session_key} delivery_id={delivery_id}"
            )
            raise

    async def _stop_renew_task(self) -> None:
        """Cancel and await the current renew task if present."""
        if self._renew_task is None:
            return
        self._renew_task.cancel()
        try:
            await self._renew_task
        except asyncio.CancelledError:
            pass
        self._renew_task = None
