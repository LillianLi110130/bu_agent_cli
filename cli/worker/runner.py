"""Main worker loop for bridging gateway deliveries into the local file queue."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import os
from pathlib import Path
from typing import Any

import httpx

from agent_core import Agent
from agent_core.bootstrap.agent_factory import create_agent
from cli.im_bridge import SqliteBridgeStore, resolve_session_binding_id
from config.model_config import ModelPreset, load_model_presets
from cron.jobs import CronJobStore
from cron.models import CronHostContext, CronJob
from cron.scheduler import CronScheduler
from tools.sandbox import get_current_agent

logger = logging.getLogger("cli.worker.runner")


class WorkerRunner:
    """Run the minimal worker lifecycle for one fixed worker binding."""

    def __init__(
        self,
        worker_id: str,
        gateway_client: Any,
        model: str | None,
        root_dir: str | Path | None,
        worker_no: str | None = None,
        parent_pid: int | None = None,
        gateway_transport: str = "sse",
        stream_max_session_seconds: float = 20 * 60,
        result_poll_interval_seconds: float = 0.5,
        empty_poll_sleep_seconds: float = 0.1,
        cron_tick_interval_seconds: float = 60.0,
        agent: Agent | None = None,
    ) -> None:
        self.worker_id = worker_id
        self.worker_no = worker_no or worker_id
        self.parent_pid = parent_pid
        self.gateway_client = gateway_client
        self.model = model
        self.root_dir = root_dir
        self.gateway_transport = gateway_transport
        self.stream_max_session_seconds = stream_max_session_seconds
        self.result_poll_interval_seconds = result_poll_interval_seconds
        self.empty_poll_sleep_seconds = empty_poll_sleep_seconds
        self.cron_tick_interval_seconds = cron_tick_interval_seconds
        self._main_agent = agent
        resolved_root_dir = (
            Path(root_dir).resolve() if root_dir is not None else Path.cwd().resolve()
        )
        self.root_path = resolved_root_dir
        self.bridge_store = SqliteBridgeStore(
            root_dir=resolved_root_dir,
            session_binding_id=resolve_session_binding_id(self.worker_no),
        )
        self.bridge_store.initialize()
        self.cron_scheduler = CronScheduler(store=CronJobStore())
        self._cron_next_tick_at: float | None = None
        self._stop_event = asyncio.Event()
        self._completion_tasks: set[asyncio.Task[Any]] = set()
        self._worker_touch_task: asyncio.Task[Any] | None = None
        self._parent_watch_task: asyncio.Task[Any] | None = None

    def stop(self) -> None:
        """Request the worker loop to stop."""
        self._stop_event.set()

    async def run_forever(self) -> None:
        """Run the worker loop until explicitly stopped."""
        if not await self._call_gateway("online", worker_id=self.worker_id, worker_no=self.worker_no):
            raise RuntimeError(f"Failed to mark worker online for worker_id={self.worker_id}")
        self.bridge_store.touch_worker(force=True)
        self._worker_touch_task = asyncio.create_task(self._touch_worker_loop())
        if self.parent_pid is not None:
            self._parent_watch_task = asyncio.create_task(self._watch_parent_loop())

        try:
            if self.gateway_transport == "sse":
                await self._run_stream_loop()
            else:
                await self._run_poll_loop()
        finally:
            self._stop_event.set()
            if self._worker_touch_task is not None:
                await self._worker_touch_task
            if self._parent_watch_task is not None:
                await self._parent_watch_task
            if self._completion_tasks:
                await asyncio.gather(*list(self._completion_tasks), return_exceptions=True)
            await self.cron_scheduler.wait_background_tasks()
            await self._call_gateway("offline", worker_id=self.worker_id, worker_no=self.worker_no)

    async def _touch_worker_loop(self) -> None:
        """Refresh local worker activity while an idle SSE connection is open."""
        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self.bridge_store.WORKER_TOUCH_INTERVAL_SECONDS,
                )
            except asyncio.TimeoutError:
                try:
                    self.bridge_store.touch_worker(force=True)
                except Exception:
                    logger.exception("Failed to refresh local bridge worker activity")

    async def _watch_parent_loop(self) -> None:
        """Stop the worker when the parent CLI process is no longer alive."""
        if self.parent_pid is None:
            return

        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=5.0)
                return
            except asyncio.TimeoutError:
                pass

            if self._is_process_alive(self.parent_pid):
                continue

            logger.warning(
                "Detected dead parent CLI process, stopping worker. worker_id=%s parent_pid=%s",
                self.worker_id,
                self.parent_pid,
            )
            self._stop_event.set()
            return

    def _is_process_alive(self, pid: int) -> bool:
        """Return True when *pid* still refers to a running process."""
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    def _gateway_kwargs(self, method_name: str, **kwargs: Any) -> dict[str, Any]:
        """Keep old test doubles working while passing worker_no to real clients."""
        method = getattr(self.gateway_client, method_name)
        if "worker_no" not in inspect.signature(method).parameters:
            kwargs.pop("worker_no", None)
        return kwargs

    async def _call_gateway(self, method_name: str, **kwargs: Any) -> Any:
        method = getattr(self.gateway_client, method_name)
        return await method(**self._gateway_kwargs(method_name, **kwargs))

    def _call_gateway_iter(self, method_name: str, **kwargs: Any) -> Any:
        method = getattr(self.gateway_client, method_name)
        return method(**self._gateway_kwargs(method_name, **kwargs))

    async def _run_poll_loop(self) -> None:
        """Consume remote messages via long polling."""
        while not self._stop_event.is_set():
            await self._maybe_tick_cron()
            try:
                messages = await self._call_gateway(
                    "poll",
                    worker_id=self.worker_id,
                    worker_no=self.worker_no,
                )
            except httpx.ReadTimeout as exc:
                logger.warning(
                    "Worker poll timed out for "
                    f"worker_id={self.worker_id}, continuing next poll: {exc}"
                )
                continue
            if not messages:
                self.bridge_store.touch_worker()
                await asyncio.sleep(self.empty_poll_sleep_seconds)
                continue

            self.bridge_store.touch_worker()
            for message in messages:
                self._schedule_message_processing(message)

    async def _run_stream_loop(self) -> None:
        """Consume remote messages via a persistent SSE stream."""
        while not self._stop_event.is_set():
            stream_iter = self._call_gateway_iter(
                "stream_events",
                worker_id=self.worker_id,
                worker_no=self.worker_no,
            )
            stream_session_deadline = self._get_stream_session_deadline()
            reconnect_delay_seconds = self.empty_poll_sleep_seconds
            try:
                while not self._stop_event.is_set():
                    event = await self._next_stream_event(
                        stream_iter=stream_iter,
                        deadline=stream_session_deadline,
                    )
                    self.bridge_store.touch_worker()
                    await self._drain_outbound_events()
                    await self._maybe_tick_cron()
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
        source = self._normalize_remote_source(getattr(message, "source", None))
        request = self.bridge_store.enqueue_text(
            message.content,
            source=source,
            source_meta={
                "worker_id": self.worker_id,
                "worker_no": self.worker_no,
                "origin": source,
            },
            remote_response_required=True,
        )

        try:
            result = await self._wait_for_result(request.request_id, source=source)
        except Exception as exc:
            logger.exception(
                f"Worker request processing failed for worker_id={self.worker_id}: {exc}"
            )
            await self._report_request_processing_error(exc, source=source)
            return
        else:
            ok = await self._complete_bridge_result(result, source=source)
        if not ok:
            logger.warning(f"Worker complete returned ok=false for worker_id={self.worker_id}")

    async def _complete_bridge_result(self, result: Any, *, source: str) -> bool:
        """Report a bridge result back to the gateway complete endpoint."""
        if result.final_status == "failed":
            return await self._call_gateway(
                "complete",
                worker_id=self.worker_id,
                worker_no=self.worker_no,
                final_content=result.final_content,
                source=source,
                final_status="failed",
                error_code=result.error_code,
                error_message=result.error_message,
            )
        return await self._call_gateway(
            "complete",
            worker_id=self.worker_id,
            worker_no=self.worker_no,
            final_content=result.final_content,
            source=source,
        )

    async def _report_request_processing_error(self, exc: Exception, *, source: str) -> None:
        """Best-effort complete-call report for failures before a bridge result exists."""
        error_message = str(exc) or exc.__class__.__name__
        try:
            ok = await self._call_gateway(
                "complete",
                worker_id=self.worker_id,
                worker_no=self.worker_no,
                final_content=f"Execution failed: {error_message}",
                source=source,
                final_status="failed",
                error_code="WORKER_REQUEST_PROCESSING_ERROR",
                error_message=error_message,
            )
        except Exception as report_exc:
            logger.exception(
                "Failed to report worker request processing error via complete for "
                f"worker_id={self.worker_id}: {report_exc}"
            )
            return
        if not ok:
            logger.warning(
                "Worker request error complete returned ok=false for " f"worker_id={self.worker_id}"
            )

    async def _wait_for_result(self, request_id: str, *, source: str):
        """Wait until the local CLI writes a result for *request_id*."""
        sent_progress_ids: set[str] = set()
        while not self._stop_event.is_set():
            await self._forward_pending_progress(request_id, sent_progress_ids, source=source)
            result = self.bridge_store.find_result(request_id)
            if result is not None:
                await self._forward_pending_progress(request_id, sent_progress_ids, source=source)
                return result
            await asyncio.sleep(self.result_poll_interval_seconds)
        raise RuntimeError(f"Worker stopped while waiting for bridge result: {request_id}")

    async def _forward_pending_progress(
        self,
        request_id: str,
        sent_progress_ids: set[str],
        *,
        source: str,
    ) -> None:
        """Forward unsent intermediate text responses for *request_id*."""
        for progress in self.bridge_store.list_progress(request_id):
            logger.warning(progress)
            if progress.progress_id in sent_progress_ids:
                continue
            sent_progress_ids.add(progress.progress_id)
            ok = await self._call_gateway(
                "progress",
                worker_id=self.worker_id,
                worker_no=self.worker_no,
                content=progress.content,
                source=source,
            )
            if not ok:
                logger.warning(f"Worker progress returned ok=false for worker_id={self.worker_id}")
                continue
            self.bridge_store.complete_progress(progress)

    def _normalize_remote_source(self, source: str | None) -> str:
        """Return a protocol-safe remote source label."""
        normalized_source = (source or "").strip().lower()
        if normalized_source in {"im", "web"}:
            return normalized_source
        return "im"

    async def _maybe_tick_cron(self) -> None:
        """Run a cron scheduler tick when the worker host interval elapses."""
        loop = asyncio.get_running_loop()
        now = loop.time()
        if self._cron_next_tick_at is None:
            self._cron_next_tick_at = now + self.cron_tick_interval_seconds
            return
        if now < self._cron_next_tick_at:
            return
        self._cron_next_tick_at = now + self.cron_tick_interval_seconds
        try:
            await self.cron_scheduler.tick(host_context=self._build_cron_host_context())
        except Exception as exc:
            logger.exception(
                f"Cron scheduler tick failed for worker_id={self.worker_id}: {exc}"
            )

    def _build_cron_host_context(self) -> CronHostContext:
        return CronHostContext(
            source="remote",
            workspace_root=self.root_path,
            session_binding_id=resolve_session_binding_id(self.worker_id),
            worker_id=self.worker_id,
            gateway_client=self.gateway_client,
            default_delivery="remote",
            fresh_agent_runner=self._run_cron_fresh_agent,
        )

    async def _run_cron_fresh_agent(self, job: CronJob) -> str:
        """Execute one scheduled job in a clean worker Agent session."""
        main_agent = self._resolve_main_agent()
        cron_agent = self._build_cron_background_agent(main_agent, job)
        prompt = self._build_cron_background_prompt(job)
        return await cron_agent.query(prompt)

    def _resolve_main_agent(self) -> Agent:
        if self._main_agent is None:
            self._main_agent, _ = create_agent(
                model=self._resolve_main_agent_model(),
                root_dir=self.root_path,
            )
        return self._main_agent

    def _resolve_main_agent_model(self) -> str | None:
        """Resolve the worker main agent model with a config-based fallback."""
        if self.model is not None:
            return self.model
        return self._select_fallback_model(load_model_presets())

    @staticmethod
    def _select_fallback_model(presets: dict[str, ModelPreset]) -> str:
        for keyword in ("Qwen3.6", "minimax"):
            keyword_lower = keyword.lower()
            for name, preset in presets.items():
                preset_model = str(preset.get("model", ""))
                if keyword_lower in name.lower() or keyword_lower in preset_model.lower():
                    return name
        return "small"

    def _build_cron_background_agent(self, main_agent: Agent, job: CronJob) -> Agent:
        dependency_overrides = dict(main_agent.dependency_overrides or {})
        tools = self._cron_background_tools(main_agent, job)
        cron_agent = Agent(
            llm=main_agent.llm,
            tools=tools,
            system_prompt=main_agent.system_prompt,
            max_iterations=main_agent.max_iterations,
            tool_choice=main_agent.tool_choice,
            compaction=main_agent.compaction,
            dependency_overrides=dependency_overrides,
            runtime_role="primary",
            llm_session_role="cron",
            hooks=[],
            use_streaming=main_agent.use_streaming,
        )
        cron_agent.dependency_overrides[get_current_agent] = lambda: cron_agent
        return cron_agent

    @staticmethod
    def _cron_background_tools(main_agent: Agent, job: CronJob) -> list[Any]:
        # 禁止cron_agent使用 定时任务，subagent，agent-team，web相关工具
        blocked_tools = {"cronjob", "delegate", "delegate_parallel"}
        tools = [
            tool
            for tool in main_agent.tools
            if tool.name not in blocked_tools
            and not tool.name.startswith("team_")
            and not tool.name.startswith("task_")
            and tool.name != "web_fetch"
        ]
        if job.enabled_toolsets is None:
            return tools
        allowed_names = set(job.enabled_toolsets)
        return [tool for tool in tools if tool.name in allowed_names]

    @staticmethod
    def _build_cron_background_prompt(job: CronJob) -> str:
        return (
            "You are running an unattended scheduled cron job.\n"
            "Execute the prompt directly. Do not ask follow-up questions. "
            "Do not send outbound messages; return the final result as your response.\n\n"
            f"Job ID: {job.id}\n"
            f"Job Name: {job.name}\n\n"
            "Prompt:\n"
            f"{job.prompt}"
        )

    async def _drain_outbound_events(self) -> None:
        """Deliver queued outbound events through the gateway."""
        while not self._stop_event.is_set():
            event = self.bridge_store.claim_next_pending_outbound()
            if event is None:
                return

            if event.action == "text":
                ok = await self._call_gateway(
                    "send_text",
                    worker_id=self.worker_id,
                    worker_no=self.worker_no,
                    text=event.text,
                )
            elif event.action == "attachment":
                file_bytes = Path(event.file_path).read_bytes()
                ok = await self._call_gateway(
                    "upload_attachment",
                    worker_id=self.worker_id,
                    worker_no=self.worker_no,
                    file_name=event.file_name,
                    mime_type=event.mime_type,
                    file_size=event.file_size,
                    file_bytes=file_bytes,
                )
            else:
                logger.warning(
                    f"Unsupported outbound action={event.action}, event_id={event.event_id}"
                )
                ok = False

            if ok:
                self.bridge_store.complete_outbound_event(event)
