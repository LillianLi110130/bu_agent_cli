"""Local agent task primitives for foreground and background delegated execution."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Iterable

from agent_core.agent import Agent
from agent_core.agent.events import (
    FinalResponseEvent,
    TextEvent,
    ThinkingEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from agent_core.runtime import AgentCallRunner

logger = logging.getLogger("agent_core.task.local_agent_task")

if TYPE_CHECKING:
    from tools.sandbox import SandboxContext


@dataclass(slots=True)
class SubagentCallRequest:
    prompt: str
    description: str
    subagent_type: str | None = None
    model: str | None = None
    run_in_background: bool | None = None


@dataclass(slots=True)
class SubagentTaskResult:
    task_id: str
    subagent_name: str
    prompt: str
    final_response: str
    execution_time_ms: float
    status: str
    description: str
    task_kind: str
    subagent_type: str | None = None
    model: str | None = None
    run_in_background: bool = False
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    error: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "subagent_name": self.subagent_name,
            "prompt": self.prompt,
            "final_response": self.final_response,
            "execution_time_ms": self.execution_time_ms,
            "status": self.status,
            "description": self.description,
            "task_kind": self.task_kind,
            "subagent_type": self.subagent_type,
            "model": self.model,
            "run_in_background": self.run_in_background,
            "tool_calls": self.tool_calls,
            "tools_used": self.tools_used,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


@dataclass(slots=True)
class SubagentTaskStatus:
    task_id: str
    subagent_name: str
    description: str
    task_kind: str
    status: str
    created_at: datetime
    subagent_type: str | None = None
    model: str | None = None
    current_step: str | None = None
    steps_completed: int = 0
    run_in_background: bool = False
    last_activity: datetime | None = None
    recent_logs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "subagent_name": self.subagent_name,
            "description": self.description,
            "task_kind": self.task_kind,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "subagent_type": self.subagent_type,
            "model": self.model,
            "current_step": self.current_step,
            "steps_completed": self.steps_completed,
            "run_in_background": self.run_in_background,
            "last_activity": self.last_activity.isoformat() if self.last_activity else None,
            "recent_logs": list(self.recent_logs),
        }


@dataclass(slots=True)
class SubagentProgressEvent:
    task_id: str
    subagent_name: str
    description: str
    status: str
    current_step: str | None
    steps_completed: int
    run_in_background: bool
    created_at: datetime
    last_activity: datetime | None = None


class SubagentTaskManager:
    """Execute delegated subagent runs in foreground or background."""

    def __init__(
        self,
        *,
        registry: Any,
        all_tools: list[Any],
        context: "SandboxContext",
        skill_registry: Any | None = None,
    ) -> None:
        self._registry = registry
        self._all_tools = list(all_tools)
        self._context = context
        self._skill_registry = skill_registry
        self._runner = AgentCallRunner(
            registry=registry,
            all_tools=all_tools,
            skill_registry=skill_registry,
        )
        self._main_agent: Agent | None = None
        self._running_tasks: dict[str, asyncio.Task[None]] = {}
        self._task_results: dict[str, SubagentTaskResult] = {}
        self._task_statuses: dict[str, SubagentTaskStatus] = {}
        self._completion_events: dict[str, asyncio.Event] = {}

    def set_main_agent(self, agent: Agent) -> None:
        self._main_agent = agent

    async def run_local_agent_task(
        self,
        *,
        parent_agent: Agent,
        request: SubagentCallRequest,
        timeout: float | None = None,
    ) -> str:
        background = self._resolve_background(request)
        if background:
            task_id = await self.start_background_run(parent_agent=parent_agent, request=request)
            return (
                f"Background task [{request.description}] started (id: {task_id}). "
                f"Use `task_status(task_id=\"{task_id}\")` to check status."
            )

        result = await self.run_foreground(
            parent_agent=parent_agent,
            request=request,
            timeout=timeout or 300.0,
        )
        if result.status == "completed":
            return result.final_response or "Task completed but returned empty result."
        if result.status == "cancelled":
            return f"Subagent was cancelled: {result.error or 'Task was cancelled'}"
        return f"Subagent failed: {result.error or 'Unknown error'}"

    async def start_background_run(
        self,
        *,
        parent_agent: Agent,
        request: SubagentCallRequest,
    ) -> str:
        task_id = self._new_task_id()
        normalized_request = self._normalize_request(request)
        status = SubagentTaskStatus(
            task_id=task_id,
            subagent_name=normalized_request.subagent_type or "fork",
            description=normalized_request.description,
            task_kind=self._resolve_task_kind(normalized_request),
            status="running",
            created_at=datetime.now(),
            subagent_type=normalized_request.subagent_type,
            model=normalized_request.model,
            current_step="Starting...",
            run_in_background=True,
        )
        self._task_statuses[task_id] = status
        self._completion_events[task_id] = asyncio.Event()
        await self._emit_progress_event(status, force=True)
        bg_task = asyncio.create_task(
            self._execute_local_agent_run(task_id, parent_agent, normalized_request)
        )
        self._running_tasks[task_id] = bg_task
        bg_task.add_done_callback(lambda _: self._finalize_local_agent_run(task_id))
        return task_id

    async def run_foreground(
        self,
        *,
        parent_agent: Agent,
        request: SubagentCallRequest,
        timeout: float = 300.0,
    ) -> SubagentTaskResult:
        task_id = self._new_task_id()
        normalized_request = self._normalize_request(request)
        logger.info(
            "foreground subagent run start task_id=%s subagent=%s description=%s timeout=%s",
            task_id,
            normalized_request.subagent_type or "fork",
            normalized_request.description,
            timeout,
        )
        self._completion_events[task_id] = asyncio.Event()
        self._task_statuses[task_id] = SubagentTaskStatus(
            task_id=task_id,
            subagent_name=normalized_request.subagent_type or "fork",
            description=normalized_request.description,
            task_kind=self._resolve_task_kind(normalized_request),
            status="running",
            created_at=datetime.now(),
            subagent_type=normalized_request.subagent_type,
            model=normalized_request.model,
            current_step="Starting...",
            run_in_background=False,
        )
        await self._emit_progress_event(self._task_statuses[task_id], force=True)
        task = asyncio.create_task(
            self._execute_local_agent_run(task_id, parent_agent, normalized_request)
        )
        self._running_tasks[task_id] = task
        task.add_done_callback(lambda _: self._finalize_local_agent_run(task_id))

        try:
            await asyncio.wait_for(self._completion_events[task_id].wait(), timeout=timeout)
        except asyncio.CancelledError:
            status = self._task_statuses.get(task_id)
            logger.warning(
                "foreground subagent run cancelled by parent task_id=%s subagent=%s current_step=%s "
                "steps_completed=%s last_activity=%s",
                task_id,
                normalized_request.subagent_type or "fork",
                status.current_step if status is not None else None,
                status.steps_completed if status is not None else None,
                status.last_activity.isoformat() if status and status.last_activity else None,
            )
            await self.cancel_run(task_id)
            raise
        except asyncio.TimeoutError:
            status = self._task_statuses.get(task_id)
            logger.error(
                "foreground subagent run timed out task_id=%s subagent=%s current_step=%s "
                "steps_completed=%s last_activity=%s",
                task_id,
                normalized_request.subagent_type or "fork",
                status.current_step if status is not None else None,
                status.steps_completed if status is not None else None,
                status.last_activity.isoformat() if status and status.last_activity else None,
            )
            await self.cancel_run(task_id)
            raise

        logger.info(
            "foreground subagent run completed task_id=%s status=%s",
            task_id,
            self._task_results[task_id].status,
        )
        return self._task_results[task_id]

    async def run_parallel_foreground(
        self,
        *,
        parent_agent: Agent,
        requests: Iterable[SubagentCallRequest],
        timeout: float = 300.0,
    ) -> list[SubagentTaskResult]:
        return await asyncio.gather(
            *[
                self.run_foreground(parent_agent=parent_agent, request=request, timeout=timeout)
                for request in requests
            ]
        )

    def get_run_status(self, task_id: str) -> str | None:
        status = self._task_statuses.get(task_id)
        if status is not None:
            return json.dumps(status.to_dict(), ensure_ascii=False, indent=2)
        result = self._task_results.get(task_id)
        if result is not None:
            return json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
        return None

    def list_all_runs(self) -> str:
        payload = {
            "running": [
                status.to_dict()
                for status in self._task_statuses.values()
                if status.status == "running"
            ],
            "finished": [result.to_dict() for result in self._task_results.values()],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def list_live_runs(self) -> list[SubagentTaskStatus]:
        """Return live status objects for interactive dashboard rendering."""
        return sorted(
            self._task_statuses.values(),
            key=lambda status: status.last_activity or status.created_at,
            reverse=True,
        )

    async def cancel_run(self, task_id: str) -> str:
        task = self._running_tasks.get(task_id)
        if task is None:
            return f"Error: Task '{task_id}' not found or already completed"
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return f"Task '{task_id}' cancellation requested"

    async def shutdown(self, *, cancel_running: bool = True) -> None:
        """Best-effort shutdown for delegated subagent tasks."""
        if not cancel_running:
            return

        running_task_ids = list(self._running_tasks.keys())
        for task_id in running_task_ids:
            try:
                await self.cancel_run(task_id)
            except Exception:
                logger.exception("Failed to cancel subagent task during shutdown: %s", task_id)

    async def _execute_local_agent_run(
        self,
        task_id: str,
        parent_agent: Agent,
        request: SubagentCallRequest,
    ) -> None:
        status = self._task_statuses[task_id]
        started = time.time()
        tool_calls: list[dict[str, Any]] = []
        tools_used: set[str] = set()
        final_response = ""
        first_event_seen = False

        try:
            status.current_step = "Building child agent..."
            logger.info(
                "subagent child build start task_id=%s subagent=%s task_kind=%s background=%s",
                task_id,
                request.subagent_type or "fork",
                self._resolve_task_kind(request),
                self._resolve_background(request),
            )
            child_agent, initial_message = self._runner.build_execution(
                parent_agent=parent_agent,
                request=request,
            )
            status.model = getattr(child_agent.llm, "model", request.model)
            status.current_step = "Child agent built; waiting for first event..."
            await self._emit_progress_event(status, force=True)
            logger.info(
                "subagent child build complete task_id=%s subagent=%s model=%s prompt_len=%s",
                task_id,
                request.subagent_type or "fork",
                status.model,
                len(initial_message),
            )
            async for event in child_agent.query_stream(initial_message):
                status.last_activity = datetime.now()
                if not first_event_seen:
                    first_event_seen = True
                    logger.info(
                        "subagent first event task_id=%s subagent=%s event_type=%s",
                        task_id,
                        request.subagent_type or "fork",
                        type(event).__name__,
                    )
                if isinstance(event, ToolCallEvent):
                    tool_calls.append({"tool": event.tool, "args": event.args, "timestamp": time.time()})
                    status.current_step = f"Calling {event.tool}..."
                    status.steps_completed += 1
                    self._append_status_log(status, f"Tool: {event.tool}")
                    await self._emit_progress_event(status, force=True)
                elif isinstance(event, ToolResultEvent):
                    tools_used.add(event.tool)
                    status.current_step = f"Result from {event.tool}"
                    result_preview = str(event.result).strip().replace("\n", " ")
                    if len(result_preview) > 120:
                        result_preview = result_preview[:117] + "..."
                    prefix = "Error" if event.is_error else "Result"
                    self._append_status_log(
                        status,
                        f"{prefix}: {event.tool} -> {result_preview or '(empty)'}",
                    )
                elif isinstance(event, ThinkingEvent):
                    status.current_step = f"Thinking: {event.content[:50]}..."
                    preview = event.content.strip().replace("\n", " ")
                    if len(preview) > 120:
                        preview = preview[:117] + "..."
                    if preview:
                        self._append_status_log(status, f"Thinking: {preview}")
                elif isinstance(event, TextEvent):
                    status.current_step = "Generating response..."
                    preview = event.content.strip().replace("\n", " ")
                    if len(preview) > 120:
                        preview = preview[:117] + "..."
                    if preview:
                        self._append_status_log(status, f"Text: {preview}")
                elif isinstance(event, FinalResponseEvent):
                    final_response = event.content
                    preview = event.content.strip().replace("\n", " ")
                    if len(preview) > 120:
                        preview = preview[:117] + "..."
                    self._append_status_log(status, f"Final: {preview or '(empty)'}")
                    await self._emit_progress_event(status, force=True)

            result = SubagentTaskResult(
                task_id=task_id,
                subagent_name=request.subagent_type or "fork",
                prompt=request.prompt,
                final_response=final_response,
                execution_time_ms=(time.time() - started) * 1000,
                status="completed",
                description=request.description,
                task_kind=self._resolve_task_kind(request),
                subagent_type=request.subagent_type,
                model=getattr(child_agent.llm, "model", request.model),
                run_in_background=self._resolve_background(request),
                tool_calls=tool_calls,
                tools_used=sorted(tools_used),
                completed_at=datetime.now(),
            )
            self._task_results[task_id] = result
            status.status = "completed"
            status.current_step = "Completed"
            await self._emit_progress_event(status, force=True)
            logger.info(
                "subagent run completed task_id=%s subagent=%s tool_calls=%s tools_used=%s duration_ms=%.1f",
                task_id,
                request.subagent_type or "fork",
                len(tool_calls),
                sorted(tools_used),
                result.execution_time_ms,
            )
            await self._notify_completion(parent_agent=parent_agent, result=result)
        except asyncio.CancelledError:
            result = SubagentTaskResult(
                task_id=task_id,
                subagent_name=request.subagent_type or "fork",
                prompt=request.prompt,
                final_response="",
                execution_time_ms=(time.time() - started) * 1000,
                status="cancelled",
                description=request.description,
                task_kind=self._resolve_task_kind(request),
                subagent_type=request.subagent_type,
                model=request.model,
                run_in_background=self._resolve_background(request),
                error="Task was cancelled",
                completed_at=datetime.now(),
            )
            self._task_results[task_id] = result
            status.status = "cancelled"
            status.current_step = "Cancelled"
            await self._emit_progress_event(status, force=True)
            logger.warning(
                "subagent run cancelled task_id=%s subagent=%s first_event_seen=%s duration_ms=%.1f",
                task_id,
                request.subagent_type or "fork",
                first_event_seen,
                result.execution_time_ms,
            )
            await self._notify_completion(parent_agent=parent_agent, result=result)
            raise
        except Exception as exc:
            logger.exception("Subagent task failed: %s", task_id)
            result = SubagentTaskResult(
                task_id=task_id,
                subagent_name=request.subagent_type or "fork",
                prompt=request.prompt,
                final_response="",
                execution_time_ms=(time.time() - started) * 1000,
                status="failed",
                description=request.description,
                task_kind=self._resolve_task_kind(request),
                subagent_type=request.subagent_type,
                model=request.model,
                run_in_background=self._resolve_background(request),
                error=str(exc),
                completed_at=datetime.now(),
            )
            self._task_results[task_id] = result
            status.status = "failed"
            status.current_step = "Failed"
            await self._emit_progress_event(status, force=True)
            logger.error(
                "subagent run failed task_id=%s subagent=%s first_event_seen=%s error=%s",
                task_id,
                request.subagent_type or "fork",
                first_event_seen,
                exc,
            )
            await self._notify_completion(parent_agent=parent_agent, result=result)
        finally:
            event = self._completion_events.get(task_id)
            if event is not None:
                logger.info("subagent completion event set task_id=%s", task_id)
                event.set()

    async def _notify_completion(self, *, parent_agent: Agent, result: SubagentTaskResult) -> None:
        main_agent = self._main_agent or parent_agent
        logger.info(
            "dispatching subagent completion task_id=%s status=%s background=%s",
            result.task_id,
            result.status,
            result.run_in_background,
        )
        ui_events = await main_agent._hook_manager.dispatch_subagent_result(main_agent, result)
        if self._context.subagent_events is not None:
            for event in ui_events:
                await self._context.subagent_events.put(event)

    async def _emit_progress_event(
        self,
        status: SubagentTaskStatus,
        *,
        force: bool = False,
    ) -> None:
        if self._context.subagent_events is None:
            return
        if not force:
            if status.current_step and status.current_step.startswith("Thinking:"):
                return
            if status.current_step and status.current_step.startswith("Result from "):
                return
            if status.current_step == "Generating response...":
                return
        await self._context.subagent_events.put(
            SubagentProgressEvent(
                task_id=status.task_id,
                subagent_name=status.subagent_name,
                description=status.description,
                status=status.status,
                current_step=status.current_step,
                steps_completed=status.steps_completed,
                run_in_background=status.run_in_background,
                created_at=status.created_at,
                last_activity=status.last_activity,
            )
        )

    def _normalize_request(self, request: SubagentCallRequest) -> SubagentCallRequest:
        prompt = str(request.prompt).strip()
        description = str(request.description).strip()
        subagent_type = (
            str(request.subagent_type).strip() if request.subagent_type is not None else None
        )
        model = str(request.model).strip() if request.model is not None else None
        if not prompt:
            raise ValueError("prompt must not be empty")
        if not description:
            raise ValueError("description must not be empty")
        return SubagentCallRequest(
            prompt=prompt,
            description=description,
            subagent_type=subagent_type or None,
            model=model or None,
            run_in_background=request.run_in_background,
        )

    def _resolve_background(self, request: SubagentCallRequest) -> bool:
        if request.run_in_background is not None:
            return bool(request.run_in_background)
        if request.subagent_type:
            config = self._registry.get_config(request.subagent_type)
            if config is not None:
                return bool(config.background)
        return False

    @staticmethod
    def _resolve_task_kind(request: SubagentCallRequest) -> str:
        return "named" if request.subagent_type else "fork"

    @staticmethod
    def _new_task_id() -> str:
        return uuid.uuid4().hex[:8]

    def _finalize_local_agent_run(self, task_id: str) -> None:
        self._running_tasks.pop(task_id, None)

    @staticmethod
    def _append_status_log(status: SubagentTaskStatus, line: str, *, limit: int = 8) -> None:
        if not line:
            return
        status.recent_logs.append(line)
        if len(status.recent_logs) > limit:
            del status.recent_logs[:-limit]
