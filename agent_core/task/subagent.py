"""Subagent task management primitives."""

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

logger = logging.getLogger("agent_core.task.subagent")

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
        }


class SubagentTaskManager:
    """Execute named subagents and forked child agents through one task interface."""

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

    async def call(
        self,
        *,
        parent_agent: Agent,
        request: SubagentCallRequest,
        timeout: float | None = None,
    ) -> str:
        background = self._resolve_background(request)
        if background:
            task_id = await self.spawn(parent_agent=parent_agent, request=request)
            return (
                f"Background task [{request.description}] started (id: {task_id}). "
                f"Use `task_status(task_id=\"{task_id}\")` to check status."
            )

        result = await self.run_and_wait(
            parent_agent=parent_agent,
            request=request,
            timeout=timeout or 300.0,
        )
        if result.status == "completed":
            return result.final_response or "Task completed but returned empty result."
        if result.status == "cancelled":
            return f"Subagent was cancelled: {result.error or 'Task was cancelled'}"
        return f"Subagent failed: {result.error or 'Unknown error'}"

    async def spawn(self, *, parent_agent: Agent, request: SubagentCallRequest) -> str:
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
        bg_task = asyncio.create_task(self._run_task(task_id, parent_agent, normalized_request))
        self._running_tasks[task_id] = bg_task
        bg_task.add_done_callback(lambda _: self._finalize_running_task(task_id))
        return task_id

    async def run_and_wait(
        self,
        *,
        parent_agent: Agent,
        request: SubagentCallRequest,
        timeout: float = 300.0,
    ) -> SubagentTaskResult:
        task_id = self._new_task_id()
        normalized_request = self._normalize_request(request)
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
        task = asyncio.create_task(self._run_task(task_id, parent_agent, normalized_request))
        self._running_tasks[task_id] = task
        task.add_done_callback(lambda _: self._finalize_running_task(task_id))

        try:
            await asyncio.wait_for(self._completion_events[task_id].wait(), timeout=timeout)
        except asyncio.TimeoutError:
            await self.cancel_task(task_id)
            raise

        return self._task_results[task_id]

    async def run_many(
        self,
        *,
        parent_agent: Agent,
        requests: Iterable[SubagentCallRequest],
        timeout: float = 300.0,
    ) -> list[SubagentTaskResult]:
        return await asyncio.gather(
            *[
                self.run_and_wait(parent_agent=parent_agent, request=request, timeout=timeout)
                for request in requests
            ]
        )

    def get_task_status(self, task_id: str) -> str | None:
        status = self._task_statuses.get(task_id)
        if status is not None:
            return json.dumps(status.to_dict(), ensure_ascii=False, indent=2)
        result = self._task_results.get(task_id)
        if result is not None:
            return json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
        return None

    def list_all_tasks(self) -> str:
        payload = {
            "running": [
                status.to_dict()
                for status in self._task_statuses.values()
                if status.status == "running"
            ],
            "finished": [result.to_dict() for result in self._task_results.values()],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    async def cancel_task(self, task_id: str) -> str:
        task = self._running_tasks.get(task_id)
        if task is None:
            return f"Error: Task '{task_id}' not found or already completed"
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return f"Task '{task_id}' cancellation requested"

    async def _run_task(
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

        try:
            child_agent, initial_message = self._runner.build_execution(
                parent_agent=parent_agent,
                request=request,
            )
            async for event in child_agent.query_stream(initial_message):
                status.last_activity = datetime.now()
                if isinstance(event, ToolCallEvent):
                    tool_calls.append({"tool": event.tool, "args": event.args, "timestamp": time.time()})
                    status.current_step = f"Calling {event.tool}..."
                    status.steps_completed += 1
                elif isinstance(event, ToolResultEvent):
                    tools_used.add(event.tool)
                    status.current_step = f"Result from {event.tool}"
                elif isinstance(event, ThinkingEvent):
                    status.current_step = f"Thinking: {event.content[:50]}..."
                elif isinstance(event, TextEvent):
                    status.current_step = "Generating response..."
                elif isinstance(event, FinalResponseEvent):
                    final_response = event.content

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
                model=request.model,
                run_in_background=self._resolve_background(request),
                tool_calls=tool_calls,
                tools_used=sorted(tools_used),
                completed_at=datetime.now(),
            )
            self._task_results[task_id] = result
            status.status = "completed"
            status.current_step = "Completed"
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
            await self._notify_completion(parent_agent=parent_agent, result=result)
        finally:
            event = self._completion_events.get(task_id)
            if event is not None:
                event.set()

    async def _notify_completion(self, *, parent_agent: Agent, result: SubagentTaskResult) -> None:
        main_agent = self._main_agent or parent_agent
        ui_events = await main_agent._hook_manager.dispatch_subagent_result(main_agent, result)
        if self._context.subagent_events is not None:
            for event in ui_events:
                await self._context.subagent_events.put(event)

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

    def _finalize_running_task(self, task_id: str) -> None:
        self._running_tasks.pop(task_id, None)
