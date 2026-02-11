"""
Subagent Manager for background task execution.

Manages asynchronous subagent tasks that run in the background,
allowing the main agent to continue processing without blocking.
"""

import asyncio
import json
import time
import uuid
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Any, Awaitable

from bu_agent_sdk.agent import Agent
from bu_agent_sdk.agent.config import AgentConfig
from bu_agent_sdk.agent.events import (
    AgentEvent,
    ToolCallEvent,
    ToolResultEvent,
    ThinkingEvent,
    TextEvent,
    FinalResponseEvent,
)

logger = logging.getLogger("bu_agent_sdk.subagent_manager")


@dataclass
class TaskResult:
    """Result of a completed subagent task."""

    task_id: str
    subagent_name: str
    prompt: str
    final_response: str
    execution_time_ms: float
    status: str  # "completed", "failed", "cancelled"
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    error: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "subagent_name": self.subagent_name,
            "prompt": self.prompt,
            "final_response": self.final_response,
            "execution_time_ms": self.execution_time_ms,
            "status": self.status,
            "tool_calls": self.tool_calls,
            "tools_used": self.tools_used,
            "error": self.error,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


@dataclass
class TaskStatus:
    """Current status of a running task."""

    task_id: str
    subagent_name: str
    label: str | None
    status: str  # "running", "completed", "failed", "cancelled"
    created_at: datetime
    current_step: str | None = None
    steps_completed: int = 0
    last_activity: datetime | None = None


class SubagentManager:
    """
    Manager for background subagent tasks.

    Subagents run asynchronously in the background, allowing the main agent
    to continue processing without blocking. Results are available via
    callbacks or polling.
    """

    def __init__(
        self,
        agent_factory: Callable[
            [AgentConfig, Any, list[Any]], Agent
        ],  # factory(config, context, all_tools) -> Agent
        registry: Any,  # AgentRegistry instance
        all_tools: list[Any],
        workspace: Path,
        context: Any,  # SandboxContext to pass to subagents
    ):
        """
        Initialize the SubagentManager.

        Args:
            agent_factory: Factory function to create Agent instances.
            registry: AgentRegistry to get subagent configurations.
            all_tools: List of all available tools (subagents filter these).
            workspace: Working directory for subagents.
            context: Context to pass to subagents (e.g., SandboxContext).
        """
        self._agent_factory = agent_factory
        self._registry = registry
        self._all_tools = all_tools
        self._workspace = workspace
        self._context = context

        # Task storage
        self._running_tasks: dict[str, asyncio.Task[None]] = {}
        self._task_results: dict[str, TaskResult] = {}
        self._task_statuses: dict[str, TaskStatus] = {}

        # Main agent instance (optional) for injecting results
        self._main_agent: Any | None = None

        # Result callback - called when a task completes
        self._result_callback: Callable[[TaskResult], Awaitable[None]] | None = None

    def set_main_agent(self, agent: Any):
        """Set the main agent instance for result injection."""
        self._main_agent = agent

    def set_result_callback(
        self, callback: Callable[[TaskResult], Awaitable[None]]
    ):
        """Set a callback function to be called when tasks complete."""
        self._result_callback = callback

    async def _inject_result_to_main_agent(self, result: TaskResult):
        """Inject task result as a system message into main agent's conversation history.

        This allows the main agent to know about subagent task results.
        """
        if not self._main_agent:
            return

        # Get context from main agent
        ctx = getattr(self._main_agent, "_context", None)
        if not ctx:
            return

        # Format result as a system message
        content = f"""
[Subagent Task Completed]

Task ID: {result.task_id}
Subagent: {result.subagent_name}
Status: {result.status}

Result:
{result.final_response}
"""
        # Add as SystemMessage (like system prompt)
        from bu_agent_sdk.llm.messages import SystemMessage
        ctx.inject_message(SystemMessage(content=content), pinned=True)

    async def spawn(
        self,
        subagent_name: str,
        prompt: str,
        label: str | None = None,
    ) -> str:
        """
        Spawn a subagent to execute a task in the background.

        Args:
            subagent_name: Name of the subagent to spawn.
            prompt: Task description/prompt for the subagent.
            label: Optional human-readable label for the task.

        Returns:
            Status message with task ID.
        """
        # Validate subagent exists
        config = self._registry.get_config(subagent_name)
        if not config:
            return f"Error: Subagent '{subagent_name}' not found"

        # Validate subagent mode
        if config.mode not in ("subagent", "all"):
            return (
                f"Error: Subagent '{subagent_name}' has mode '{config.mode}', "
                "cannot be called as a background task"
            )

        # Generate task ID
        task_id = str(uuid.uuid4())[:8]
        display_label = label or subagent_name

        # Create status entry
        status = TaskStatus(
            task_id=task_id,
            subagent_name=subagent_name,
            label=label,
            status="running",
            created_at=datetime.now(),
            current_step="Starting...",
        )
        self._task_statuses[task_id] = status

        # Create background task
        bg_task = asyncio.create_task(
            self._run_subagent(task_id, subagent_name, prompt, display_label)
        )
        self._running_tasks[task_id] = bg_task

        # Cleanup when done
        bg_task.add_done_callback(lambda _: self._on_task_complete(task_id))

        logger.info(f"Spawned subagent task [{task_id}]: {display_label}")
        return f"Background task [{display_label}] started (id: {task_id}). Use `/task {task_id}` to check status."

    async def _run_subagent(
        self, task_id: str, subagent_name: str, prompt: str, label: str
    ) -> None:
        """Execute the subagent task and collect results."""
        logger.info(f"Subagent [{task_id}] starting: {label}")

        config = self._registry.get_config(subagent_name)
        if not config:
            await self._handle_task_error(
                task_id, subagent_name, prompt, f"Subagent config not found"
            )
            return

        try:
            # Create subagent with mode="subagent" (automatically filters tools)
            subagent = self._agent_factory(
                config=config, parent_ctx=self._context, all_tools=self._all_tools
            )

            start_time = time.time()
            tool_calls = []
            tools_used = set()
            final_response = ""
            events_log = []
            task_status = self._task_statuses.get(task_id)

            # Execute subagent and collect events
            async for event in subagent.query_stream(prompt):
                events_log.append(event)
                if task_status:
                    task_status.last_activity = datetime.now()

                if isinstance(event, ToolCallEvent):
                    tool_calls.append(
                        {
                            "tool": event.tool,
                            "args": event.args,
                            "timestamp": time.time(),
                        }
                    )
                    if task_status:
                        task_status.current_step = f"Calling {event.tool}..."
                        task_status.steps_completed += 1
                    logger.debug(f"Subagent [{task_id}] calling: {event.tool}")

                elif isinstance(event, ToolResultEvent):
                    tools_used.add(event.tool)
                    if task_status:
                        task_status.current_step = f"Result from {event.tool}"
                    logger.debug(f"Subagent [{task_id}] result: {event.tool}")

                elif isinstance(event, ThinkingEvent):
                    if task_status:
                        task_status.current_step = f"Thinking: {event.content[:50]}..."

                elif isinstance(event, TextEvent):
                    if task_status:
                        task_status.current_step = f"Generating response..."

                elif isinstance(event, FinalResponseEvent):
                    final_response = event.content
                    if task_status:
                        task_status.current_step = "Completed"

            execution_time_ms = (time.time() - start_time) * 1000

            # Create result
            result = TaskResult(
                task_id=task_id,
                subagent_name=subagent_name,
                prompt=prompt,
                final_response=final_response,
                execution_time_ms=execution_time_ms,
                status="completed",
                tool_calls=tool_calls,
                tools_used=list(tools_used),
                completed_at=datetime.now(),
            )

            # Store result
            self._task_results[task_id] = result

            # Update status
            if task_status:
                task_status.status = "completed"
                task_status.current_step = "Completed"

            logger.info(f"Subagent [{task_id}] completed successfully")

            # Notify via callback
            if self._result_callback:
                try:
                    await self._result_callback(result)
                except Exception as e:
                    logger.error(f"Error in result callback: {e}")

            # Inject result into main agent's conversation
            await self._inject_result_to_main_agent(result)

        except asyncio.CancelledError:
            logger.info(f"Subagent [{task_id}] was cancelled")
            await self._handle_task_cancelled(task_id, subagent_name, prompt)

        except Exception as e:
            logger.error(f"Subagent [{task_id}] failed: {e}")
            await self._handle_task_error(task_id, subagent_name, prompt, str(e))

    async def _handle_task_error(
        self, task_id: str, subagent_name: str, prompt: str, error: str
    ):
        """Handle a task error."""
        task_status = self._task_statuses.get(task_id)
        result = TaskResult(
            task_id=task_id,
            subagent_name=subagent_name,
            prompt=prompt,
            final_response="",
            execution_time_ms=0,
            status="failed",
            error=error,
            completed_at=datetime.now(),
        )
        self._task_results[task_id] = result

        # Update status
        if task_status:
            task_status.status = "cancelled"
            task_status.current_step = "Cancelled"

        # Notify
        if self._result_callback:
            try:
                await self._result_callback(result)
            except Exception as e:
                logger.error(f"Error in result callback: {e}")

        # Inject result into main agent's conversation
        await self._inject_result_to_main_agent(result)

    async def _handle_task_cancelled(
        self, task_id: str, subagent_name: str, prompt: str
    ):
        """Handle a task cancellation."""
        result = TaskResult(
            task_id=task_id,
            subagent_name=subagent_name,
            prompt=prompt,
            final_response="",
            execution_time_ms=0,
            status="cancelled",
            error="Task was cancelled",
            completed_at=datetime.now(),
        )
        self._task_results[task_id] = result

        # Update status
        task_status = self._task_statuses.get(task_id)
        if task_status:
            task_status.status = "cancelled"
            task_status.current_step = "Cancelled"

        # Notify
        if self._result_callback:
            try:
                await self._result_callback(result)
            except Exception as e:
                logger.error(f"Error in result callback: {e}")

    def _on_task_complete(self, task_id: str):
        """Callback when task completes (for cleanup)."""
        self._running_tasks.pop(task_id, None)

    async def cancel_task(self, task_id: str) -> str:
        """
        Cancel a running task.

        Args:
            task_id: ID of the task to cancel.

        Returns:
            Status message.
        """
        # Check if task exists
        status = self._task_statuses.get(task_id)
        if not status:
            return f"Error: Task '{task_id}' not found"

        if status.status != "running":
            return f"Error: Task '{task_id}' is not running (status: {status.status})"

        # Cancel the task
        bg_task = self._running_tasks.get(task_id)
        if not bg_task:
            return f"Error: Task '{task_id}' not found in running tasks"

        bg_task.cancel()
        return f"Task '{task_id}' cancellation requested"

    def get_task_status(self, task_id: str) -> str | None:
        """
        Get status of a specific task.

        Args:
            task_id: ID of the task.

        Returns:
            Task status information as JSON string, or None if not found.
        """
        status = self._task_statuses.get(task_id)
        if not status:
            return None

        # Combine with result if completed
        result = self._task_results.get(task_id)

        info = {
            "task_id": status.task_id,
            "subagent_name": status.subagent_name,
            "label": status.label,
            "status": status.status,
            "created_at": status.created_at.isoformat(),
            "current_step": status.current_step,
            "steps_completed": status.steps_completed,
            "last_activity": status.last_activity.isoformat() if status.last_activity else None,
        }

        if result:
            info["execution_time_ms"] = result.execution_time_ms
            info["final_response"] = result.final_response[:500]  # Preview
            info["error"] = result.error
            if result.completed_at:
                info["completed_at"] = result.completed_at.isoformat()

        return json.dumps(info, indent=2, default=str)

    def list_all_tasks(self) -> str:
        """
        List all tasks (running and completed).

        Returns:
            JSON string with task information.
        """
        tasks = []
        for task_id, status in self._task_statuses.items():
            task_info = {
                "task_id": status.task_id,
                "subagent_name": status.subagent_name,
                "label": status.label,
                "status": status.status,
                "created_at": status.created_at.isoformat(),
                "current_step": status.current_step,
            }

            # Add result info if available
            result = self._task_results.get(task_id)
            if result:
                task_info["execution_time_ms"] = result.execution_time_ms
                task_info["tools_used"] = result.tools_used
                if result.completed_at:
                    task_info["completed_at"] = result.completed_at.isoformat()

            tasks.append(task_info)

        return json.dumps(
            {"total_tasks": len(tasks), "tasks": tasks}, indent=2, default=str
        )

    def get_running_count(self) -> int:
        """Return the number of currently running tasks."""
        return len(self._running_tasks)

    def get_completed_count(self) -> int:
        """Return the number of completed tasks."""
        return len([r for r in self._task_results.values() if r.status == "completed"])

    def get_total_count(self) -> int:
        """Return the total number of tasks created."""
        return len(self._task_statuses)
