"""Internal event-driven runtime loop for the agent."""

from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime, timezone
from typing import TYPE_CHECKING, AsyncIterator

from agent_core.agent.events import (
    AgentEvent,
    FinalResponseEvent,
    StepCompleteEvent,
    StepStartEvent,
    TextEvent,
    ThinkingEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from agent_core.agent.hooks import HookContext
from agent_core.agent.runtime_events import (
    ContextMaintenanceRequested,
    EphemeralPruneRequested,
    FinishRequested,
    IterationStarted,
    LLMCallRequested,
    LLMResponseReceived,
    RunFailed,
    RunFinished,
    RunStarted,
    RuntimeEvent,
    ToolCallRequested,
    ToolResultReceived,
)
from agent_core.agent.runtime_state import AgentRunState
from agent_core.agent.tool_args import parse_tool_arguments_for_display
from agent_core.llm.messages import AssistantMessage, SystemMessage, ToolMessage, UserMessage

if TYPE_CHECKING:
    from agent_core.agent.service import Agent


class AgentRuntimeLoop:
    """Run the agent loop as an internal event queue."""

    def __init__(self, agent: "Agent", state: AgentRunState):
        self.agent = agent
        self.state = state

    async def run(self, message: str) -> str:
        async for _ in self.run_stream(message, emit_ui_events=False):
            pass

        if self.state.error is not None:
            raise self.state.error

        return self.state.final_response

    async def run_stream(
        self,
        message: str,
        emit_ui_events: bool = True,
    ) -> AsyncIterator[AgentEvent]:
        queue: deque[RuntimeEvent] = deque(
            [RunStarted(message=message, query_mode=self.state.query_mode)]
        )
        hook_ctx = HookContext(agent=self.agent, state=self.state)

        while queue and not self.state.done:
            # Check for cancellation
            if self.state.cancel_event and self.state.cancel_event.is_set():
                self.state.done = True
                self.state.final_response = "[Cancelled by user]"
                self.state.finished_at = datetime.now(timezone.utc)
                yield FinalResponseEvent(content="[Cancelled by user]")
                break

            event = queue.popleft()
            self.state.current_event_name = type(event).__name__

            before = await self.agent._hook_manager.before_event(event, hook_ctx)
            event = before.event
            if before.aborted:
                self._prepend_events(queue, before.emitted_events)
                for ui_event in self._drain_ui_events(hook_ctx, emit_ui_events):
                    yield ui_event
                continue

            emitted_events, ui_events = await self._handle_event(event, before.override_result)
            after = await self.agent._hook_manager.after_event(event, hook_ctx, emitted_events)

            if after.aborted:
                self._prepend_events(queue, after.emitted_events)
                for ui_event in self._merge_ui_events(hook_ctx, ui_events, emit_ui_events):
                    yield ui_event
                continue

            emitted_events = after.emitted_events
            self._prepend_events(queue, emitted_events)
            for ui_event in self._merge_ui_events(hook_ctx, ui_events, emit_ui_events):
                yield ui_event

        if self.state.error is not None:
            raise self.state.error

    async def _handle_event(
        self,
        event: RuntimeEvent,
        override_result,
    ) -> tuple[list[RuntimeEvent], list[AgentEvent]]:
        if isinstance(event, RunStarted):
            if not self.agent._context and self.agent.system_prompt:
                self.agent._context.add_message(
                    SystemMessage(content=self.agent.system_prompt, cache=True)
                )
            self.agent._context.add_message(UserMessage(content=event.message))
            return [IterationStarted(iteration=1)], []

        if isinstance(event, IterationStarted):
            if event.iteration > self.state.max_iterations:
                summary = await self.agent._generate_max_iterations_summary()
                return [RunFinished(final_response=summary, iterations=self.state.iterations)], []

            self.state.iterations = event.iteration
            return [
                EphemeralPruneRequested(iteration=event.iteration),
                LLMCallRequested(
                    messages=self.agent._context.get_messages(),
                    tools=self.agent.tool_definitions if self.agent.tools else None,
                    tool_choice=self.agent.tool_choice if self.agent.tools else None,
                    iteration=event.iteration,
                ),
            ], []

        if isinstance(event, EphemeralPruneRequested):
            self.agent._destroy_ephemeral_messages()
            return [], []

        if isinstance(event, LLMCallRequested):
            # Check for cancellation before LLM call
            if self.state.cancel_event and self.state.cancel_event.is_set():
                return [
                    RunFinished(
                        final_response="[Cancelled by user]", iterations=self.state.iterations
                    )
                ], []

            if event.messages != self.agent._context.get_messages():
                self.agent._context.replace_messages(event.messages)

            try:
                response = await self.agent._invoke_llm()
            except asyncio.CancelledError:
                # LLM call was cancelled
                return [
                    RunFinished(
                        final_response="[Cancelled by user]", iterations=self.state.iterations
                    )
                ], []
            except Exception as error:
                if (
                    not self.state.overflow_recovery_attempted
                    and self.agent._is_context_overflow_error(error)
                ):
                    self.state.overflow_recovery_attempted = True
                    compacted = await self.agent._compact_messages_now()
                    if compacted:
                        return [
                            LLMCallRequested(
                                messages=self.agent._context.get_messages(),
                                tools=self.agent.tool_definitions if self.agent.tools else None,
                                tool_choice=self.agent.tool_choice if self.agent.tools else None,
                                iteration=event.iteration,
                            )
                        ], []
                return [RunFailed(error=error, iteration=event.iteration)], []

            self.state.last_response = response
            self.state.last_usage = response.usage
            return [LLMResponseReceived(response=response, iteration=event.iteration)], []

        if isinstance(event, LLMResponseReceived):
            response = event.response
            self.agent._context.add_message(
                AssistantMessage(
                    content=response.content,
                    tool_calls=response.tool_calls if response.tool_calls else None,
                )
            )

            ui_events: list[AgentEvent] = []
            if response.thinking:
                ui_events.append(ThinkingEvent(content=response.thinking))
            if response.content:
                ui_events.append(TextEvent(content=response.content))

            if not response.has_tool_calls:
                return [
                    ContextMaintenanceRequested(
                        response=response,
                        iteration=event.iteration,
                    ),
                    FinishRequested(
                        final_response=response.content or "",
                        iteration=event.iteration,
                    ),
                ], ui_events

            emitted_events: list[RuntimeEvent] = [
                ToolCallRequested(tool_call=tool_call, iteration=event.iteration)
                for tool_call in response.tool_calls
            ]
            emitted_events.extend(
                [
                    ContextMaintenanceRequested(
                        response=response,
                        iteration=event.iteration,
                    ),
                    IterationStarted(iteration=event.iteration + 1),
                ]
            )
            return emitted_events, ui_events

        if isinstance(event, ToolCallRequested):
            # Check for cancellation before tool execution
            if self.state.cancel_event and self.state.cancel_event.is_set():
                return [
                    RunFinished(
                        final_response="[Cancelled by user]", iterations=self.state.iterations
                    )
                ], []

            tool_name = event.tool_call.function.name
            ui_events: list[AgentEvent] = [
                StepStartEvent(
                    step_id=event.tool_call.id,
                    title=tool_name,
                    step_number=event.iteration,
                ),
                ToolCallEvent(
                    tool=tool_name,
                    args=self._parse_args(event.tool_call.function.arguments),
                    tool_call_id=event.tool_call.id,
                    display_name=tool_name,
                ),
            ]

            if override_result is not None:
                tool_result = self._coerce_override_result(event.tool_call, override_result)
                return [
                    ToolResultReceived(
                        tool_call=event.tool_call,
                        tool_result=tool_result,
                        iteration=event.iteration,
                    )
                ], ui_events

            try:
                tool_result = await self.agent._execute_tool_call(event.tool_call)
            except asyncio.CancelledError:
                return [
                    RunFinished(
                        final_response="[Cancelled by user]", iterations=self.state.iterations
                    )
                ], ui_events
            except self.agent.task_complete_exc_type as task_complete:
                terminal_tool_message = ToolMessage(
                    tool_call_id=event.tool_call.id,
                    tool_name=tool_name,
                    content=f"Task completed: {task_complete.message}",
                    is_error=False,
                )
                return [
                    ToolResultReceived(
                        tool_call=event.tool_call,
                        tool_result=terminal_tool_message,
                        iteration=event.iteration,
                    ),
                    RunFinished(
                        final_response=task_complete.message,
                        iterations=self.state.iterations,
                    ),
                ], ui_events

            return [
                ToolResultReceived(
                    tool_call=event.tool_call,
                    tool_result=tool_result,
                    iteration=event.iteration,
                )
            ], ui_events

        if isinstance(event, ToolResultReceived):
            self.agent._context.add_message(event.tool_result)
            ui_events = [
                ToolResultEvent(
                    tool=event.tool_call.function.name,
                    result=event.tool_result.text,
                    tool_call_id=event.tool_call.id,
                    is_error=event.tool_result.is_error,
                    screenshot_base64=self.agent._extract_screenshot(event.tool_result),
                ),
                StepCompleteEvent(
                    step_id=event.tool_call.id,
                    status="error" if event.tool_result.is_error else "completed",
                    duration_ms=0.0,
                ),
            ]
            return [], ui_events

        if isinstance(event, ContextMaintenanceRequested):
            await self.agent._maintain_context(event.response)
            return [], []

        if isinstance(event, FinishRequested):
            if self.agent.require_done_tool:
                return [IterationStarted(iteration=event.iteration + 1)], []
            return [
                RunFinished(
                    final_response=event.final_response,
                    iterations=self.state.iterations,
                )
            ], []

        if isinstance(event, RunFinished):
            self.state.done = True
            self.state.final_response = event.final_response
            self.state.finished_at = datetime.now(timezone.utc)
            return [], [FinalResponseEvent(content=event.final_response)]

        if isinstance(event, RunFailed):
            self.state.done = True
            self.state.error = event.error
            self.state.finished_at = datetime.now(timezone.utc)
            return [], []

        raise RuntimeError(f"Unhandled runtime event: {type(event).__name__}")

    @staticmethod
    def _prepend_events(queue: deque[RuntimeEvent], events: list[RuntimeEvent]) -> None:
        for event in reversed(events):
            queue.appendleft(event)

    @staticmethod
    def _parse_args(arguments: str) -> dict:
        return parse_tool_arguments_for_display(arguments)

    @staticmethod
    def _drain_ui_events(ctx: HookContext, emit_ui_events: bool) -> list[AgentEvent]:
        if not emit_ui_events or not ctx.ui_events:
            ctx.ui_events.clear()
            return []

        ui_events = list(ctx.ui_events)
        ctx.ui_events.clear()
        return ui_events

    def _merge_ui_events(
        self,
        ctx: HookContext,
        ui_events: list[AgentEvent],
        emit_ui_events: bool,
    ) -> list[AgentEvent]:
        merged = self._drain_ui_events(ctx, emit_ui_events)
        if not emit_ui_events:
            return []
        return [*merged, *ui_events]

    @staticmethod
    def _coerce_override_result(tool_call, override_result) -> ToolMessage:
        if isinstance(override_result, ToolMessage):
            return override_result
        return ToolMessage(
            tool_call_id=tool_call.id,
            tool_name=tool_call.function.name,
            content=str(override_result),
            is_error=False,
        )
