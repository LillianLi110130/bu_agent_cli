"""Hook protocol and built-in hooks for the runtime loop."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol

from agent_core.agent.events import HiddenUserMessageEvent
from agent_core.agent.permissions import PermissionDecision, PermissionEngine
from agent_core.agent.runtime_events import (
    FinishRequested,
    IterationStarted,
    RunFinished,
    RuntimeEvent,
    ToolCallRequested,
    ToolResultReceived,
)
from agent_core.llm.messages import BaseMessage, ToolMessage, UserMessage

if TYPE_CHECKING:
    from agent_core.agent.runtime_state import AgentRunState
    from agent_core.agent.service import Agent
    from agent_core.task import SubagentTaskResult

logger = logging.getLogger("agent_core.agent.hooks")


class HookAction(str, Enum):
    CONTINUE = "continue"
    ABORT = "abort"
    REPLACE_EVENT = "replace_event"
    EMIT_EVENTS = "emit_events"
    OVERRIDE_RESULT = "override_result"


@dataclass
class HookDecision:
    action: HookAction = HookAction.CONTINUE
    replacement_event: RuntimeEvent | None = None
    emitted_events: list[RuntimeEvent] = field(default_factory=list)
    override_result: Any | None = None
    reason: str | None = None


@dataclass
class HookDispatchResult:
    event: RuntimeEvent
    emitted_events: list[RuntimeEvent] = field(default_factory=list)
    override_result: Any | None = None
    aborted: bool = False
    reason: str | None = None


@dataclass
class HookContext:
    agent: "Agent"
    state: "AgentRunState"
    ui_events: list[Any] = field(default_factory=list)

    @property
    def query_mode(self) -> str:
        return self.state.query_mode

    @property
    def context_manager(self):
        return self.agent._context

    def add_message(self, message: BaseMessage) -> None:
        self.agent._context.add_message(message)

    def inject_message(self, message: BaseMessage, pinned: bool = True) -> None:
        self.agent._context.inject_message(message, pinned=pinned)

    def emit_ui_event(self, event: Any) -> None:
        self.ui_events.append(event)

    def mark_done(self, response: str) -> None:
        self.state.done = True
        self.state.final_response = response

    def abort(self, reason: str | None = None) -> HookDecision:
        return HookDecision(action=HookAction.ABORT, reason=reason)


class AgentHook(Protocol):
    priority: int

    async def before_event(
        self,
        event: RuntimeEvent,
        ctx: HookContext,
    ) -> HookDecision | None: ...

    async def after_event(
        self,
        event: RuntimeEvent,
        ctx: HookContext,
        emitted_events: list[RuntimeEvent],
    ) -> HookDecision | None: ...


@dataclass
class BaseAgentHook:
    priority: int = 100

    async def before_event(
        self,
        event: RuntimeEvent,
        ctx: HookContext,
    ) -> HookDecision | None:
        return None

    async def after_event(
        self,
        event: RuntimeEvent,
        ctx: HookContext,
        emitted_events: list[RuntimeEvent],
    ) -> HookDecision | None:
        return None

    async def on_subagent_result(
        self,
        result: "SubagentTaskResult",
        ctx: "SubagentHookContext",
    ) -> None:
        return None


@dataclass
class SubagentHookContext:
    agent: "Agent"
    ui_events: list[Any] = field(default_factory=list)

    def inject_message(self, message: BaseMessage, pinned: bool = True) -> None:
        self.agent._context.inject_message(message, pinned=pinned)

    def emit_ui_event(self, event: Any) -> None:
        self.ui_events.append(event)


@dataclass
class FinishGuardHook(BaseAgentHook):
    """Prevent the agent from finishing before its own todo guard passes."""

    priority: int = 10

    async def before_event(
        self,
        event: RuntimeEvent,
        ctx: HookContext,
    ) -> HookDecision | None:
        if not isinstance(event, FinishRequested):
            return None

        if ctx.agent.require_done_tool or ctx.state.incomplete_todos_prompted:
            return None

        incomplete_prompt = await ctx.agent._get_incomplete_todos_prompt()
        if not incomplete_prompt:
            return None

        ctx.state.incomplete_todos_prompted = True
        ctx.add_message(UserMessage(content=incomplete_prompt))
        if ctx.query_mode != "query":
            ctx.emit_ui_event(HiddenUserMessageEvent(content=incomplete_prompt))

        return HookDecision(
            action=HookAction.ABORT,
            emitted_events=[IterationStarted(iteration=ctx.state.iterations + 1)],
            reason="finish deferred by FinishGuardHook",
        )


@dataclass
class ToolPolicyHook(BaseAgentHook):
    """Block or allow tool invocations by name before execution."""

    allow_tool_names: set[str] | None = None
    deny_tool_names: set[str] | None = None
    priority: int = 20

    async def before_event(
        self,
        event: RuntimeEvent,
        ctx: HookContext,
    ) -> HookDecision | None:
        if not isinstance(event, ToolCallRequested):
            return None

        tool_name = event.tool_call.function.name
        denied = self.deny_tool_names and tool_name in self.deny_tool_names
        not_allowed = self.allow_tool_names is not None and tool_name not in self.allow_tool_names
        if not denied and not not_allowed:
            return None

        reason = f"Tool '{tool_name}' blocked by ToolPolicyHook"
        return HookDecision(
            action=HookAction.OVERRIDE_RESULT,
            override_result=ToolMessage(
                tool_call_id=event.tool_call.id,
                tool_name=tool_name,
                content=f"Error: {reason}",
                is_error=True,
            ),
            reason=reason,
        )


@dataclass
class PermissionEnforcementHook(BaseAgentHook):
    """Enforce PermissionEngine decisions before tool execution."""

    engine: PermissionEngine = field(default_factory=PermissionEngine)
    priority: int = 12
    _session_approval_keys: set[str] = field(default_factory=set)

    async def before_event(
        self,
        event: RuntimeEvent,
        ctx: HookContext,
    ) -> HookDecision | None:
        if not isinstance(event, ToolCallRequested):
            return None

        decision = self.engine.evaluate_tool_call(event, ctx)
        if decision.decision == "allow":
            return None
        if decision.decision == "deny":
            return self._deny_tool_call(event, decision)
        if decision.decision == "ask":
            return await self._ask_or_allow(event, ctx, decision)
        return None

    async def _ask_or_allow(
        self,
        event: ToolCallRequested,
        ctx: HookContext,
        decision: PermissionDecision,
    ) -> HookDecision | None:
        request = decision.approval_request
        if request is None:
            return None

        if self._is_session_approved(request):
            return None
        if not ctx.agent.human_in_loop_config.enabled:
            return None

        handler = ctx.agent.human_in_loop_handler
        if handler is None:
            return self._terminate_current_turn(
                event,
                "需要人工审批，但当前未配置审批处理器",
            )

        human_decision = await handler.request_approval(request)
        if human_decision.approved:
            if human_decision.scope == "session":
                self._session_approval_keys.update(request.approval_keys)
            return None

        return self._terminate_current_turn(
            event,
            human_decision.reason or "该工具调用已被人工审批拒绝",
        )

    @staticmethod
    def _deny_tool_call(
        event: ToolCallRequested,
        decision: PermissionDecision,
    ) -> HookDecision:
        tool_name = event.tool_call.function.name
        reason_lines = "\n".join(
            f"- {reason.source}:{reason.rule_id or 'unknown'}: {reason.message}"
            for reason in decision.reasons
        )
        guidance_lines = tuple(
            dict.fromkeys(reason.guidance for reason in decision.reasons if reason.guidance)
        )
        if not guidance_lines:
            guidance_lines = (
                "This command is classified as a hard-blocked destructive operation. "
                "Do not retry it or rephrase it through another shell/interpreter. "
                "Ask the user for a safer, narrower operation instead.",
            )

        content_parts = [
            "Error: Bash command blocked by permission policy.",
            "Command was not executed.",
            f"Matched rule(s):\n{reason_lines}",
            "Guidance:\n" + "\n".join(f"- {line}" for line in guidance_lines),
        ]
        content = "\n\n".join(content_parts)
        return HookDecision(
            action=HookAction.OVERRIDE_RESULT,
            override_result=ToolMessage(
                tool_call_id=event.tool_call.id,
                tool_name=tool_name,
                content=content,
                is_error=True,
            ),
            reason="permission denied by PermissionEngine",
        )

    def _is_session_approved(self, request) -> bool:
        if request.approval_kind != "safety":
            return False
        if not request.approval_keys:
            return False
        return all(key in self._session_approval_keys for key in request.approval_keys)

    @staticmethod
    def _terminate_current_turn(
        event: ToolCallRequested,
        reason: str,
    ) -> HookDecision:
        tool_name = event.tool_call.function.name
        normalized_reason = " ".join(reason.split())
        message = f"工具 '{tool_name}' 已被人工审批拒绝"
        if normalized_reason:
            message += f"（{normalized_reason}）"
        final_response = (
            f"工具 '{tool_name}' 未执行，因为人工审批未通过。"
            "本轮对话已结束，请输入你的下一条请求。"
        )
        return HookDecision(
            action=HookAction.ABORT,
            emitted_events=[
                ToolResultReceived(
                    tool_call=event.tool_call,
                    tool_result=ToolMessage(
                        tool_call_id=event.tool_call.id,
                        tool_name=tool_name,
                        content=message,
                        is_error=True,
                    ),
                    iteration=event.iteration,
                ),
                RunFinished(
                    final_response=final_response,
                    iterations=event.iteration,
                ),
            ],
            reason=final_response,
        )


@dataclass
class AuditHook(BaseAgentHook):
    """Record runtime activity for debugging and observability."""

    priority: int = 1000
    records: list[dict[str, Any]] = field(default_factory=list)

    async def before_event(
        self,
        event: RuntimeEvent,
        ctx: HookContext,
    ) -> HookDecision | None:
        self.records.append({"phase": "before", "event": type(event).__name__})
        logger.debug("[hook:audit] before %s", type(event).__name__)
        return None

    async def after_event(
        self,
        event: RuntimeEvent,
        ctx: HookContext,
        emitted_events: list[RuntimeEvent],
    ) -> HookDecision | None:
        self.records.append(
            {
                "phase": "after",
                "event": type(event).__name__,
                "emitted": [type(item).__name__ for item in emitted_events],
            }
        )
        logger.debug("[hook:audit] after %s -> %s", type(event).__name__, len(emitted_events))
        return None

    async def on_subagent_result(
        self,
        result: "SubagentTaskResult",
        ctx: "SubagentHookContext",
    ) -> None:
        self.records.append(
            {
                "phase": "subagent",
                "event": "SubagentTaskResult",
                "task_id": result.task_id,
                "status": result.status,
                "subagent_name": result.subagent_name,
            }
        )


@dataclass
class SubagentCompletionHook(BaseAgentHook):
    """Inject completed background subagent results back into the parent agent."""

    priority: int = 110

    async def on_subagent_result(
        self,
        result: "SubagentTaskResult",
        ctx: "SubagentHookContext",
    ) -> None:
        if not result.run_in_background:
            return

        title = (
            f"Background subagent '{result.description or result.subagent_name}' completed."
            if result.status == "completed"
            else (
                f"Background subagent '{result.description or result.subagent_name}' "
                f"finished with status '{result.status}'."
            )
        )
        body = result.final_response.strip() or (result.error or "").strip() or "(no output)"
        message = (
            f"{title}\n"
            f"task_id={result.task_id}\n"
            f"type={result.subagent_type or 'fork'}\n"
            f"mode={result.task_kind}\n"
            f"result:\n{body}"
        )
        ctx.inject_message(UserMessage(content=message), pinned=True)
        ctx.emit_ui_event(result)


@dataclass
class HookManager:
    hooks: list[AgentHook] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.hooks.sort(key=lambda hook: getattr(hook, "priority", 100))

    async def before_event(
        self,
        event: RuntimeEvent,
        ctx: HookContext,
    ) -> HookDispatchResult:
        return await self._dispatch("before", event, ctx, [])

    async def after_event(
        self,
        event: RuntimeEvent,
        ctx: HookContext,
        emitted_events: list[RuntimeEvent],
    ) -> HookDispatchResult:
        return await self._dispatch("after", event, ctx, emitted_events)

    async def _dispatch(
        self,
        phase: str,
        event: RuntimeEvent,
        ctx: HookContext,
        emitted_events: list[RuntimeEvent],
    ) -> HookDispatchResult:
        current_event = event
        queued_events = list(emitted_events)
        override_result = None

        for hook in self.hooks:
            if phase == "before":
                decision = await hook.before_event(current_event, ctx)
            else:
                decision = await hook.after_event(current_event, ctx, queued_events)
            if decision is None:
                continue

            queued_events.extend(decision.emitted_events)

            if decision.action == HookAction.CONTINUE or decision.action == HookAction.EMIT_EVENTS:
                continue
            if decision.action == HookAction.REPLACE_EVENT:
                if decision.replacement_event is not None:
                    current_event = decision.replacement_event
                continue
            if decision.action == HookAction.OVERRIDE_RESULT:
                override_result = decision.override_result
                return HookDispatchResult(
                    event=current_event,
                    emitted_events=queued_events,
                    override_result=override_result,
                    aborted=False,
                )
            if decision.action == HookAction.ABORT:
                return HookDispatchResult(
                    event=current_event,
                    emitted_events=queued_events,
                    override_result=override_result,
                    aborted=True,
                    reason=decision.reason,
                )

        return HookDispatchResult(
            event=current_event,
            emitted_events=queued_events,
            override_result=override_result,
            aborted=False,
        )

    async def dispatch_subagent_result(
        self,
        agent: "Agent",
        result: "SubagentTaskResult",
    ) -> list[Any]:
        ctx = SubagentHookContext(agent=agent)
        for hook in self.hooks:
            handler = getattr(hook, "on_subagent_result", None)
            if handler is None:
                continue
            await handler(result, ctx)
        return list(ctx.ui_events)
