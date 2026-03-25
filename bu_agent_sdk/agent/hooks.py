"""Hook protocol and built-in hooks for the runtime loop."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol

from bu_agent_sdk.agent.events import HiddenUserMessageEvent
from bu_agent_sdk.agent.hitl import HumanApprovalRequest
from bu_agent_sdk.agent.tool_args import ToolArgumentsError, parse_tool_arguments_for_execution
from bu_agent_sdk.agent.runtime_events import (
    FinishRequested,
    IterationStarted,
    RuntimeEvent,
    RunFinished,
    ToolCallRequested,
    ToolResultReceived,
)
from bu_agent_sdk.llm.messages import BaseMessage, ToolMessage, UserMessage

if TYPE_CHECKING:
    from bu_agent_sdk.agent.runtime_state import AgentRunState
    from bu_agent_sdk.agent.service import Agent

logger = logging.getLogger("bu_agent_sdk.agent.hooks")
_EXCEL_COMMAND_RE = re.compile(r"(?i)(openpyxl|\.xlsx\b|\.xlsm\b|\.xltx\b|\.xltm\b)")


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
class ExcelReadGuardHook(BaseAgentHook):
    """Block Excel-related shell retries after a successful read_excel call."""

    priority: int = 19
    state_attr: str = "_successful_excel_reads"

    async def before_event(
        self,
        event: RuntimeEvent,
        ctx: HookContext,
    ) -> HookDecision | None:
        if not isinstance(event, ToolCallRequested):
            return None

        if event.tool_call.function.name != "bash":
            return None

        successful_reads = self._get_successful_reads(ctx)
        if not successful_reads:
            return None

        try:
            args = parse_tool_arguments_for_execution(event.tool_call.function.arguments)
        except ToolArgumentsError:
            return None

        command = args.get("command")
        if not isinstance(command, str) or not _EXCEL_COMMAND_RE.search(command):
            return None

        listed_paths = "\n".join(f"- {path}" for path in successful_reads)
        return HookDecision(
            action=HookAction.OVERRIDE_RESULT,
            override_result=ToolMessage(
                tool_call_id=event.tool_call.id,
                tool_name="bash",
                content=(
                    "Error: `read_excel` already succeeded for the current turn.\n"
                    f"Resolved Excel file(s):\n{listed_paths}\n"
                    "Do not use `bash` to reopen or enumerate Excel workbooks. "
                    "Answer from the existing `read_excel` result, or call `read_excel` "
                    "again on the same resolved path with a different `sheet_name`, "
                    "`max_rows`, or `max_cols` if you need more detail."
                ),
                is_error=True,
            ),
            reason="blocked Excel-related bash retry after read_excel",
        )

    async def after_event(
        self,
        event: RuntimeEvent,
        ctx: HookContext,
        emitted_events: list[RuntimeEvent],
    ) -> HookDecision | None:
        if not isinstance(event, ToolResultReceived):
            return None

        if event.tool_call.function.name != "read_excel":
            return None

        try:
            payload = json.loads(event.tool_result.content)
        except (TypeError, json.JSONDecodeError):
            return None

        resolved_path = payload.get("resolved_path")
        if not isinstance(resolved_path, str) or not resolved_path:
            return None

        successful_reads = self._get_successful_reads(ctx)
        if resolved_path not in successful_reads:
            successful_reads.append(resolved_path)
        return None

    def _get_successful_reads(self, ctx: HookContext) -> list[str]:
        reads = getattr(ctx.state, self.state_attr, None)
        if isinstance(reads, list):
            return reads

        reads = []
        setattr(ctx.state, self.state_attr, reads)
        return reads


@dataclass
class HumanApprovalHook(BaseAgentHook):
    """Request human approval before executing selected tool calls."""

    policy: Any = None
    priority: int = 18

    async def before_event(
        self,
        event: RuntimeEvent,
        ctx: HookContext,
    ) -> HookDecision | None:
        if not isinstance(event, ToolCallRequested):
            return None

        if not ctx.agent.human_in_loop_config.enabled:
            return None

        request = self._build_request(event, ctx)
        if request is None:
            return None

        handler = ctx.agent.human_in_loop_handler
        if handler is None:
            return self._terminate_current_turn(
                event,
                "需要人工审批，但当前未配置审批处理器",
            )

        decision = await handler.request_approval(request)
        if decision.approved:
            return None

        return self._terminate_current_turn(
            event,
            decision.reason or "该工具调用已被人工审批拒绝",
        )

    def _build_request(
        self,
        event: ToolCallRequested,
        ctx: HookContext,
    ) -> HumanApprovalRequest | None:
        if self.policy is None:
            return None
        return self.policy(event, ctx)

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
                if override_result is not None:
                    raise ValueError("Multiple hooks attempted to override the same result.")
                override_result = decision.override_result
                continue
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
