"""Permission decisions for tool calls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from agent_core.agent.command_safety import (
    CommandSafetyFinding,
    CommandSafetySeverity,
    check_dangerous_command,
    command_preview,
    format_findings,
    safety_session_label,
)
from agent_core.agent.hitl import HumanApprovalRequest
from agent_core.agent.tool_args import ToolArgumentsError, parse_tool_arguments_for_execution

if TYPE_CHECKING:
    from agent_core.agent.hooks import HookContext
    from agent_core.agent.runtime_events import ToolCallRequested


PermissionDecisionType = Literal["allow", "ask", "deny"]


@dataclass(frozen=True)
class PermissionReason:
    source: str
    rule_id: str | None
    message: str
    category: str | None = None
    severity: str | None = None


@dataclass(frozen=True)
class PermissionDecision:
    decision: PermissionDecisionType
    reasons: tuple[PermissionReason, ...] = ()
    approval_request: HumanApprovalRequest | None = None

    @classmethod
    def allow(cls) -> "PermissionDecision":
        return cls(decision="allow")


class PermissionEngine:
    """Evaluate tool calls before execution."""

    def evaluate_tool_call(
        self,
        event: "ToolCallRequested",
        ctx: "HookContext",
    ) -> PermissionDecision:
        del ctx
        if event.tool_call.function.name != "bash":
            return PermissionDecision.allow()

        try:
            args = parse_tool_arguments_for_execution(event.tool_call.function.arguments)
        except ToolArgumentsError:
            return PermissionDecision.allow()

        command = args.get("command")
        if not isinstance(command, str) or not command.strip():
            return PermissionDecision.allow()

        result = check_dangerous_command(command)
        if result.action == "allow":
            return PermissionDecision.allow()

        reasons = _reasons_from_findings(result.findings)
        if result.action == "block":
            return PermissionDecision(decision="deny", reasons=reasons)

        findings = result.ask_findings
        request = HumanApprovalRequest(
            tool_name="bash",
            tool_call_id=event.tool_call.id,
            arguments=args,
            reason=(
                "该 bash 命令命中安全审查规则，需要人工确认。\n"
                f"Matched safety rule(s):\n{format_findings(findings)}"
            ),
            risk_level=_highest_severity(findings),
            command_preview=command_preview(command, max_chars=200),
            approval_kind="safety",
            approval_keys=tuple(finding.approval_key for finding in findings),
            session_approval_label=safety_session_label(findings),
        )
        return PermissionDecision(
            decision="ask",
            reasons=_reasons_from_findings(findings),
            approval_request=request,
        )


def _reasons_from_findings(
    findings: tuple[CommandSafetyFinding, ...],
) -> tuple[PermissionReason, ...]:
    return tuple(
        PermissionReason(
            source="command_safety",
            rule_id=finding.rule_id,
            message=finding.description,
            category=finding.category,
            severity=finding.severity,
        )
        for finding in findings
    )


def _highest_severity(
    findings: tuple[CommandSafetyFinding, ...],
) -> CommandSafetySeverity:
    order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    if not findings:
        return "medium"
    return max((finding.severity for finding in findings), key=lambda item: order[item])
