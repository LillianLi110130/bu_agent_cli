"""Human-in-the-loop approval models and policies."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from bu_agent_sdk.agent.runtime_events import ToolCallRequested

if TYPE_CHECKING:
    from bu_agent_sdk.agent.hooks import HookContext


@dataclass
class HumanInLoopConfig:
    """Runtime configuration for human approval."""

    enabled: bool = False


@dataclass
class HumanApprovalRequest:
    """Approval request shown to a human operator."""

    tool_name: str
    tool_call_id: str
    arguments: dict[str, Any]
    reason: str
    risk_level: str = "medium"
    command_preview: str | None = None


@dataclass
class HumanApprovalDecision:
    """Decision returned by the human approval handler."""

    approved: bool
    reason: str | None = None


class HumanInLoopHandler(Protocol):
    """Runtime interface used by hooks to request approval."""

    async def request_approval(
        self,
        request: HumanApprovalRequest,
    ) -> HumanApprovalDecision:
        ...


ApprovalPolicy = Callable[[ToolCallRequested, "HookContext"], HumanApprovalRequest | None]


def _parse_tool_arguments(arguments: str) -> dict[str, Any]:
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return {"_raw": arguments}

    if isinstance(parsed, dict):
        return parsed
    return {"_value": parsed}


def build_default_approval_policy(mode: str) -> ApprovalPolicy:
    """Build the default approval policy for an agent mode."""

    def policy(event: ToolCallRequested, ctx: HookContext) -> HumanApprovalRequest | None:
        del ctx

        if mode == "subagent":
            return None

        tool_name = event.tool_call.function.name
        if tool_name != "bash":
            return None

        arguments = _parse_tool_arguments(event.tool_call.function.arguments)
        command = str(arguments.get("command", "")).strip()
        if command:
            preview = command if len(command) <= 120 else command[:117].rstrip() + "..."
            reason = "即将执行 Shell 命令，需人工审批。"
        else:
            reason = "Shell 命令需要人工审批。"
            preview = None

        return HumanApprovalRequest(
            tool_name=tool_name,
            tool_call_id=event.tool_call.id,
            arguments=arguments,
            reason=reason,
            risk_level="high",
            command_preview=preview,
        )

    return policy
