from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agent_core.agent.command_safety import (
    build_command_safety_approval_policy,
    check_dangerous_command,
    normalize_command,
)
from agent_core.agent.hitl import (
    HumanApprovalDecision,
    HumanInLoopConfig,
    build_default_approval_policy,
)
from agent_core.agent.hooks import DangerousBashCommandGuardHook, HookAction, HumanApprovalHook
from agent_core.agent.runtime_events import ToolCallRequested
from agent_core.llm.messages import Function, ToolCall


class FakeApprovalHandler:
    def __init__(self, decision: HumanApprovalDecision):
        self.decision = decision
        self.requests = []

    async def request_approval(self, request):
        self.requests.append(request)
        return self.decision


def _bash_event(command: str) -> ToolCallRequested:
    return ToolCallRequested(
        tool_call=ToolCall(
            id="call-1",
            function=Function(
                name="bash",
                arguments=json.dumps({"command": command}),
            ),
        ),
        iteration=1,
    )


def _ctx(handler, *, enabled: bool = False):
    return SimpleNamespace(
        agent=SimpleNamespace(
            human_in_loop_config=HumanInLoopConfig(enabled=enabled),
            human_in_loop_handler=handler,
        )
    )


def test_command_safety_normalizes_ansi_null_and_width_chars() -> None:
    command = "\x1b[31mＧＩＴ\x00   RESET   --HARD\x1b[0m"

    assert normalize_command(command) == "git reset --hard"


def test_command_safety_classifies_block_ask_and_allow() -> None:
    assert check_dangerous_command("rm -rf /").action == "block"
    assert check_dangerous_command("git reset --hard").action == "ask"
    assert check_dangerous_command("pytest -q").action == "allow"


@pytest.mark.asyncio
async def test_dangerous_bash_command_guard_blocks_hard_block_command() -> None:
    hook = DangerousBashCommandGuardHook()

    decision = await hook.before_event(_bash_event("rm -rf /"), SimpleNamespace())

    assert decision is not None
    assert decision.action == HookAction.OVERRIDE_RESULT
    assert decision.override_result.is_error is True
    assert "Dangerous bash command blocked" in decision.override_result.text
    assert "rm_recursive_broad_scope" in decision.override_result.text


@pytest.mark.asyncio
async def test_dangerous_bash_command_guard_ignores_ask_command() -> None:
    hook = DangerousBashCommandGuardHook()

    decision = await hook.before_event(_bash_event("git reset --hard"), SimpleNamespace())

    assert decision is None


@pytest.mark.asyncio
async def test_mandatory_policy_asks_even_when_approval_mode_disabled() -> None:
    handler = FakeApprovalHandler(HumanApprovalDecision(approved=True, scope="once"))
    hook = HumanApprovalHook(
        mandatory_policy=build_command_safety_approval_policy(),
        policy=build_default_approval_policy(),
    )

    decision = await hook.before_event(
        _bash_event("git reset --hard"),
        _ctx(handler, enabled=False),
    )

    assert decision is None
    assert len(handler.requests) == 1
    request = handler.requests[0]
    assert request.approval_kind == "safety"
    assert request.approval_keys == ("safety:git_reset_hard",)
    assert request.session_approval_label == "git reset --hard"


@pytest.mark.asyncio
async def test_safety_session_approval_skips_later_same_rule() -> None:
    handler = FakeApprovalHandler(HumanApprovalDecision(approved=True, scope="session"))
    hook = HumanApprovalHook(mandatory_policy=build_command_safety_approval_policy())

    first = await hook.before_event(_bash_event("git reset --hard"), _ctx(handler))
    second = await hook.before_event(_bash_event("git reset --hard"), _ctx(handler))

    assert first is None
    assert second is None
    assert len(handler.requests) == 1


@pytest.mark.asyncio
async def test_safety_approval_deny_aborts_and_emits_tool_result() -> None:
    handler = FakeApprovalHandler(
        HumanApprovalDecision(approved=False, reason="operator denied", scope="deny")
    )
    hook = HumanApprovalHook(mandatory_policy=build_command_safety_approval_policy())

    decision = await hook.before_event(_bash_event("git clean -fdx"), _ctx(handler))

    assert decision is not None
    assert decision.action == HookAction.ABORT
    assert decision.emitted_events[0].tool_result.is_error is True
    assert "operator denied" in decision.emitted_events[0].tool_result.text


@pytest.mark.asyncio
async def test_normal_approval_still_respects_approval_mode_switch() -> None:
    handler = FakeApprovalHandler(HumanApprovalDecision(approved=True, scope="once"))
    hook = HumanApprovalHook(policy=build_default_approval_policy())

    skipped = await hook.before_event(_bash_event("pytest -q"), _ctx(handler, enabled=False))
    asked = await hook.before_event(_bash_event("pytest -q"), _ctx(handler, enabled=True))

    assert skipped is None
    assert asked is None
    assert len(handler.requests) == 1
    assert handler.requests[0].approval_kind == "normal"
