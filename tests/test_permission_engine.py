from __future__ import annotations

import json
from types import SimpleNamespace

from agent_core.agent.permissions import PermissionEngine
from agent_core.agent.runtime_events import ToolCallRequested
from agent_core.llm.messages import Function, ToolCall


def _event(tool_name: str, arguments: dict) -> ToolCallRequested:
    return ToolCallRequested(
        tool_call=ToolCall(
            id="call-1",
            function=Function(name=tool_name, arguments=json.dumps(arguments)),
        ),
        iteration=1,
    )


def _bash_event(command: str) -> ToolCallRequested:
    return _event("bash", {"command": command})


def test_permission_engine_allows_non_bash_tool() -> None:
    decision = PermissionEngine().evaluate_tool_call(
        _event("read", {"file_path": "README.md"}),
        SimpleNamespace(),
    )

    assert decision.decision == "allow"
    assert decision.reasons == ()
    assert decision.approval_request is None


def test_permission_engine_allows_safe_bash_command() -> None:
    decision = PermissionEngine().evaluate_tool_call(
        _bash_event("pytest -q"),
        SimpleNamespace(),
    )

    assert decision.decision == "allow"
    assert decision.reasons == ()
    assert decision.approval_request is None


def test_permission_engine_asks_for_ask_level_bash_command() -> None:
    decision = PermissionEngine().evaluate_tool_call(
        _bash_event("rm -rf build"),
        SimpleNamespace(),
    )

    assert decision.decision == "ask"
    assert decision.approval_request is not None
    assert decision.approval_request.approval_kind == "safety"
    assert decision.approval_request.approval_keys == ("safety:rm_recursive",)
    assert decision.approval_request.session_approval_label == "rm 递归删除"
    assert decision.reasons[0].source == "command_safety"
    assert decision.reasons[0].rule_id == "rm_recursive"


def test_permission_engine_denies_block_level_bash_command() -> None:
    decision = PermissionEngine().evaluate_tool_call(
        _bash_event("git reset --hard"),
        SimpleNamespace(),
    )

    assert decision.decision == "deny"
    assert decision.approval_request is None
    assert decision.reasons[0].source == "command_safety"
    assert decision.reasons[0].rule_id == "git_reset_hard"


def test_permission_engine_denies_broad_recursive_delete() -> None:
    decision = PermissionEngine().evaluate_tool_call(
        _bash_event("rm -rf /"),
        SimpleNamespace(),
    )

    assert decision.decision == "deny"
    assert any(reason.rule_id == "rm_recursive_broad_scope" for reason in decision.reasons)
