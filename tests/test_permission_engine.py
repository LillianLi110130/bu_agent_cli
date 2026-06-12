from __future__ import annotations

import json
from types import SimpleNamespace

from agent_core.agent.permissions import PermissionEngine
from agent_core.agent.runtime_events import ToolCallRequested
from agent_core.llm.messages import Function, ToolCall
from tools.sandbox import SandboxContext


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


def _ctx_with_sandbox(root) -> SimpleNamespace:
    return SimpleNamespace(
        agent=SimpleNamespace(
            _sandbox_context=SandboxContext.create(root),
        )
    )


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


def test_permission_engine_denies_bash_file_read() -> None:
    decision = PermissionEngine().evaluate_tool_call(
        _bash_event("cat README.md"),
        SimpleNamespace(),
    )

    assert decision.decision == "deny"
    assert decision.approval_request is None
    assert decision.reasons[0].source == "bash_file_task"
    assert decision.reasons[0].rule_id == "file_read"
    assert decision.reasons[0].guidance is not None
    assert "`read`" in decision.reasons[0].guidance


def test_permission_engine_denies_bash_file_discovery() -> None:
    decision = PermissionEngine().evaluate_tool_call(
        _bash_event("ls"),
        SimpleNamespace(),
    )

    assert decision.decision == "deny"
    assert decision.reasons[0].source == "bash_file_task"
    assert decision.reasons[0].rule_id == "file_discovery"
    assert decision.reasons[0].guidance is not None
    assert "`glob_search`" in decision.reasons[0].guidance


def test_permission_engine_denies_bash_text_search() -> None:
    decision = PermissionEngine().evaluate_tool_call(
        _bash_event("rg TODO"),
        SimpleNamespace(),
    )

    assert decision.decision == "deny"
    assert decision.reasons[0].source == "bash_file_task"
    assert decision.reasons[0].rule_id == "text_search"
    assert decision.reasons[0].guidance is not None
    assert "`grep`" in decision.reasons[0].guidance


def test_permission_engine_denies_plain_find_as_bash_file_task() -> None:
    decision = PermissionEngine().evaluate_tool_call(
        _bash_event('find . -name "*.py"'),
        SimpleNamespace(),
    )

    assert decision.decision == "deny"
    assert decision.reasons[0].source == "bash_file_task"
    assert decision.reasons[0].rule_id == "text_search"


def test_permission_engine_keeps_destructive_find_in_command_safety_only() -> None:
    engine = PermissionEngine()
    cases = [
        "find . -name '*.tmp' -delete",
        "find . -name '*.tmp' -exec rm {} \\;",
    ]

    for command in cases:
        decision = engine.evaluate_tool_call(_bash_event(command), SimpleNamespace())
        assert decision.decision == "ask"
        assert decision.approval_request is not None
        assert {reason.source for reason in decision.reasons} == {"command_safety"}
        assert {reason.rule_id for reason in decision.reasons} == {"find_delete"}


def test_permission_engine_keeps_hidden_execution_in_command_safety_only() -> None:
    engine = PermissionEngine()
    cases = [
        'bash -c "cat README.md"',
        'python -c "open(\\"README.md\\").read()"',
    ]

    for command in cases:
        decision = engine.evaluate_tool_call(_bash_event(command), SimpleNamespace())
        assert decision.decision == "ask"
        assert decision.approval_request is not None
        assert {reason.source for reason in decision.reasons} == {"command_safety"}


def test_permission_engine_allows_safe_git_read_commands() -> None:
    engine = PermissionEngine()

    cases = [
        "git show HEAD",
        "git show --stat HEAD",
        "git show HEAD --stat",
        "git show --name-only HEAD",
        "git show HEAD -- agent_core/agent/permissions.py",
        "git show HEAD:agent_core/agent/permissions.py",
        'git grep "PermissionEngine" HEAD -- agent_core/',
        'git grep "PermissionEngine" -n HEAD -- agent_core/',
        "git ls-files agent_core/",
        "git diff -- agent_core/agent/permissions.py",
        "git show HEAD --stat 2>&1",
    ]
    for command in cases:
        decision = engine.evaluate_tool_call(_bash_event(command), SimpleNamespace())
        assert decision.decision == "allow"
        assert decision.reasons == ()


def test_permission_engine_allows_safe_cd_git_read_commands(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "agent_core").mkdir()
    ctx = _ctx_with_sandbox(workspace)
    engine = PermissionEngine()

    cases = [
        "cd agent_core && git show HEAD -- agent/permissions.py",
        f"cd /d {workspace} && git show HEAD",
    ]
    for command in cases:
        decision = engine.evaluate_tool_call(_bash_event(command), ctx)
        assert decision.decision == "allow"
        assert decision.reasons == ()


def test_permission_engine_allows_safe_git_read_pipeline_filters() -> None:
    engine = PermissionEngine()
    cases = [
        'git diff -- agent_core/agent/permissions.py | grep PermissionEngine',
        'git diff -- agent_core/agent/permissions.py | grep -n "PermissionEngine"',
        "git diff -- agent_core/agent/permissions.py | findstr PermissionEngine",
    ]
    for command in cases:
        decision = engine.evaluate_tool_call(_bash_event(command), SimpleNamespace())
        assert decision.decision == "allow"
        assert decision.reasons == ()


def test_permission_engine_denies_unsafe_git_read_commands(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "agent_core").mkdir()
    ctx = _ctx_with_sandbox(workspace)
    engine = PermissionEngine()

    cases = [
        r"git -C C:\outside show HEAD",
        r"git --git-dir C:\outside\.git show HEAD",
        r"git --work-tree C:\outside show HEAD",
        "git show HEAD -- ../secret",
        r"git show HEAD:C:\secret",
        "git show HEAD:../secret",
        "cd ../outside && git show HEAD",
        "git show HEAD > out.txt",
        "git diff -- agent_core/agent/permissions.py > out.txt",
        "git show HEAD 2> err.txt",
        "git show HEAD | tee out.txt",
        "git diff -- agent_core/agent/permissions.py | tee out.txt",
        "git show HEAD | grep TODO /etc/passwd",
        "git show HEAD | grep -f patterns.txt",
        "git show HEAD | grep -r TODO",
        "git show HEAD | grep TODO | wc -l",
        "cd agent_core && git show HEAD && cat secret",
    ]
    for command in cases:
        decision = engine.evaluate_tool_call(_bash_event(command), ctx)
        assert decision.decision == "deny", command
        assert any(reason.source == "bash_file_task" for reason in decision.reasons)


def test_permission_engine_denies_shell_task_log_read_with_task_output_guidance() -> None:
    decision = PermissionEngine().evaluate_tool_call(
        _bash_event("cat /repo/.tg_agent/shell_tasks/22b62dbb/58983d6c.log"),
        SimpleNamespace(),
    )

    assert decision.decision == "deny"
    assert decision.reasons[0].source == "bash_file_task"
    assert decision.reasons[0].rule_id == "shell_task_log_read"
    assert decision.reasons[0].guidance is not None
    assert 'task_output(task_id="58983d6c"' in decision.reasons[0].guidance


def test_permission_engine_prefers_file_task_deny_over_safety_ask() -> None:
    decision = PermissionEngine().evaluate_tool_call(
        _bash_event("rm -rf build; cat README.md"),
        SimpleNamespace(),
    )

    assert decision.decision == "deny"
    assert decision.approval_request is None
    assert any(
        reason.source == "command_safety" and reason.rule_id == "rm_recursive"
        for reason in decision.reasons
    )
    assert any(
        reason.source == "bash_file_task" and reason.rule_id == "file_read"
        for reason in decision.reasons
    )
