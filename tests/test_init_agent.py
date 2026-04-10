from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from cli.init_agent import (
    InitDraftBeforeMoreInspectionHook,
    InitOutputGuardHook,
    InitRepeatedToolCallGuardHook,
    InitWriteTargetGuardHook,
    build_init_agent,
    build_init_system_prompt,
    build_init_tools,
    build_init_user_prompt,
    validate_init_output,
)
from agent_core.agent.runtime_events import ToolCallRequested, ToolResultReceived
from agent_core.llm.messages import Function, ToolCall, ToolMessage


class DummyLLM:
    def __init__(self, model: str = "dummy-model") -> None:
        self.model = model

    @property
    def provider(self) -> str:
        return "dummy"

    @property
    def name(self) -> str:
        return self.model

    async def ainvoke(self, messages, tools=None, tool_choice=None, **kwargs):
        raise NotImplementedError

    async def astream(self, messages, tools=None, tool_choice=None, **kwargs):
        if False:
            yield None


def _valid_init_document() -> str:
    return """# Repository Guidelines

## Project Structure & Module Organization
源代码集中在 `src/` 与 `cli/`，测试位于 `tests/`，构建脚本与仓库级配置放在根目录。新增模块时保持目录职责单一，并优先沿用现有命名模式与路径布局。

## Build, Test, and Development Commands
使用 `pytest -q` 运行 Python 测试，使用 `ruff check .` 做静态检查，使用 `black .` 做格式化。若有本地启动命令，应直接写出完整命令和作用，避免只写“运行项目”。

## Coding Style & Naming Conventions
Python 代码使用 4 空格缩进，模块与函数采用 `snake_case`，类采用 `PascalCase`。提交前保持 lint 与 format 结果干净，新增配置项时同步更新文档与示例。

## Testing Guidelines
测试文件命名保持 `test_*.py`，优先覆盖关键流程、边界条件和失败路径。若某部分暂未验证，应在文档中明确写出“未验证”而不是省略。

## Commit & Pull Request Guidelines
提交信息使用简短祈使句，Pull Request 需要说明改动影响、验证命令与相关 issue。涉及界面或交互变化时附上截图或终端输出示例。

## Security & Configuration Tips
敏感配置通过环境变量或 `.env` 管理，不提交真实密钥。新增配置前先检查 README、部署脚本和默认配置是否需要同步更新。
"""


def test_build_init_tools_uses_restricted_tool_whitelist() -> None:
    tool_names = [tool.name for tool in build_init_tools()]

    assert tool_names == [
        "resolve_path",
        "glob_search",
        "grep",
        "read",
        "write",
        "edit",
        "done",
    ]


def test_build_init_agent_requires_done_and_uses_restricted_tools() -> None:
    agent = build_init_agent(llm=DummyLLM(), workspace_root=Path("."))

    assert agent.require_done_tool is True
    assert agent.tool_choice == "auto"
    assert agent.max_iterations == 40
    assert agent.compaction is not None
    assert agent.compaction.enabled is False
    assert agent._context.sliding_window_messages is None
    assert [tool.name for tool in agent.tools] == [
        "resolve_path",
        "glob_search",
        "grep",
        "read",
        "write",
        "edit",
        "done",
    ]
    assert len(agent.hooks) == 4
    assert isinstance(agent.hooks[0], InitWriteTargetGuardHook)
    assert isinstance(agent.hooks[1], InitDraftBeforeMoreInspectionHook)
    assert isinstance(agent.hooks[2], InitOutputGuardHook)
    assert isinstance(agent.hooks[3], InitRepeatedToolCallGuardHook)


def test_init_prompts_contain_expected_constraints(tmp_path) -> None:
    system_prompt = build_init_system_prompt()
    user_prompt = build_init_user_prompt(tmp_path)

    assert "Only modify TGAGENTS.md" in system_prompt
    assert "Do not use shell commands" in system_prompt
    assert "Only include claims supported by files or search results you actually inspected" in system_prompt
    assert "Prefer high-signal files first" in system_prompt
    assert "Do not repeatedly read the same file slice" in system_prompt
    assert "After a small number of high-signal inspections" in system_prompt
    assert "write them as 未验证" in system_prompt
    assert "write a concise contributor guide" in system_prompt
    assert "Only after the file passes validation may you call `done`" in system_prompt
    assert str(tmp_path) in user_prompt
    assert "TGAGENTS.md" in user_prompt
    assert 'Title the document "Repository Guidelines"' in user_prompt
    assert "Write the document in Chinese" in user_prompt
    assert "Adapt the outline as needed" in user_prompt
    assert "Project Structure & Module Organization" in user_prompt
    assert "Commit & Pull Request Guidelines" in user_prompt
    assert "Optional: Agent-Specific Instructions" in user_prompt
    assert "Call the done tool after TGAGENTS.md passes validation" in user_prompt


def test_validate_init_output_checks_document_structure(tmp_path) -> None:
    ok, error = validate_init_output(tmp_path)
    assert ok is False
    assert "TGAGENTS.md" in (error or "")

    agents_md = tmp_path / "TGAGENTS.md"
    agents_md.write_text("", encoding="utf-8")
    ok, error = validate_init_output(tmp_path)
    assert ok is False
    assert "空" in (error or "")

    agents_md.write_text("# Wrong Title\n\n## Section\ncontent", encoding="utf-8")
    ok, error = validate_init_output(tmp_path)
    assert ok is False
    assert "Repository Guidelines" in (error or "")

    agents_md.write_text(_valid_init_document(), encoding="utf-8")
    ok, error = validate_init_output(tmp_path)
    assert ok is True
    assert error is None


def test_validate_init_output_rejects_short_or_placeholder_drafts(tmp_path) -> None:
    agents_md = tmp_path / "TGAGENTS.md"
    agents_md.write_text(
        "# Repository Guidelines\n\n## Project Structure\nTODO\n\n## Testing\nTODO\n\n## Commands\nTODO\n\n## PR\nTODO\n",
        encoding="utf-8",
    )
    ok, error = validate_init_output(tmp_path)
    assert ok is False
    assert "占位" in (error or "") or "过短" in (error or "")


@pytest.mark.asyncio
async def test_init_output_guard_hook_blocks_done_before_file_exists(tmp_path) -> None:
    hook = InitOutputGuardHook(workspace_root=tmp_path)
    event = ToolCallRequested(
        tool_call=ToolCall(
            id="done-1",
            function=Function(name="done", arguments='{"message":"finished"}'),
        ),
        iteration=1,
    )

    decision = await hook.before_event(event, SimpleNamespace())

    assert decision is not None
    assert decision.action == "override_result"
    assert decision.override_result is not None
    assert "TGAGENTS.md" in decision.override_result.content


@pytest.mark.asyncio
async def test_init_output_guard_hook_allows_done_after_file_exists(tmp_path) -> None:
    hook = InitOutputGuardHook(workspace_root=tmp_path)
    (tmp_path / "TGAGENTS.md").write_text(_valid_init_document(), encoding="utf-8")
    event = ToolCallRequested(
        tool_call=ToolCall(
            id="done-1",
            function=Function(name="done", arguments='{"message":"finished"}'),
        ),
        iteration=1,
    )

    decision = await hook.before_event(event, SimpleNamespace())

    assert decision is None


@pytest.mark.asyncio
async def test_init_write_target_guard_blocks_non_tgagents_file(tmp_path) -> None:
    hook = InitWriteTargetGuardHook(workspace_root=tmp_path)
    event = ToolCallRequested(
        tool_call=ToolCall(
            id="write-1",
            function=Function(
                name="write",
                arguments='{"file_path":"README.md","content":"bad target"}',
            ),
        ),
        iteration=1,
    )

    decision = await hook.before_event(event, SimpleNamespace())

    assert decision is not None
    assert decision.action == "override_result"
    assert decision.override_result is not None
    assert "may only modify TGAGENTS.md" in decision.override_result.content


@pytest.mark.asyncio
async def test_init_write_target_guard_allows_tgagents_file(tmp_path) -> None:
    hook = InitWriteTargetGuardHook(workspace_root=tmp_path)
    event = ToolCallRequested(
        tool_call=ToolCall(
            id="write-1",
            function=Function(
                name="write",
                arguments='{"file_path":"TGAGENTS.md","content":"# Repository Guidelines"}',
            ),
        ),
        iteration=1,
    )

    decision = await hook.before_event(event, SimpleNamespace())

    assert decision is None


@pytest.mark.asyncio
async def test_init_draft_before_more_inspection_hook_blocks_thirteenth_inspection_before_draft(
    tmp_path,
) -> None:
    hook = InitDraftBeforeMoreInspectionHook(workspace_root=tmp_path, draft_required_after=12)

    for iteration in range(1, 13):
        decision = await hook.before_event(
            ToolCallRequested(
                tool_call=ToolCall(
                    id=f"read-{iteration}",
                    function=Function(
                        name="read",
                        arguments=f'{{"file_path":"docs/file_{iteration}.md","n_lines":50}}',
                    ),
                ),
                iteration=iteration,
            ),
            SimpleNamespace(),
        )
        assert decision is None

    blocked_decision = await hook.before_event(
        ToolCallRequested(
            tool_call=ToolCall(
                id="read-13",
                function=Function(
                    name="read",
                    arguments='{"file_path":"docs/file_13.md","n_lines":50}',
                ),
            ),
            iteration=13,
        ),
        SimpleNamespace(),
    )

    assert blocked_decision is not None
    assert blocked_decision.override_result.is_error is True
    assert "done enough repository inspection for the first draft" in (
        blocked_decision.override_result.content
    )


@pytest.mark.asyncio
async def test_init_draft_before_more_inspection_hook_allows_more_inspection_after_write(
    tmp_path,
) -> None:
    hook = InitDraftBeforeMoreInspectionHook(workspace_root=tmp_path, draft_required_after=1)

    first_inspection = await hook.before_event(
        ToolCallRequested(
            tool_call=ToolCall(
                id="read-1",
                function=Function(
                    name="read",
                    arguments='{"file_path":"README.md","n_lines":50}',
                ),
            ),
            iteration=1,
        ),
        SimpleNamespace(),
    )
    assert first_inspection is None

    blocked_inspection = await hook.before_event(
        ToolCallRequested(
            tool_call=ToolCall(
                id="read-2",
                function=Function(
                    name="read",
                    arguments='{"file_path":"pyproject.toml","n_lines":50}',
                ),
            ),
            iteration=2,
        ),
        SimpleNamespace(),
    )
    assert blocked_inspection is not None
    assert blocked_inspection.override_result.is_error is True

    write_decision = await hook.before_event(
        ToolCallRequested(
            tool_call=ToolCall(
                id="write-1",
                function=Function(
                    name="write",
                    arguments='{"file_path":"TGAGENTS.md","content":"# draft"}',
                ),
            ),
            iteration=3,
        ),
        SimpleNamespace(),
    )
    assert write_decision is None

    resumed_inspection = await hook.before_event(
        ToolCallRequested(
            tool_call=ToolCall(
                id="read-3",
                function=Function(
                    name="read",
                    arguments='{"file_path":"cli/app.py","n_lines":50}',
                ),
            ),
            iteration=4,
        ),
        SimpleNamespace(),
    )
    assert resumed_inspection is None


@pytest.mark.asyncio
async def test_init_draft_before_more_inspection_hook_skips_guard_when_output_exists(
    tmp_path,
) -> None:
    (tmp_path / "TGAGENTS.md").write_text(_valid_init_document(), encoding="utf-8")
    hook = InitDraftBeforeMoreInspectionHook(workspace_root=tmp_path, draft_required_after=1)

    first_inspection = await hook.before_event(
        ToolCallRequested(
            tool_call=ToolCall(
                id="read-1",
                function=Function(
                    name="read",
                    arguments='{"file_path":"README.md","n_lines":50}',
                ),
            ),
            iteration=1,
        ),
        SimpleNamespace(),
    )
    assert first_inspection is None

    second_inspection = await hook.before_event(
        ToolCallRequested(
            tool_call=ToolCall(
                id="read-2",
                function=Function(
                    name="read",
                    arguments='{"file_path":"cli/app.py","n_lines":50}',
                ),
            ),
            iteration=2,
        ),
        SimpleNamespace(),
    )
    assert second_inspection is None


@pytest.mark.asyncio
async def test_init_repeated_tool_call_guard_blocks_identical_read_call(tmp_path) -> None:
    del tmp_path
    hook = InitRepeatedToolCallGuardHook()
    tool_call = ToolCall(
        id="read-1",
        function=Function(
            name="read",
            arguments='{"file_path":"docs/guide.md","offset_line":100,"n_lines":20}',
        ),
    )

    first_decision = await hook.before_event(ToolCallRequested(tool_call=tool_call, iteration=1), SimpleNamespace())
    assert first_decision is None

    await hook.after_event(
        ToolResultReceived(
            tool_call=tool_call,
            tool_result=ToolMessage(
                tool_call_id="read-1",
                tool_name="read",
                content="[Lines 100-119 of 200]",
                is_error=False,
            ),
            iteration=1,
        ),
        SimpleNamespace(),
        [],
    )

    repeated_call = ToolCall(
        id="read-2",
        function=Function(
            name="read",
            arguments='{"file_path":"docs/guide.md","offset_line":100,"n_lines":20}',
        ),
    )
    repeated_decision = await hook.before_event(
        ToolCallRequested(tool_call=repeated_call, iteration=2),
        SimpleNamespace(),
    )

    assert repeated_decision is not None
    assert repeated_decision.action == "override_result"
    assert repeated_decision.override_result.is_error is False
    assert "repeated identical `/init` file-inspection call suppressed" in repeated_decision.override_result.content
    assert "[Lines 100-119 of 200]" in repeated_decision.override_result.content


@pytest.mark.asyncio
async def test_init_repeated_tool_call_guard_blocks_non_consecutive_identical_call(tmp_path) -> None:
    del tmp_path
    hook = InitRepeatedToolCallGuardHook()
    read_call = ToolCall(
        id="read-1",
        function=Function(
            name="read",
            arguments='{"file_path":"docs/guide.md","offset_line":100,"n_lines":20}',
        ),
    )
    grep_call = ToolCall(
        id="grep-1",
        function=Function(
            name="grep",
            arguments='{"pattern":"init","path":"docs"}',
        ),
    )

    await hook.after_event(
        ToolResultReceived(
            tool_call=read_call,
            tool_result=ToolMessage(
                tool_call_id="read-1",
                tool_name="read",
                content="[Lines 100-119 of 200]",
                is_error=False,
            ),
            iteration=1,
        ),
        SimpleNamespace(),
        [],
    )
    await hook.after_event(
        ToolResultReceived(
            tool_call=grep_call,
            tool_result=ToolMessage(
                tool_call_id="grep-1",
                tool_name="grep",
                content="docs/guide.md:10: init",
                is_error=False,
            ),
            iteration=2,
        ),
        SimpleNamespace(),
        [],
    )

    repeated_decision = await hook.before_event(
        ToolCallRequested(
            tool_call=ToolCall(
                id="read-2",
                function=Function(
                    name="read",
                    arguments='{"file_path":"docs/guide.md","offset_line":100,"n_lines":20}',
                ),
            ),
            iteration=3,
        ),
        SimpleNamespace(),
    )

    assert repeated_decision is not None
    assert repeated_decision.override_result.is_error is False
    assert "repeated identical `/init` file-inspection call suppressed" in repeated_decision.override_result.content


@pytest.mark.asyncio
async def test_init_repeated_tool_call_guard_does_not_stack_reuse_notice(tmp_path) -> None:
    del tmp_path
    hook = InitRepeatedToolCallGuardHook()
    tool_call = ToolCall(
        id="read-1",
        function=Function(
            name="read",
            arguments='{"file_path":"docs/guide.md","offset_line":100,"n_lines":20}',
        ),
    )

    await hook.after_event(
        ToolResultReceived(
            tool_call=tool_call,
            tool_result=ToolMessage(
                tool_call_id="read-1",
                tool_name="read",
                content="[Lines 100-119 of 200]",
                is_error=False,
            ),
            iteration=1,
        ),
        SimpleNamespace(),
        [],
    )

    repeated_call = ToolCall(
        id="read-2",
        function=Function(
            name="read",
            arguments='{"file_path":"docs/guide.md","offset_line":100,"n_lines":20}',
        ),
    )
    repeated_decision = await hook.before_event(
        ToolCallRequested(tool_call=repeated_call, iteration=2),
        SimpleNamespace(),
    )
    await hook.after_event(
        ToolResultReceived(
            tool_call=repeated_call,
            tool_result=repeated_decision.override_result,
            iteration=2,
        ),
        SimpleNamespace(),
        [],
    )

    repeated_again = ToolCall(
        id="read-3",
        function=Function(
            name="read",
            arguments='{"file_path":"docs/guide.md","offset_line":100,"n_lines":20}',
        ),
    )
    repeated_again_decision = await hook.before_event(
        ToolCallRequested(tool_call=repeated_again, iteration=3),
        SimpleNamespace(),
    )

    assert repeated_again_decision is not None
    content = repeated_again_decision.override_result.content
    assert isinstance(content, str)
    assert content.count("repeated identical `/init` file-inspection call suppressed") == 1


@pytest.mark.asyncio
async def test_init_repeated_tool_call_guard_allows_changed_read_range(tmp_path) -> None:
    del tmp_path
    hook = InitRepeatedToolCallGuardHook()
    first_call = ToolCall(
        id="read-1",
        function=Function(
            name="read",
            arguments='{"file_path":"docs/guide.md","offset_line":100,"n_lines":20}',
        ),
    )

    await hook.after_event(
        ToolResultReceived(
            tool_call=first_call,
            tool_result=ToolMessage(
                tool_call_id="read-1",
                tool_name="read",
                content="[Lines 100-119 of 200]",
                is_error=False,
            ),
            iteration=1,
        ),
        SimpleNamespace(),
        [],
    )

    changed_call = ToolCall(
        id="read-2",
        function=Function(
            name="read",
            arguments='{"file_path":"docs/guide.md","offset_line":120,"n_lines":20}',
        ),
    )

    decision = await hook.before_event(
        ToolCallRequested(tool_call=changed_call, iteration=2),
        SimpleNamespace(),
    )

    assert decision is None
