from __future__ import annotations

import asyncio
import io
from datetime import datetime
from pathlib import Path

from rich.console import Console

from agent_core.agent import Agent
from agent_core.agent.runtime_events import RunFinished
from agent_core.llm.messages import ToolMessage
from agent_core.skill.manager import SkillManagementError, SkillManager
from agent_core.skill.review import (
    SKILL_REVIEW_PROMPT,
    SkillReviewRunner,
    SkillReviewHook,
    SkillReviewResult,
    _extract_skill_manage_changes,
    _extract_skill_manage_errors,
    _is_nothing_to_save,
)
from agent_core.skill.runtime_service import SkillRuntimeService
from cli.at_commands import AtCommandRegistry
from cli.skills_handler import SkillReviewHistoryItem, SkillSlashHandler
from tools.skills import skill_manage


def _skill_content(name: str, description: str = "Reusable workflow") -> str:
    return "\n".join(
        [
            "---",
            f"name: {name}",
            f"description: {description}",
            "category: Test",
            "---",
            "",
            f"# {name}",
            "",
            "Use this workflow for similar tasks.",
            "",
        ]
    )


def _write_skill(root: Path, directory: str, name: str | None = None) -> Path:
    skill_dir = root / directory
    skill_dir.mkdir(parents=True, exist_ok=True)
    path = skill_dir / "SKILL.md"
    path.write_text(_skill_content(name or directory), encoding="utf-8")
    return path


def test_skill_manager_creates_and_patches_user_skill(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    manager = SkillManager()

    created = manager.create(name="debug-flow", content=_skill_content("debug-flow"))
    assert created.path == home / ".tg_agent" / "skills" / "debug-flow" / "SKILL.md"
    assert created.path.exists()

    patched = manager.patch(
        name="debug-flow",
        old_string="Use this workflow for similar tasks.",
        new_string="Run the focused verification command before finalizing.",
    )
    assert patched.path == created.path
    assert "focused verification" in created.path.read_text(encoding="utf-8")


def test_skill_manager_rejects_builtin_root_writes(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    builtin_root = home / ".tg_agent" / "skills" / ".builtin"
    manager = SkillManager(skills_root=builtin_root)

    try:
        manager.create(name="builtin-edit", content=_skill_content("builtin-edit"))
    except SkillManagementError as exc:
        assert "non-user skill path" in str(exc)
    else:
        raise AssertionError("expected builtin write to be rejected")


def test_skill_manage_rejects_delete_and_reloads_after_create(
    tmp_path: Path,
    monkeypatch,
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    user_root = home / ".tg_agent" / "skills"
    registry = AtCommandRegistry(skill_dirs=[user_root])
    service = SkillRuntimeService(skill_registry=registry)

    deleted = asyncio.run(
        skill_manage.func(
            action="delete",
            name="demo",
            service=service,
        )
    )
    assert "delete is disabled" in deleted

    created = asyncio.run(
        skill_manage.func(
            action="create",
            name="demo",
            content=_skill_content("demo"),
            service=service,
        )
    )
    assert "Skill created: demo" in created
    assert registry.get("demo") is not None


def test_skills_slash_handler_lists_reload_and_show(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    user_root = home / ".tg_agent" / "skills"
    _write_skill(user_root, "demo")

    registry = AtCommandRegistry(skill_dirs=[user_root])
    service = SkillRuntimeService(skill_registry=registry)
    output = io.StringIO()
    console = Console(file=output, force_terminal=False, width=140)
    handler = SkillSlashHandler(service=service, console=console)

    result = asyncio.run(handler.handle([]))
    assert result.handled is True
    assert "demo" in output.getvalue()
    assert "user" in output.getvalue()

    result = asyncio.run(handler.handle(["reload"]))
    assert result.reloaded is True

    result = asyncio.run(handler.handle(["show", "demo"]))
    assert result.handled is True
    assert "可修改" in output.getvalue()


def test_skills_slash_handler_shows_review_history(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    registry = AtCommandRegistry(skill_dirs=[home / ".tg_agent" / "skills"])
    service = SkillRuntimeService(skill_registry=registry)
    history = [
        SkillReviewHistoryItem(
            created_at=datetime(2026, 4, 17, 14, 20, 11),
            status="updated",
            skill_name="demo",
            summary="已更新 skill",
        ),
        SkillReviewHistoryItem(
            created_at=datetime(2026, 4, 17, 14, 31, 45),
            status="failed",
            summary="RateLimitError: request timeout",
        ),
    ]
    output = io.StringIO()
    console = Console(file=output, force_terminal=False, width=140)
    handler = SkillSlashHandler(service=service, console=console, review_history=history)

    result = asyncio.run(handler.handle(["review"]))

    assert result.handled is True
    text = output.getvalue()
    assert "updated" in text
    assert "demo" in text
    assert "failed" in text
    assert "RateLimitError" in text


def test_skill_review_extracts_only_successful_skill_manage_changes() -> None:
    changes = _extract_skill_manage_changes(
        [
            ToolMessage(
                tool_call_id="call-1",
                tool_name="skill_manage",
                content=(
                    "Skill patched: demo\n"
                    "Path: C:\\Users\\me\\.tg_agent\\skills\\demo\\SKILL.md\n"
                    "Skills reloaded: total=1."
                ),
                is_error=False,
            ),
            ToolMessage(
                tool_call_id="call-2",
                tool_name="skill_manage",
                content="Error: delete is disabled.",
                is_error=False,
            ),
            ToolMessage(
                tool_call_id="call-3",
                tool_name="skill_view",
                content="Skill patched: ignored",
                is_error=False,
            ),
        ]
    )

    assert len(changes) == 1
    assert changes[0].action == "patched"
    assert changes[0].name == "demo"
    assert changes[0].path.endswith("demo\\SKILL.md")


def test_skill_review_extracts_skill_manage_errors() -> None:
    errors = _extract_skill_manage_errors(
        [
            ToolMessage(
                tool_call_id="call-1",
                tool_name="skill_manage",
                content="Error: old_string not found in user-level skill",
                is_error=False,
            ),
            ToolMessage(
                tool_call_id="call-2",
                tool_name="skill_manage",
                content="Tool execution failed",
                is_error=True,
            ),
            ToolMessage(
                tool_call_id="call-3",
                tool_name="skill_view",
                content="Error: ignored",
                is_error=False,
            ),
            ToolMessage(
                tool_call_id="call-4",
                tool_name="skill_manage",
                content="Skill patched: demo",
                is_error=False,
            ),
        ]
    )

    assert errors == [
        "Error: old_string not found in user-level skill",
        "Tool execution failed",
    ]


def test_skill_review_detects_exact_nothing_to_save_response() -> None:
    assert _is_nothing_to_save("Nothing to save.")
    assert _is_nothing_to_save("  Nothing to save.\n")
    assert not _is_nothing_to_save("Nothing to save")
    assert not _is_nothing_to_save("No skill was saved.")


def test_skill_review_runner_builds_hidden_review_agent_with_runtime_role(tmp_path: Path) -> None:
    registry = AtCommandRegistry(skill_dirs=[tmp_path / "skills"])
    service = SkillRuntimeService(skill_registry=registry)
    runner = SkillReviewRunner(service=service)
    main_agent = Agent(llm=object(), tools=[], system_prompt="system prompt")
    captured: dict[str, object] = {}

    async def _fake_query(self, message: str) -> str:
        captured["runtime_role"] = self.runtime_role
        captured["message"] = message
        return "Nothing to save."

    original_query = Agent.query
    Agent.query = _fake_query  # type: ignore[assignment]
    try:
        result = asyncio.run(runner.run(main_agent, []))
    finally:
        Agent.query = original_query  # type: ignore[assignment]

    assert result.final_response == "Nothing to save."
    assert captured["runtime_role"] == "skill_review"
    assert captured["message"] == SKILL_REVIEW_PROMPT


def test_skill_review_hook_only_triggers_for_primary_agents() -> None:
    hook = SkillReviewHook(runner=object(), interval=1)  # type: ignore[arg-type]
    hook._iters_since_skill = 1  # noqa: SLF001
    event = RunFinished(final_response="done", iterations=1)

    primary_agent = Agent(llm=object(), tools=[skill_manage], runtime_role="primary")
    subagent = Agent(llm=object(), tools=[skill_manage], runtime_role="subagent")
    review_agent = Agent(llm=object(), tools=[skill_manage], runtime_role="skill_review")

    assert hook._should_trigger(event, primary_agent) is True  # noqa: SLF001
    assert hook._should_trigger(event, subagent) is False  # noqa: SLF001
    assert hook._should_trigger(event, review_agent) is False  # noqa: SLF001


def test_skill_review_hook_reports_background_errors() -> None:
    class FailingTask:
        def __init__(self, error: Exception) -> None:
            self.error = error

        def result(self):
            raise self.error

    captured: list[Exception] = []
    hook = SkillReviewHook(
        runner=object(),  # type: ignore[arg-type]
        on_error=captured.append,
    )
    error = RuntimeError("boom")

    hook._handle_background_result(FailingTask(error))  # type: ignore[arg-type]  # noqa: SLF001

    assert captured == [error]


def test_skill_review_hook_times_out_background_review() -> None:
    class SlowRunner:
        async def run(self, agent, snapshot):
            del agent, snapshot
            await asyncio.sleep(0.05)
            return SkillReviewResult(final_response="Nothing to save.")

    hook = SkillReviewHook(
        runner=SlowRunner(),  # type: ignore[arg-type]
        timeout_seconds=0.001,
    )

    try:
        asyncio.run(
            hook._run_review_with_timeout(  # type: ignore[arg-type]  # noqa: SLF001
                object(),
                [],
            )
        )
    except asyncio.TimeoutError:
        pass
    else:
        raise AssertionError("expected skill review timeout")


def test_skill_review_hook_reports_skill_manage_errors() -> None:
    class CompletedTask:
        def result(self):
            return SkillReviewResult(
                final_response="I tried to update the skill but patch failed.",
                manage_errors=["Error: old_string not found in user-level skill"],
            )

    captured: list[list[str]] = []
    hook = SkillReviewHook(
        runner=object(),  # type: ignore[arg-type]
        on_manage_errors=captured.append,
    )

    hook._handle_background_result(CompletedTask())  # type: ignore[arg-type]  # noqa: SLF001

    assert captured == [["Error: old_string not found in user-level skill"]]


def test_skill_review_hook_reports_unclassified_no_change() -> None:
    class CompletedTask:
        def result(self):
            return SkillReviewResult(
                final_response="No reusable workflow was found.",
            )

    captured: list[str] = []
    hook = SkillReviewHook(
        runner=object(),  # type: ignore[arg-type]
        on_unclassified_no_change=captured.append,
    )

    hook._handle_background_result(CompletedTask())  # type: ignore[arg-type]  # noqa: SLF001

    assert captured == ["No reusable workflow was found."]
