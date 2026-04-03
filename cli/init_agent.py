from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_core import Agent
from agent_core.agent.hooks import BaseAgentHook, HookAction, HookDecision
from agent_core.agent.runtime_events import ToolCallRequested
from agent_core.llm.base import BaseChatModel
from agent_core.llm.messages import ToolMessage
from tools import done, edit, glob_search, grep, read, resolve_path, write

INIT_OUTPUT_FILENAME = "TGAGENTS.md"


@dataclass
class InitOutputGuardHook(BaseAgentHook):
    """Block `done` until `/init` has produced a non-empty TGAGENTS.md."""

    workspace_root: Path = Path(".")
    priority: int = 15

    async def before_event(self, event, ctx) -> HookDecision | None:
        if not isinstance(event, ToolCallRequested):
            return None

        if event.tool_call.function.name != "done":
            return None

        ok, error = validate_init_output(self.workspace_root)
        if ok:
            return None

        return HookDecision(
            action=HookAction.OVERRIDE_RESULT,
            override_result=ToolMessage(
                tool_call_id=event.tool_call.id,
                tool_name="done",
                content=(
                    f"{error}\n"
                    "You must create or update TGAGENTS.md at the workspace root with `write` "
                    "or `edit` before calling `done`."
                ),
                is_error=True,
            ),
            reason="blocked init completion before TGAGENTS.md exists",
        )


def build_init_tools() -> list[Any]:
    """Return the restricted tool set used by `/init`."""
    return [
        resolve_path,
        glob_search,
        grep,
        read,
        write,
        edit,
        done,
    ]


def build_init_system_prompt() -> str:
    """Build the dedicated system prompt for `/init`."""
    return (
        "You are executing the `/init` command for this repository.\n"
        "Your only goal is to generate or update a repository-specific TGAGENTS.md at the "
        "workspace root.\n"
        "You must use the available tools to inspect the repository and verify your claims.\n"
        "Only modify TGAGENTS.md. Do not change any other file.\n"
        "Do not use shell commands.\n"
        "If TGAGENTS.md already exists, read it first and update it instead of blindly "
        "rewriting it.\n"
        "If information cannot be verified from tool outputs, mark it as 未验证 or 推测.\n"
        "Prefer high-signal files first, such as README, pyproject.toml, existing docs, and "
        "project overview documents when they exist.\n"
        "Do not repeatedly read the same file slice with identical parameters unless you are "
        "verifying a specific detail that was not already captured.\n"
        "Once you have enough verified information to fill the required sections, stop "
        "exploring and draft the full TGAGENTS.md immediately.\n"
        "Before each additional read or search, ask whether it is necessary to complete a "
        "missing section. If not, move on to drafting or editing TGAGENTS.md.\n"
        "Execution procedure:\n"
        "1. Check whether TGAGENTS.md already exists at the workspace root.\n"
        "2. Inspect only the minimum set of files needed to verify the required sections.\n"
        "3. Reuse high-signal overview documents when available instead of exhaustively reading "
        "many files.\n"
        "4. Draft the full TGAGENTS.md content as soon as the required sections can be "
        "supported.\n"
        "5. Use `write` or `edit` to persist TGAGENTS.md.\n"
        "6. Only after the file exists and is non-empty may you call `done`.\n"
        "When the file is complete, Call the done tool with a short completion summary."
    )


def build_init_user_prompt(workspace_root: Path) -> str:
    """Build the task prompt for the dedicated `/init` agent."""
    output_path = workspace_root / INIT_OUTPUT_FILENAME
    return (
        f"Analyze the repository rooted at {workspace_root} and generate or update "
        f"{output_path}.\n\n"
        "The document must be written in Chinese and help future coding and analysis sessions work "
        "efficiently in this repository.\n\n"
        "Required structure:\n"
        "1. 项目目标\n"
        "2. 目录与职责\n"
        "3. 推荐阅读路径\n"
        "4. 关键入口与核心链路\n"
        "5. 可暂时忽略的内容\n"
        "6. 开发与验证命令\n"
        "7. 约束与假设\n\n"
        "Requirements:\n"
        "- Explore the repository with tools instead of guessing.\n"
        "- Only include claims supported by files or search results you actually inspected.\n"
        "- Favor concise, actionable guidance over generic introduction text.\n"
        "- If a section cannot be verified, say so explicitly.\n"
        "- Prefer high-signal files first, especially README, pyproject.toml, and overview docs.\n"
        "- Do not repeat the same read or search with identical parameters unless it is required "
        "to verify one specific missing detail.\n"
        "- As soon as you can cover the required structure with verified facts, stop exploring "
        "and write TGAGENTS.md.\n"
        "- Only modify TGAGENTS.md.\n"
        "- You must use write or edit to persist TGAGENTS.md before finishing.\n"
        "- Do not call done until TGAGENTS.md exists and is non-empty.\n"
        "- Call the done tool after the file is written."
    )


def build_init_agent(
    *,
    llm: BaseChatModel,
    workspace_root: Path,
    dependency_overrides: dict | None = None,
    max_iterations: int = 24,
) -> Agent:
    """Create the temporary agent used by `/init`."""
    return Agent(
        llm=llm,
        tools=build_init_tools(),
        system_prompt=build_init_system_prompt(),
        dependency_overrides=dependency_overrides,
        max_iterations=max_iterations,
        tool_choice="required",
        require_done_tool=True,
        hooks=[InitOutputGuardHook(workspace_root=workspace_root)],
    )


def validate_init_output(workspace_root: Path) -> tuple[bool, str | None]:
    """Validate the expected `/init` output file."""
    output_path = workspace_root / INIT_OUTPUT_FILENAME
    if not output_path.exists():
        return False, f"`/init` 未成功生成 {INIT_OUTPUT_FILENAME}。"

    content = output_path.read_text(encoding="utf-8").strip()
    if not content:
        return False, f"`/init` 生成了空的 {INIT_OUTPUT_FILENAME}。"

    return True, None
