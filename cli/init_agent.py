from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_core import Agent
from agent_core.agent.hooks import BaseAgentHook, HookAction, HookDecision
from agent_core.agent.runtime_events import ToolCallRequested, ToolResultReceived
from agent_core.agent.tool_args import ToolArgumentsError, parse_tool_arguments_for_execution
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


@dataclass
class InitRepeatedToolCallGuardHook(BaseAgentHook):
    """Block repeated identical file-inspection calls during `/init`."""

    priority: int = 16
    watched_tool_names: tuple[str, ...] = ("read", "glob_search", "grep", "resolve_path")
    _last_signature: tuple[str, str] | None = None
    _cached_results: dict[tuple[str, str], ToolMessage] = field(default_factory=dict)

    _REUSED_RESULT_NOTICE = (
        "Notice: repeated identical `/init` file-inspection call suppressed. "
        "Reuse the cached result below instead of calling the same tool again."
    )

    async def before_event(self, event, ctx) -> HookDecision | None:
        if not isinstance(event, ToolCallRequested):
            return None

        tool_name = event.tool_call.function.name
        if tool_name not in self.watched_tool_names:
            return None

        signature = self._build_signature(event)
        if signature is None:
            return None

        if signature != self._last_signature:
            return None

        cached_result = self._cached_results.get(signature)
        if cached_result is not None:
            reused_content = cached_result.content
            if isinstance(reused_content, str):
                reused_content = f"{self._REUSED_RESULT_NOTICE}\n\n{reused_content}"
            return HookDecision(
                action=HookAction.OVERRIDE_RESULT,
                override_result=ToolMessage(
                    tool_call_id=event.tool_call.id,
                    tool_name=tool_name,
                    content=reused_content,
                    is_error=False,
                ),
                reason="reused cached result for repeated identical init tool call",
            )

        return HookDecision(
            action=HookAction.OVERRIDE_RESULT,
            override_result=ToolMessage(
                tool_call_id=event.tool_call.id,
                tool_name=tool_name,
                content=(
                    "Error: You just called the same tool with identical parameters and already "
                    "received the result.\n"
                    "Do not repeat the same file-inspection step.\n"
                    "Reuse the previous output, change the file/range/query, or move on to "
                    "`write`, `edit`, or `done` if you already have enough information."
                ),
                is_error=True,
            ),
            reason="blocked repeated identical init tool call",
        )

    async def after_event(self, event, ctx, emitted_events) -> HookDecision | None:
        del ctx, emitted_events
        if not isinstance(event, ToolResultReceived):
            return None

        tool_name = event.tool_call.function.name
        if tool_name not in self.watched_tool_names:
            self._last_signature = None
            return None

        signature = self._build_signature_from_call(event.tool_call)
        if signature is not None:
            self._last_signature = signature
            if not bool(getattr(event.tool_result, "is_error", False)):
                content = getattr(event.tool_result, "content", None)
                if not (
                    isinstance(content, str)
                    and content.startswith(self._REUSED_RESULT_NOTICE)
                ):
                    self._cached_results[signature] = event.tool_result.model_copy(deep=True)
        return None

    def _build_signature(self, event: ToolCallRequested) -> tuple[str, str] | None:
        return self._build_signature_from_call(event.tool_call)

    @staticmethod
    def _build_signature_from_call(tool_call) -> tuple[str, str] | None:
        tool_name = tool_call.function.name
        arguments = tool_call.function.arguments
        try:
            parsed = parse_tool_arguments_for_execution(arguments)
        except ToolArgumentsError:
            normalized_args = arguments.strip()
        else:
            normalized_args = json.dumps(parsed, ensure_ascii=False, sort_keys=True, default=str)
        if not normalized_args:
            return None
        return (tool_name, normalized_args)


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
        "When a tool result says `Context preview`, `trimmed`, `summary`, `Artifact file`, "
        "`Output path`, `log_path`, or similar, treat it as context-limited rather than the "
        "full raw result.\n"
        "Do not repeat the same `read`, `glob_search`, `grep`, or `resolve_path` call with "
        "identical parameters just to try to see more.\n"
        "If more file content is required, first change the read window such as `offset_line` "
        "or `n_lines`.\n"
        "If the result explicitly points to an artifact file, read that artifact with explicit "
        "`offset_line` and `n_lines` instead of repeating the same source read.\n"
        "If the result points to an output file or log path, read that file directly.\n"
        "Do not repeatedly read the same file slice with identical parameters unless you are "
        "verifying a specific detail that was not already captured.\n"
        "Once you have enough verified information to fill the required sections, stop "
        "exploring and draft the full TGAGENTS.md immediately.\n"
        "If the current tool outputs already support the required sections, do not keep reading "
        "for completeness; switch to drafting with `write` or `edit` immediately.\n"
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
        "- If a tool result shows `Context preview`, `trimmed`, `summary`, `Artifact file`, "
        "`Output path`, `log_path`, or similar, treat it as context-limited rather than the "
        "full raw result.\n"
        "- Do not repeat an identical `read`, `glob_search`, `grep`, or `resolve_path` call just "
        "to see more; change the range/query, read the referenced artifact with explicit window "
        "parameters, or read the referenced output/log file when necessary.\n"
        "- Do not repeat the same read or search with identical parameters unless it is required "
        "to verify one specific missing detail.\n"
        "- As soon as you can cover the required structure with verified facts, stop exploring "
        "and write TGAGENTS.md.\n"
        "- If the current evidence is already sufficient to draft TGAGENTS.md, start writing "
        "immediately instead of continuing to inspect more files.\n"
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
        hooks=[
            InitOutputGuardHook(workspace_root=workspace_root),
            InitRepeatedToolCallGuardHook(),
        ],
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
