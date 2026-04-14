from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_core import Agent
from agent_core.agent.compaction import CompactionConfig
from agent_core.agent.hooks import BaseAgentHook, HookAction, HookDecision
from agent_core.agent.runtime_events import ToolCallRequested, ToolResultReceived
from agent_core.agent.tool_args import ToolArgumentsError, parse_tool_arguments_for_execution
from agent_core.llm.base import BaseChatModel
from agent_core.llm.messages import ToolMessage
from tools import done, edit, glob_search, grep, read, resolve_path, write

INIT_OUTPUT_FILENAME = "TGAGENTS.md"
INIT_DOCUMENT_TITLE = "仓库指南"
_PLACEHOLDER_PATTERNS = (
    re.compile(r"(?<![A-Za-z0-9_.-])todo(?![A-Za-z0-9_-])", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z0-9_.-])tbd(?![A-Za-z0-9_-])", re.IGNORECASE),
    re.compile(r"待补充"),
    re.compile(r"(?<![A-Za-z0-9_.-])coming\s+soon(?![A-Za-z0-9_-])", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z0-9_.-])xxx(?![A-Za-z0-9_-])", re.IGNORECASE),
)
_SECTION_KEYWORD_GROUPS = (
    ("项目结构", "目录", "module", "structure"),
    ("构建", "开发命令", "build", "development command"),
    ("测试", "test", "pytest", "unittest"),
    ("代码风格", "命名", "lint", "format"),
    ("提交", "pull request", "pr", "commit"),
    ("配置", "安全", "configuration", "security"),
)


def _strip_markdown_code(text: str) -> str:
    without_fenced_blocks = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    return re.sub(r"`[^`\n]*`", "", without_fenced_blocks)


def _contains_placeholder_content(text: str) -> bool:
    searchable_text = _strip_markdown_code(text)
    return any(pattern.search(searchable_text) for pattern in _PLACEHOLDER_PATTERNS)


def _resolve_init_output_path(workspace_root: Path, file_path: str) -> Path:
    candidate = Path(file_path)
    if not candidate.is_absolute():
        candidate = workspace_root / candidate
    return candidate.resolve(strict=False)


def _targets_init_output_path(workspace_root: Path, event: ToolCallRequested) -> bool:
    try:
        parsed = parse_tool_arguments_for_execution(event.tool_call.function.arguments)
    except ToolArgumentsError:
        return False

    file_path = parsed.get("file_path")
    if not isinstance(file_path, str) or not file_path.strip():
        return False

    output_path = (workspace_root / INIT_OUTPUT_FILENAME).resolve(strict=False)
    return _resolve_init_output_path(workspace_root, file_path) == output_path


@dataclass
class InitWriteTargetGuardHook(BaseAgentHook):
    """Block `/init` from modifying files other than the expected output."""

    workspace_root: Path = Path(".")
    priority: int = 13
    guarded_tool_names: tuple[str, ...] = ("write", "edit")

    async def before_event(self, event, ctx) -> HookDecision | None:
        del ctx
        if not isinstance(event, ToolCallRequested):
            return None

        tool_name = event.tool_call.function.name
        if tool_name not in self.guarded_tool_names:
            return None

        if _targets_init_output_path(self.workspace_root, event):
            return None

        return HookDecision(
            action=HookAction.OVERRIDE_RESULT,
            override_result=ToolMessage(
                tool_call_id=event.tool_call.id,
                tool_name=tool_name,
                content=(
                    f"Error: `/init` may only modify {INIT_OUTPUT_FILENAME} at the workspace root.\n"
                    "Do not write or edit any other file."
                ),
                is_error=True,
            ),
            reason="blocked init write outside TGAGENTS.md",
        )


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
class InitDraftBeforeMoreInspectionHook(BaseAgentHook):
    """Require a first TGAGENTS.md draft after limited `/init` inspection."""

    workspace_root: Path = Path(".")
    priority: int = 14
    draft_required_after: int = 12
    inspection_tool_names: tuple[str, ...] = ("read", "glob_search", "grep", "resolve_path")
    drafting_tool_names: tuple[str, ...] = ("write", "edit")
    _inspection_count_before_draft: int = 0
    _draft_started: bool = False

    async def before_event(self, event, ctx) -> HookDecision | None:
        del ctx
        if not isinstance(event, ToolCallRequested):
            return None

        tool_name = event.tool_call.function.name

        if tool_name in self.drafting_tool_names and self._targets_init_output(event):
            self._draft_started = True
            return None

        if tool_name not in self.inspection_tool_names:
            return None

        if self._draft_started:
            return None

        ok, _ = validate_init_output(self.workspace_root)
        if ok:
            self._draft_started = True
            return None

        self._inspection_count_before_draft += 1
        if self._inspection_count_before_draft <= self.draft_required_after:
            return None

        return HookDecision(
            action=HookAction.OVERRIDE_RESULT,
            override_result=ToolMessage(
                tool_call_id=event.tool_call.id,
                tool_name=tool_name,
                content=(
                    "Error: `/init` has done enough repository inspection for the first draft.\n"
                    "You must now create TGAGENTS.md with all required section headings before "
                    "inspecting more files.\n"
                    "Use `write` to create the first complete draft skeleton.\n"
                    "If some facts remain uncertain, mark them as 未验证.\n"
                    "After the draft exists, you may continue refining it with `edit` and "
                    "additional inspection."
                ),
                is_error=True,
            ),
            reason="blocked extra init inspection before TGAGENTS first draft existed",
        )

    def _targets_init_output(self, event: ToolCallRequested) -> bool:
        return _targets_init_output_path(self.workspace_root, event)


@dataclass
class InitRepeatedToolCallGuardHook(BaseAgentHook):
    """Block repeated identical file-inspection calls during `/init`."""

    priority: int = 16
    watched_tool_names: tuple[str, ...] = ("read", "glob_search", "grep", "resolve_path")
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

        cached_result = self._cached_results.get(signature)
        if cached_result is None:
            return None

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

    async def after_event(self, event, ctx, emitted_events) -> HookDecision | None:
        del ctx, emitted_events
        if not isinstance(event, ToolResultReceived):
            return None

        tool_name = event.tool_call.function.name
        if tool_name not in self.watched_tool_names:
            return None

        signature = self._build_signature_from_call(event.tool_call)
        if signature is not None:
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
        "Only include claims supported by files or search results you actually inspected.\n"
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
        "After a small number of high-signal inspections, you must write the first complete "
        "TGAGENTS.md draft instead of continuing open-ended exploration.\n"
        "If some details remain uncertain, write them as 未验证 rather than delaying the first "
        "draft.\n"
        "Once you have enough verified information to write a concise contributor guide, stop "
        "exploring and draft TGAGENTS.md immediately.\n"
        "If the current tool outputs already support the document requirements, switch to "
        "drafting with `write` or `edit` immediately.\n"
        "Execution procedure:\n"
        "1. Check whether TGAGENTS.md already exists at the workspace root.\n"
        "2. Inspect only the minimum set of files needed to verify the contributor guide.\n"
        "3. Reuse high-signal overview documents when available instead of exhaustively reading "
        "many files.\n"
        "4. Draft the full TGAGENTS.md content as soon as the document requirements can be "
        "supported.\n"
        "5. Use `write` or `edit` to persist TGAGENTS.md.\n"
        "6. Only after the file passes validation may you call `done`.\n"
        "When the file is complete, call the done tool with a short completion summary."
    )


def build_init_user_prompt(workspace_root: Path) -> str:
    """Build the task prompt for the dedicated `/init` agent."""
    output_path = workspace_root / INIT_OUTPUT_FILENAME
    return (
        f"Analyze the repository rooted at {workspace_root} and generate or update {output_path}.\n\n"
        "Generate a file named TGAGENTS.md that serves as a contributor guide for this "
        "repository. Your goal is to produce a clear, concise, and well-structured document "
        "with descriptive headings and actionable explanations for each section.\n\n"
        "Document Requirements:\n"
        f"- Title the document \"{INIT_DOCUMENT_TITLE}\".\n"
        "- Write the document in Chinese.\n"
        "- Use Markdown headings (#, ##, etc.) for structure.\n"
        "- Keep the document concise; 200-400 English words worth of content is optimal.\n"
        "- Keep explanations short, direct, and specific to this repository.\n"
        "- Provide examples where helpful, such as commands, paths, and naming patterns.\n"
        "- Maintain a professional, instructional tone.\n"
        "- If a fact cannot be verified from inspected files, mark it as 未验证.\n"
        "- Adapt the outline as needed: add sections if relevant, and omit those that do not apply.\n"
        "- Only include claims supported by files or search results you actually inspected.\n\n"
        "Recommended Sections:\n"
        "- Project Structure & Module Organization\n"
        "- Build, Test, and Development Commands\n"
        "- Coding Style & Naming Conventions\n"
        "- Testing Guidelines\n"
        "- Commit & Pull Request Guidelines\n"
        "- Optional: Security & Configuration Tips\n"
        "- Optional: Architecture Overview\n"
        "- Optional: Agent-Specific Instructions\n\n"
        "Repository-specific expectations:\n"
        "- Prefer concrete paths and commands from the repository over generic advice.\n"
        "- Reuse existing high-signal docs such as README, pyproject.toml, package manifests, "
        "or contributor docs when available.\n"
        "- Only modify TGAGENTS.md.\n"
        "- You must use write or edit to persist TGAGENTS.md before finishing.\n"
        "- Call the done tool after TGAGENTS.md passes validation."
    )


def build_init_agent(
    *,
    llm: BaseChatModel,
    workspace_root: Path,
    dependency_overrides: dict | None = None,
    max_iterations: int = 40,
) -> Agent:
    """Create the temporary agent used by `/init`."""
    agent = Agent(
        llm=llm,
        tools=build_init_tools(),
        system_prompt=build_init_system_prompt(),
        dependency_overrides=dependency_overrides,
        max_iterations=max_iterations,
        compaction=CompactionConfig(enabled=False),
        tool_choice="auto",
        require_done_tool=True,
        hooks=[
            InitWriteTargetGuardHook(workspace_root=workspace_root),
            InitDraftBeforeMoreInspectionHook(workspace_root=workspace_root),
            InitOutputGuardHook(workspace_root=workspace_root),
            InitRepeatedToolCallGuardHook(),
        ],
    )
    agent._context.sliding_window_messages = None
    return agent


def validate_init_output(workspace_root: Path) -> tuple[bool, str | None]:
    """Validate the expected `/init` output file."""
    output_path = workspace_root / INIT_OUTPUT_FILENAME
    if not output_path.exists():
        return False, f"`/init` 未成功生成 {INIT_OUTPUT_FILENAME}。"

    content = output_path.read_text(encoding="utf-8").strip()
    if not content:
        return False, f"`/init` 生成了空的 {INIT_OUTPUT_FILENAME}。"

    lines = [line.rstrip() for line in content.splitlines()]
    normalized = "\n".join(lines)
    normalized_lower = normalized.lower()

    title_line = next((line.strip() for line in lines if line.strip()), "")
    if title_line != f"# {INIT_DOCUMENT_TITLE}":
        return (
            False,
            f"`/init` 生成的 {INIT_OUTPUT_FILENAME} 缺少标题 `# {INIT_DOCUMENT_TITLE}`。",
        )

    level_two_headings = [line for line in lines if re.match(r"^##\s+\S+", line)]
    if len(level_two_headings) < 4:
        return (
            False,
            f"`/init` 生成的 {INIT_OUTPUT_FILENAME} 章节过少，至少应包含 4 个二级标题。",
        )

    visible_chars = len("".join(ch for ch in content if not ch.isspace()))
    if visible_chars < 200:
        return False, f"`/init` 生成的 {INIT_OUTPUT_FILENAME} 内容过短，像是未完成草稿。"
    if visible_chars > 4000:
        return False, f"`/init` 生成的 {INIT_OUTPUT_FILENAME} 过长，不符合简明指南要求。"

    if _contains_placeholder_content(normalized):
        return False, f"`/init` 生成的 {INIT_OUTPUT_FILENAME} 仍包含占位内容。"

    matched_keyword_groups = sum(
        1
        for group in _SECTION_KEYWORD_GROUPS
        if any(keyword.lower() in normalized_lower for keyword in group)
    )
    if matched_keyword_groups < 3:
        return (
            False,
            f"`/init` 生成的 {INIT_OUTPUT_FILENAME} 缺少 contributor guide 的核心章节。",
        )

    return True, None
