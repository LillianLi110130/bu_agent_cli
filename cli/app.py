"""CLI Application for Crab CLI.

Contains the main TGAgentCLI class and loading indicator.
Pure UI logic - receives pre-configured Agent and context.
"""

import json
import asyncio
import io
import logging
import os
import re
import sys
import threading
import time
import hashlib
import uuid
from dataclasses import dataclass
from datetime import datetime
from html import escape as html_escape
from pathlib import Path
from typing import Any, Callable

from agent_core import Agent
from agent_core.agent import (
    FinalResponseEvent,
    HumanApprovalDecision,
    HumanApprovalRequest,
    HumanInLoopConfig,
    ModelRoutingHook,
    TextEvent,
    ThinkingEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from agent_core.agent.events import (
    TextDeltaEvent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
)
from agent_core.agent.hooks import BaseAgentHook
from agent_core.agent.registry import AgentRegistry
from agent_core.agent.runtime_events import (
    ContextMaintenanceRequested,
    LLMResponseReceived,
    ToolResultReceived as RuntimeToolResultReceived,
)
from agent_core.llm import ChatOpenAI
from agent_core.llm.messages import (
    ContentPartImageParam,
    ContentPartTextParam,
    ImageURL,
    SystemMessage,
    UserMessage,
)
from agent_core.memory.review import MemoryReviewChange, MemoryReviewHook
from agent_core.memory.store import MemoryStore
from agent_core.plugin import (
    PluginCommandExecutor,
    PluginExecutionError,
    PluginManager,
)
from agent_core.bootstrap.session_bootstrap import (
    WorkspaceInstructionState,
    sync_workspace_agents_md,
)
from agent_core.runtime_paths import application_root, tg_agent_home
from agent_core.skill.discovery import builtin_skills_dir, user_skills_dir
from agent_core.version import get_cli_version
from agent_core.skill.review import SkillReviewChange, SkillReviewHook
from agent_core.skill.runtime_service import SkillRuntimeService
from agent_core.team.protocol import LEAD_AUTO_TRIGGER_MESSAGE_TYPES
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import ThreadedCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.lexers import PygmentsLexer
from pygments.lexers import BashLexer
from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text

from cli.slash_commands import (
    SlashCommand,
    SlashCommandCompleter,
    SlashCommandRegistry,
    is_slash_command,
    parse_slash_command,
)
from cli.at_commands import (
    AtCommandCompleter,
    AtCommandRegistry,
    expand_at_command,
    is_at_command,
    parse_at_command,
)
from cli.image_input import (
    IMAGE_USAGE,
    ImageInputError,
    is_image_command,
    parse_image_command,
    parse_remote_image_message,
)
from cli.init_agent import build_init_agent, build_init_user_prompt, validate_init_output
from cli.im_bridge import BridgeRequest, FileBridgeStore
from cli.interactive_input import InteractivePrompter
from cli.model_switch_service import ModelAutoState, ModelSwitchService
from cli.memory_handler import MemoryReviewHistoryItem, MemorySlashHandler
from cli.plugins_handler import PluginSlashHandler
from cli.ralph_commands import RalphSlashHandler
from cli.resume_handler import ResumeSlashHandler
from cli.session_runtime import CLISessionRuntime
from cli.session_store import (
    CLISessionStore,
    SessionStoreError,
    workspace_identity,
)
from cli.skills_handler import SkillReviewHistoryItem, SkillSlashHandler
from config.model_config import (
    ModelPreset,
    get_auto_vision_preset,
    get_default_preset,
    get_image_summary_preset,
    load_model_presets,
)
from tools import SandboxContext, SecurityError

UserInputPayload = str | list[ContentPartTextParam | ContentPartImageParam]

logger = logging.getLogger("cli.app")

_REMOTE_RESET_STARTUP_PROMPT_PATH = (
    application_root() / "agent_core" / "prompts" / "remote_reset_startup.md"
)
_SERVER_PROGRESS_EXCLUDED_BLOCKS = ("think", "tool_call")
_UNCLASSIFIED_SKILL_REVIEW_EMPTY_SUMMARY = (
    "review agent 没有产生变更，也没有返回标准 Nothing to save."
)


def _summarize_unclassified_skill_review(final_response: str) -> str:
    summary = final_response.strip()
    if not summary:
        return _UNCLASSIFIED_SKILL_REVIEW_EMPTY_SUMMARY
    if len(summary) <= 400:
        return summary
    return f"{summary[:200]}\n...\n{summary[-200:]}"


def _extract_plain_server_progress_text(content: str) -> str:
    """Return text outside think/tool_call XML-style blocks for server progress."""
    filtered = content
    for tag in _SERVER_PROGRESS_EXCLUDED_BLOCKS:
        filtered = re.sub(
            rf"<{tag}\b[^>]*>.*?</{tag}>",
            "",
            filtered,
            flags=re.IGNORECASE | re.DOTALL,
        )
        filtered = re.sub(
            rf"<{tag}\b[^>]*>.*$",
            "",
            filtered,
            flags=re.IGNORECASE | re.DOTALL,
        )
    return filtered.strip()


_REMOTE_RESET_RESPONSE = "当前上下文已重置"


_UNCLASSIFIED_MEMORY_REVIEW_EMPTY_SUMMARY = (
    "review agent 没有产生 memory 变更，也没有返回标准 Nothing to save."
)


def _summarize_unclassified_memory_review(final_response: str) -> str:
    summary = final_response.strip()
    if not summary:
        return _UNCLASSIFIED_MEMORY_REVIEW_EMPTY_SUMMARY
    if len(summary) <= 400:
        return summary
    return f"{summary[:200]}\n...\n{summary[-200:]}"


@dataclass
class _ExecutionOutcome:
    """Normalized outcome for one logical input execution."""

    continue_running: bool = True
    final_content: str = ""


@dataclass(slots=True)
class _CLIContextBudgetSnapshot:
    """CLI-private snapshot for displaying remaining context budget."""

    model: str
    estimated_tokens: int
    context_limit: int
    remaining_tokens: int
    context_utilization: float
    remaining_ratio: float
    message_count: int
    trigger: str | None = None
    token_estimate_source: str = "unknown"


class _CLIHumanApprovalHandler:
    """CLI-backed approval handler used by runtime hooks."""

    def __init__(self, cli: "TGAgentCLI"):
        self._cli = cli

    async def request_approval(
        self,
        request: HumanApprovalRequest,
    ) -> HumanApprovalDecision:
        return await self._cli._request_human_approval(request)


@dataclass
class _CLIContextBudgetHook(BaseAgentHook):
    """Emit CLI-private budget snapshots after context-changing runtime events."""

    priority: int = 900
    on_compaction_status: Callable[[str], None] | None = None

    async def before_event(self, event, ctx):
        if isinstance(event, ContextMaintenanceRequested):
            assessment = await ctx.agent._context.assess_budget(
                model=ctx.agent.llm.model,
                trigger=None,
            )
            if self._will_attempt_compaction(ctx, assessment):
                self._emit_compaction_status("Compaction start")
        return None

    async def after_event(self, event, ctx, emitted_events):
        del emitted_events
        if isinstance(event, LLMResponseReceived):
            await self._emit_budget(ctx, "post_llm_response")
        elif isinstance(event, RuntimeToolResultReceived):
            await self._emit_budget(ctx, "post_tool_result")
        elif isinstance(event, ContextMaintenanceRequested):
            snapshot = await self._emit_budget(ctx, None)
            if snapshot.trigger == "post_compaction":
                self._emit_compaction_status("Context compacted")
        return None

    def _emit_compaction_status(self, message: str) -> None:
        if self.on_compaction_status is not None:
            self.on_compaction_status(message)

    @staticmethod
    def _will_attempt_compaction(ctx, assessment) -> bool:
        if not getattr(assessment, "needs_compaction", False):
            return False
        context = ctx.agent._context
        compaction_service = getattr(context, "_compaction_service", None)
        if compaction_service is None:
            return False
        if not compaction_service.config.enabled:
            return False
        messages = context.get_messages()
        countable_indices = context._countable_indices_from(
            messages,
            start_index=context.summarized_boundary,
        )
        keep_indices = context._build_recent_keep_indices(
            messages,
            countable_indices,
            compaction_service.config.preserve_recent_messages,
            token_budget=context._recent_keep_token_budget(assessment),
        )
        return any(index not in keep_indices for index in countable_indices)

    @staticmethod
    async def _emit_budget(ctx, trigger: str | None) -> _CLIContextBudgetSnapshot:
        assessment = await ctx.agent._context.assess_budget(
            model=ctx.agent.llm.model,
            trigger=trigger,
        )
        snapshot = _CLIContextBudgetSnapshot(
            model=assessment.model,
            estimated_tokens=assessment.estimated_tokens,
            context_limit=assessment.context_limit,
            remaining_tokens=max(
                0,
                assessment.context_limit - assessment.estimated_tokens,
            ),
            context_utilization=assessment.context_utilization,
            remaining_ratio=max(0.0, 1.0 - assessment.context_utilization),
            message_count=assessment.message_count,
            trigger=assessment.trigger,
            token_estimate_source=assessment.token_estimate_source,
        )
        ctx.emit_ui_event(snapshot)
        return snapshot


# =============================================================================
# Loading Indicator
# =============================================================================


class _LoadingIndicator:
    """A simple loading indicator using direct stdout with ANSI codes."""

    def __init__(self, message: str = "思考中"):
        self.message = message
        self._stop_event = threading.Event()
        self._thread = None
        self._frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def _show_frame(self, frame: int):
        """Show a single frame."""
        sys.stdout.write(f"\r\033[36m{self._frames[frame]}\033[0m {self.message}...")
        sys.stdout.flush()

    def _clear(self):
        """Clear the loading line."""
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()

    def start(self):
        """Start the loading animation in a separate thread."""
        self._stop_event.clear()
        self._show_frame(0)

        def _run():
            frame = 1
            while not self._stop_event.is_set():
                self._show_frame(frame % len(self._frames))
                frame += 1
                time.sleep(0.08)
            self._clear()

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the loading animation."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=0.5)
        self._clear()
        self._clear()
        sys.stdout.write("\n")
        sys.stdout.flush()


class _SafeLoadingIndicator:
    """A loading indicator that avoids ANSI escapes and non-ASCII glyphs."""

    def __init__(self, message: str = "思考中"):
        self.message = message
        self._stop_event = threading.Event()
        self._thread = None
        self._frames = ["-", "\\", "|", "/"]

    def _show_frame(self, frame: int):
        """Show a single frame."""
        sys.stdout.write(f"\r{self._frames[frame]} {self.message}...")
        sys.stdout.flush()

    def _clear(self):
        """Clear the loading line."""
        clear_width = len(self.message) + 6
        sys.stdout.write("\r" + (" " * clear_width) + "\r")
        sys.stdout.flush()

    def start(self):
        """Start the loading animation in a separate thread."""
        self._stop_event.clear()
        self._show_frame(0)

        def _run():
            frame = 1
            while not self._stop_event.is_set():
                self._show_frame(frame % len(self._frames))
                frame += 1
                time.sleep(0.08)
            self._clear()

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the loading animation."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=0.5)
        self._clear()
        self._clear()
        sys.stdout.write("\n")
        sys.stdout.flush()


class _ConsoleMirror:
    """Mirror Rich output to the real console and a plain-text recorder."""

    def __init__(self, primary: Console):
        self._primary = primary
        self._buffer = io.StringIO()
        self._recorder = Console(
            file=self._buffer,
            force_terminal=False,
            color_system=None,
            width=120,
        )

    def print(self, *args, **kwargs):
        self._primary.print(*args, **kwargs)
        self._recorder.print(*args, **kwargs)

    def export_text(self) -> str:
        return self._buffer.getvalue().strip()

    def __getattr__(self, name: str):
        return getattr(self._primary, name)


# =============================================================================
# CLI Application
# =============================================================================


class TGAgentCLI:
    """Interactive CLI for Crab CLI.

    Pure UI class - displays agent events and handles user input.
    """

    # Color scheme
    COLOR_TOOL_CALL = "bright_blue"
    COLOR_TOOL_RESULT = "green"
    COLOR_ERROR = "red"
    COLOR_THINKING = "dim cyan"
    COLOR_FINAL = "bold green"
    IMAGE_DETAIL_MAX_CHARS = 2200
    IMAGE_MEMORY_MAX_CHARS = 600
    BRIDGE_POLL_INTERVAL_SECONDS = 0.1
    TEAM_INBOX_POLL_INTERVAL_SECONDS = 1.0
    TEAM_AUTO_TRIGGER_TYPES = LEAD_AUTO_TRIGGER_MESSAGE_TYPES

    def __init__(
        self,
        agent: Agent,
        context: SandboxContext,
        *,
        slash_registry: SlashCommandRegistry | None = None,
        at_registry: AtCommandRegistry | None = None,
        agent_registry: AgentRegistry | None = None,
        plugin_manager: PluginManager | None = None,
        system_prompt_builder: Callable[[], str] | None = None,
        skill_runtime_service: SkillRuntimeService | None = None,
        bridge_store: FileBridgeStore | None = None,
        session_runtime: CLISessionRuntime | None = None,
    ):
        """Initialize CLI with pre-configured agent and context.

        Args:
            agent: Configured Agent instance
            context: SandboxContext for the session
        """
        self._console = Console()
        self._agent = agent
        self._ctx = context
        self._step_number = 0
        self._loading: _SafeLoadingIndicator | None = None
        self._interactive_terminal_ui_enabled = True
        self._prompter = InteractivePrompter(self._console)
        self._slash_registry = slash_registry or SlashCommandRegistry()
        self._at_registry = at_registry or AtCommandRegistry(
            skill_dirs=[
                builtin_skills_dir(),
                user_skills_dir(),
                self._ctx.working_dir / "skills",
            ]
        )
        self._agent_registry = agent_registry
        self._plugin_manager = plugin_manager
        self._plugin_executor = PluginCommandExecutor()
        self._system_prompt_builder = system_prompt_builder
        self._skill_runtime_service = skill_runtime_service or getattr(
            self._agent,
            "_skill_runtime_service",
            None,
        )
        if self._skill_runtime_service is None:
            self._skill_runtime_service = SkillRuntimeService(
                skill_registry=self._at_registry,
                plugin_manager=self._plugin_manager,
                agent=self._agent,
                system_prompt_builder=self._system_prompt_builder,
            )
        else:
            self._skill_runtime_service.bind_agent(self._agent)
            if self._skill_runtime_service.system_prompt_builder is None:
                self._skill_runtime_service.system_prompt_builder = self._system_prompt_builder
        self._bridge_store = bridge_store
        self._session_runtime = session_runtime
        if self._session_runtime is not None:
            self._agent.bind_session_runtime(self._session_runtime)
        self._model_presets_path = (
            Path(__file__).resolve().parent.parent / "config" / "model_presets.json"
        )
        self._default_model_preset: str | None = None
        self._auto_vision_preset: str | None = None
        self._image_summary_preset: str | None = None
        self._model_presets = self._load_model_presets()
        self._model_switch_service = ModelSwitchService(
            agent=self._agent,
            model_presets=self._model_presets,
            default_model_preset=self._default_model_preset,
            auto_vision_preset=self._auto_vision_preset,
            image_summary_preset=self._image_summary_preset,
            console=self._console,
        )
        self._model_auto_state = ModelAutoState(
            sticky_preset=self._model_switch_service.resolve_current_preset_name()
        )
        self._last_command_final_content = ""
        self._model_pick_active = False
        self._model_pick_order: list[str] = []
        self._skill_review_history: list[SkillReviewHistoryItem] = []
        self._memory_review_history: list[MemoryReviewHistoryItem] = []
        self._memory_store = getattr(
            self._agent,
            "_memory_store",
            getattr(self._ctx, "memory_store", MemoryStore()),
        )
        self._seen_team_inbox_message_ids: set[str] = set()
        self._team_auto_trigger_queue: asyncio.Queue[str] = asyncio.Queue()
        self._agents_md_hash: str | None = None
        self._agents_md_content: str | None = None
        self._ralph_handler: RalphSlashHandler | None = None
        self._approval_handler = _CLIHumanApprovalHandler(self)
        self._agent.human_in_loop_handler = self._approval_handler
        self._agent.human_in_loop_config = HumanInLoopConfig(enabled=False)
        self._agent.register_hook(
            ModelRoutingHook(
                service=self._model_switch_service,
                auto_state=self._model_auto_state,
            )
        )
        self._bind_skill_review_notifications()
        self._bind_memory_review_notifications()
        self._workspace_instruction_state = WorkspaceInstructionState()
        self._last_context_budget: _CLIContextBudgetSnapshot | None = None
        self._last_context_budget_status_line: str | None = None
        self._context_budget_hook_agents: set[int] = set()
        self._subagent_progress_signatures: dict[str, tuple[str, int, str]] = {}
        self._foreground_delegate_depth = 0
        self._active_thinking_id: str | None = None
        self._session_store: CLISessionStore | None = None
        self._conversation_session_id = (
            self._session_runtime.session_id
            if self._session_runtime is not None
            else self._ctx.session_id
        )
        self._last_transcript_flushed_idx = 0
        self._conversation_session_created = False
        self._initialize_session_store()
        self._resume_handler = ResumeSlashHandler(
            store=self._session_store,
            console=self._console,
            workspace_dir=self._ctx.working_dir,
        )
        if self._bridge_store is not None:
            self._bridge_store.initialize()

    def _initialize_session_store(self) -> None:
        """Initialize user-level conversation history storage if available."""
        try:
            store = CLISessionStore(tg_agent_home() / "sessions.db")
        except Exception as exc:  # noqa: BLE001 - resume must not block normal CLI startup.
            self._session_store = None
            self._console.print(f"[yellow]会话历史存储不可用，/resume 将不可用：{exc}[/yellow]")
            return
        self._session_store = store

    def _ensure_current_session_created(self) -> bool:
        """Create the current conversation session row on first real transcript write."""
        if self._session_store is None:
            return False
        if self._conversation_session_created:
            return True
        workspace_root, workspace_key = workspace_identity(self._ctx.working_dir)
        self._session_store.create_session(
            session_id=self._conversation_session_id,
            workspace_root=workspace_root,
            workspace_key=workspace_key,
            model=getattr(self._agent.llm, "model", None),
            system_prompt=self._agent.system_prompt,
        )
        self._conversation_session_created = True
        return True

    def _bind_skill_review_notifications(self) -> None:
        for hook in getattr(self._agent, "hooks", []):
            if isinstance(hook, SkillReviewHook):
                hook.on_changes = self._on_skill_review_changes
                hook.on_manage_errors = self._on_skill_review_manage_errors
                hook.on_nothing_to_save = self._on_skill_review_nothing_to_save
                hook.on_unclassified_no_change = self._on_skill_review_unclassified_no_change
                hook.on_error = self._on_skill_review_error

    def _bind_memory_review_notifications(self) -> None:
        for hook in getattr(self._agent, "hooks", []):
            if isinstance(hook, MemoryReviewHook):
                hook.on_changes = self._on_memory_review_changes
                hook.on_manage_errors = self._on_memory_review_manage_errors
                hook.on_nothing_to_save = self._on_memory_review_nothing_to_save
                hook.on_unclassified_no_change = self._on_memory_review_unclassified_no_change
                hook.on_error = self._on_memory_review_error

    def _on_skill_review_changes(self, changes: list[SkillReviewChange]) -> None:
        action_labels = {
            "created": "已创建 skill",
            "patched": "已更新 skill",
            "edited": "已更新 skill",
            "written": "已更新 skill 文件",
            "removed": "已移除 skill 文件",
        }
        for change in changes:
            label = action_labels.get(change.action, "已更新 skill")
            status = self._skill_review_status_for_action(change.action)
            self._append_skill_review_history(
                status=status,
                summary=label,
                skill_name=change.name,
            )
            self._console.print(f"[dim]{label}：[/dim][cyan]{change.name}[/cyan]")

    def _on_skill_review_nothing_to_save(self) -> None:
        self._append_skill_review_history(
            status="nothing_to_save",
            summary="没有发现值得保存的 skill",
        )
        self._console.print("[dim]Skill review：没有发现值得保存的 skill。[/dim]")

    def _on_skill_review_manage_errors(self, errors: list[str]) -> None:
        summary = "; ".join(error.strip() for error in errors if error.strip())
        if not summary:
            summary = "skill_manage 失败，但没有返回错误详情"
        summary = summary[:240]
        self._append_skill_review_history(
            status="attempt_failed",
            summary=summary,
        )
        self._console.print(
            f"[dim]Skill review：skill_manage 失败：[/dim][yellow]{summary}[/yellow]"
        )

    def _on_skill_review_unclassified_no_change(self, final_response: str) -> None:
        summary = _summarize_unclassified_skill_review(final_response)
        self._append_skill_review_history(
            status="no_change_unclassified",
            summary=summary,
        )
        self._console.print("[dim]Skill review：没有产生 skill 变更，且结果未分类。[/dim]")

    def _on_skill_review_error(self, error: Exception) -> None:
        error_summary = f"{type(error).__name__}: {error}"
        self._append_skill_review_history(
            status="failed",
            summary=error_summary[:240],
        )

    def _append_skill_review_history(
        self,
        *,
        status: str,
        summary: str,
        skill_name: str | None = None,
    ) -> None:
        self._skill_review_history.append(
            SkillReviewHistoryItem(
                created_at=datetime.now(),
                status=status,
                summary=summary,
                skill_name=skill_name,
            )
        )
        self._skill_review_history = self._skill_review_history[-50:]

    @staticmethod
    def _skill_review_status_for_action(action: str) -> str:
        if action == "created":
            return "created"
        if action in {"written", "removed"}:
            return "file_updated"
        return "updated"

    def _on_memory_review_changes(self, changes: list[MemoryReviewChange]) -> None:
        action_labels = {
            "added": "已新增",
            "replaced": "已更新",
            "removed": "已移除",
        }
        target_labels = {
            "user": "用户记忆",
            "memory": "长期记忆",
        }
        for change in changes:
            label = action_labels.get(change.action, "已更新")
            target_label = target_labels.get(change.target, change.target)
            self._append_memory_review_history(
                status=self._memory_review_status_for_action(change.action),
                summary=f"{label}{target_label}",
                target=change.target,
            )
            self._console.print(f"[dim]Memory review：{label}{target_label}。[/dim]")

    def _on_memory_review_nothing_to_save(self) -> None:
        self._append_memory_review_history(
            status="nothing_to_save",
            summary="没有发现值得保存的 memory",
        )
        self._console.print("[dim]Memory review：没有发现值得保存的记忆。[/dim]")

    def _on_memory_review_manage_errors(self, errors: list[str]) -> None:
        summary = "; ".join(error.strip() for error in errors if error.strip())
        if not summary:
            summary = "memory 写入失败，但没有返回错误详情"
        summary = summary[:240]
        self._append_memory_review_history(
            status="attempt_failed",
            summary=summary,
        )
        self._console.print(
            f"[dim]Memory review：memory 写入失败：[/dim][yellow]{summary}[/yellow]"
        )

    def _on_memory_review_unclassified_no_change(self, final_response: str) -> None:
        summary = _summarize_unclassified_memory_review(final_response)
        self._append_memory_review_history(
            status="no_change_unclassified",
            summary=summary,
        )
        self._console.print("[dim]Memory review：没有产生 memory 变更，且结果未分类。[/dim]")

    def _on_memory_review_error(self, error: Exception) -> None:
        error_summary = f"{type(error).__name__}: {error}"
        self._append_memory_review_history(
            status="failed",
            summary=error_summary[:240],
        )

    def _append_memory_review_history(
        self,
        *,
        status: str,
        summary: str,
        target: str | None = None,
    ) -> None:
        self._memory_review_history.append(
            MemoryReviewHistoryItem(
                created_at=datetime.now(),
                status=status,
                summary=summary,
                target=target,
            )
        )
        self._memory_review_history = self._memory_review_history[-50:]

    @staticmethod
    def _memory_review_status_for_action(action: str) -> str:
        if action == "added":
            return "added"
        if action == "removed":
            return "removed"
        return "updated"

    def _load_model_presets(self) -> dict[str, ModelPreset]:
        """Load model presets from config/model_presets.json."""
        if not self._model_presets_path.exists():
            return {}

        try:
            presets = load_model_presets()
        except Exception as e:
            self._console.print(f"[yellow]加载模型预设失败：{e}[/yellow]")
            return {}

        self._default_model_preset = get_default_preset(presets)
        self._auto_vision_preset = get_auto_vision_preset(presets)
        self._image_summary_preset = get_image_summary_preset(presets)

        return presets

    def _clear_auto_switch_state(self) -> None:
        self._model_switch_service.clear_auto_switch_state(self._model_auto_state)

    def _preset_supports_vision(self, preset_name: str | None) -> bool:
        return self._model_switch_service.preset_supports_vision(preset_name)

    def _resolve_vision_preset_name(self) -> str | None:
        return self._model_switch_service.resolve_vision_preset_name()

    def _resolve_image_summary_preset_name(self) -> str | None:
        return self._model_switch_service.resolve_image_summary_preset_name()

    def _resolve_image_summary_llm(self) -> tuple[Any | None, str | None, str | None]:
        return self._model_switch_service._resolve_image_summary_llm()

    def _normalize_image_summary(self, text: str, max_chars: int) -> str:
        return self._model_switch_service._normalize_image_summary(text, max_chars)

    def _normalize_image_detail(self, text: str) -> str:
        return self._model_switch_service._normalize_image_detail(text)

    async def _extract_image_detail(
        self,
        llm: Any,
        image_part: ContentPartImageParam,
        user_text_hint: str,
    ) -> str:
        return await self._model_switch_service._extract_image_detail(
            llm,
            image_part,
            user_text_hint,
        )

    async def _compress_image_detail(self, llm: Any, detail_text: str) -> str:
        return await self._model_switch_service._compress_image_detail(llm, detail_text)

    async def _prepare_text_model_image_memory(self, *, manual: bool) -> None:
        await self._model_switch_service.prepare_text_model_image_memory(manual=manual)

    async def _apply_auto_model_policy(self, has_image: bool) -> bool:
        """Backward-compatible wrapper for automatic model switching."""
        return await self._model_switch_service.ensure_model_for_turn(
            has_image=has_image,
            auto_state=self._model_auto_state,
        )

    def _maybe_inject_agents_md(self) -> None:
        """Inject workspace TGAGENTS.md into context if present."""
        self._workspace_instruction_state = sync_workspace_agents_md(
            agent=self._agent,
            workspace_dir=self._ctx.working_dir,
            state=self._workspace_instruction_state,
        )

    async def _handle_reset_command(self, *, source: str = "local") -> str:
        """Reset conversation state."""
        self._agent.clear_history()
        self._reset_current_session_persistence_cursor()
        await self._refresh_empty_context_budget_display()
        self._console.print("[yellow]会话上下文已重置。[/yellow]")
        return _REMOTE_RESET_RESPONSE

    async def _handle_new_command(self) -> None:
        """Start a clean conversation session in the current CLI process."""
        old_conversation_session_id = self._conversation_session_id
        self._persist_current_session_state()
        if self._session_store is not None and self._conversation_session_created:
            self._session_store.end_session(old_conversation_session_id, reason="new_session")

        self._conversation_session_id = str(uuid.uuid4())[:8]
        self._conversation_session_created = False
        self._last_transcript_flushed_idx = 0
        self._agent.clear_history()

        self._resume_handler.clear_pick()
        self._model_pick_active = False
        self._model_pick_order = []
        self._workspace_instruction_state = WorkspaceInstructionState()
        self._foreground_delegate_depth = 0
        self._active_thinking_id = None

        os.system("cls" if os.name == "nt" else "clear")
        self._print_welcome()
        await self._refresh_empty_context_budget_display()

    def _ensure_context_budget_hook(self, agent: Agent) -> None:
        key = id(agent)
        if key in self._context_budget_hook_agents:
            return
        agent.register_hook(
            _CLIContextBudgetHook(on_compaction_status=self._print_compaction_status)
        )
        self._context_budget_hook_agents.add(key)

    async def _refresh_empty_context_budget_display(self, agent: Agent | None = None) -> None:
        active_agent = agent or self._agent
        assessment = await active_agent._context.assess_budget(
            model=active_agent.llm.model,
            trigger=None,
        )
        self._last_context_budget = _CLIContextBudgetSnapshot(
            model=assessment.model,
            estimated_tokens=0,
            context_limit=assessment.context_limit,
            remaining_tokens=assessment.context_limit,
            context_utilization=0.0,
            remaining_ratio=1.0,
            message_count=assessment.message_count,
            trigger=assessment.trigger,
            token_estimate_source="empty",
        )
        self._last_context_budget_status_line = None

    async def _refresh_current_context_budget_display(
        self,
        agent: Agent | None = None,
        *,
        trigger: str | None = None,
        print_status: bool = False,
    ) -> None:
        active_agent = agent or self._agent
        assessment = await active_agent._context.assess_budget(
            model=active_agent.llm.model,
            trigger=trigger,
        )
        snapshot = _CLIContextBudgetSnapshot(
            model=assessment.model,
            estimated_tokens=assessment.estimated_tokens,
            context_limit=assessment.context_limit,
            remaining_tokens=max(0, assessment.context_limit - assessment.estimated_tokens),
            context_utilization=assessment.context_utilization,
            remaining_ratio=max(0.0, 1.0 - assessment.context_utilization),
            message_count=assessment.message_count,
            trigger=assessment.trigger,
            token_estimate_source=assessment.token_estimate_source,
        )
        self._last_context_budget = snapshot
        if print_status:
            self._print_context_budget_status(snapshot)

    def _context_budget_left_percent(self, snapshot: _CLIContextBudgetSnapshot) -> int:
        return max(0, min(100, round(snapshot.remaining_ratio * 100)))

    def _format_token_count(self, value: int) -> str:
        if value == 0:
            return "0k"
        if value >= 1_000_000:
            return f"{value / 1_000_000:.1f}M"
        if value >= 1_000:
            return f"{value / 1_000:.1f}k"
        return str(value)

    def _format_duration_compact(self, seconds: float | int | None) -> str:
        if seconds is None:
            return "-"
        total_seconds = max(0, int(round(float(seconds))))
        minutes, secs = divmod(total_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours > 0:
            return f"{hours}h{minutes}m{secs}s"
        if minutes > 0:
            return f"{minutes}m{secs}s"
        return f"{secs}s"

    def _format_context_budget_status(self, snapshot: _CLIContextBudgetSnapshot) -> str:
        percent = self._context_budget_left_percent(snapshot)
        used = self._format_token_count(snapshot.estimated_tokens)
        if snapshot.token_estimate_source == "local_full" and snapshot.estimated_tokens > 0:
            used = f"约 {used}"
        limit = self._format_token_count(snapshot.context_limit)
        return f"上下文 {percent}% left · {used}/{limit} tokens · {snapshot.model}"

    def _print_context_budget_status(self, snapshot: _CLIContextBudgetSnapshot) -> None:
        status_line = self._format_context_budget_status(snapshot)
        if status_line == self._last_context_budget_status_line:
            return
        self._last_context_budget_status_line = status_line
        self._console.print(Text(status_line, style="grey50"))

    def _print_compaction_status(self, message: str) -> None:
        had_loading = self._loading is not None
        if had_loading:
            self._stop_loading(self._loading)
            self._loading = None
        self._console.print(Text(message, style="yellow"))
        if had_loading:
            self._loading = self._start_loading("思考中")

    def _render_context_budget_toolbar(self) -> str:
        if self._last_context_budget is None:
            status = "上下文 100% left"
        else:
            status = self._format_context_budget_status(self._last_context_budget)
        return html_escape(status)

    def _build_project_snapshot(self) -> str:
        """Build a lightweight snapshot of the project for summarization."""
        root = self._ctx.working_dir

        def is_ignored_dir(name: str) -> bool:
            return name in {
                ".git",
                ".venv",
                "venv",
                "node_modules",
                "__pycache__",
                ".pytest_cache",
                ".mypy_cache",
                "dist",
                "build",
                ".idea",
            }

        def is_blocked(path: Path) -> bool:
            return self._ctx.is_ignored(path)

        # Tree (depth 4)
        tree_lines: list[str] = []
        for current, dirs, files in os.walk(root):
            current_path = Path(current)
            if is_blocked(current_path):
                dirs[:] = []
                continue
            rel = os.path.relpath(current, root)
            depth = 0 if rel == "." else rel.count(os.sep) + 1
            if depth > 4:
                dirs[:] = []
                continue
            dirs[:] = [
                d for d in dirs if not is_ignored_dir(d) and not is_blocked(current_path / d)
            ]
            indent = "  " * depth
            tree_lines.append(f"{indent}{os.path.basename(current)}/")
            for f in sorted(files):
                if is_blocked(current_path / f):
                    continue
                tree_lines.append(f"{indent}  {f}")

        # Important files (top-level)
        important = [
            "README.md",
            "pyproject.toml",
            "package.json",
            "requirements.txt",
        ]
        file_snippets: list[str] = []
        for name in important:
            path = root / name
            if path.exists() and not is_blocked(path):
                content = path.read_text(encoding="utf-8")[:4000]
                file_snippets.append(f"## {name}\n{content}")

        # Files (depth-limited, read all with truncation)
        file_snippets_all: list[str] = []
        for current, dirs, files in os.walk(root):
            current_path = Path(current)
            if is_blocked(current_path):
                dirs[:] = []
                continue
            rel = os.path.relpath(current, root)
            depth = 0 if rel == "." else rel.count(os.sep) + 1
            if depth > 4:
                dirs[:] = []
                continue
            dirs[:] = [
                d for d in dirs if not is_ignored_dir(d) and not is_blocked(current_path / d)
            ]
            for f in sorted(files):
                path = Path(current) / f
                if is_blocked(path):
                    continue
                try:
                    content = path.read_text(encoding="utf-8", errors="ignore")[:2000]
                except Exception:
                    continue
                file_snippets_all.append(f"## {path.relative_to(root)}\n{content}")

        snapshot = "\n".join(
            [
                f"Project root: {root}",
                "",
                "Tree (depth 4):",
                "\n".join(tree_lines),
                "",
                "Key files:",
                "\n\n".join(file_snippets) if file_snippets else "(none found)",
                "",
                "Files (samples):",
                "\n\n".join(file_snippets_all) if file_snippets_all else "(none found)",
            ]
        )
        return snapshot

    def _print_current_model(self):
        """Print the current model configuration."""
        model = str(self._agent.llm.model)
        base_url = getattr(self._agent.llm, "base_url", None)
        base_url_display = str(base_url) if base_url else "(默认)"
        preset_name = self._resolve_current_preset_name()
        preset_line = preset_name or "(未匹配预设)"
        if preset_name and self._preset_supports_vision(preset_name):
            preset_line += " [视觉]"
        self._console.print(f"当前模型： [cyan]{model}[/cyan]")
        self._console.print(f"预设： [dim]{preset_line}[/dim]")
        self._console.print(f"接口地址： [dim]{base_url_display}[/dim]")
        self._console.print(f"上下文消息数： [dim]{len(self._agent.messages)}[/dim]")

    def _print_model_presets(self):
        """Print configured model presets."""
        if not self._model_presets:
            self._console.print(f"[yellow]未在 {self._model_presets_path} 找到模型预设[/yellow]")
            return

        self._console.print("[bold cyan]模型预设：[/bold cyan]")
        for name, preset in self._model_presets.items():
            model = str(preset["model"])
            vision_marker = " 视觉" if self._preset_supports_vision(name) else ""
            marker = " [green](默认)[/green]" if name == self._default_model_preset else ""
            if name == self._auto_vision_preset:
                marker += " [magenta](自动视觉)[/magenta]"
            if name == self._image_summary_preset:
                marker += " [blue](图像摘要)[/blue]"
            self._console.print(
                f"  [cyan]{name}[/cyan]{marker} -> {model} " f"[dim]{vision_marker}[/dim]"
            )

    def _resolve_current_preset_name(self) -> str | None:
        """Best-effort preset match for current model/base URL."""
        return self._model_switch_service.resolve_current_preset_name()

    def _resolve_exact_current_preset_name(self) -> str | None:
        """Exact preset match for current model/base URL, without fallback."""
        return self._model_switch_service.resolve_exact_current_preset_name()

    def _start_model_pick_mode(self):
        """Show numbered model presets and enter pick mode."""
        if not self._model_presets:
            self._console.print("[yellow]未配置模型预设。[/yellow]")
            self._model_pick_active = False
            self._model_pick_order = []
            return

        self._model_pick_order = list(self._model_presets.keys())
        self._model_pick_active = True
        current_preset = self._resolve_current_preset_name()

        self._console.print()
        self._console.print("[bold cyan]请选择模型预设：[/bold cyan]")
        for idx, name in enumerate(self._model_pick_order, 1):
            preset = self._model_presets[name]
            model = str(preset["model"])
            markers: list[str] = []
            if name == current_preset:
                markers.append("当前")
            if name == self._default_model_preset:
                markers.append("默认")
            if self._preset_supports_vision(name):
                markers.append("视觉")
            if name == self._auto_vision_preset:
                markers.append("自动视觉")
            if name == self._image_summary_preset:
                markers.append("图像摘要")
            marker_text = f" [dim]({', '.join(markers)})[/dim]" if markers else ""
            self._console.print(f"  {idx}. [cyan]{name}[/cyan] -> {model}{marker_text}")
        self._console.print("[dim]输入编号后回车即可切换，输入 q 可取消。[/dim]")

    async def _handle_model_pick_input(self, user_input: str) -> bool:
        """Handle one line of input while in numbered model-pick mode."""
        if not self._model_pick_active:
            return False

        value = user_input.strip()
        if not value:
            self._console.print("[dim]请输入编号，或输入 q 取消。[/dim]")
            return True

        if value.lower() in {"q", "quit", "cancel", "exit"}:
            self._model_pick_active = False
            self._model_pick_order = []
            self._console.print("[yellow]已取消模型选择。[/yellow]")
            return True

        if not value.isdigit():
            self._console.print("[red]选择无效，请输入编号。[/red]")
            return True

        index = int(value)
        if index < 1 or index > len(self._model_pick_order):
            self._console.print(
                f"[red]选择超出范围，请输入 1-{len(self._model_pick_order)}。[/red]"
            )
            return True

        preset_name = self._model_pick_order[index - 1]
        self._model_pick_active = False
        self._model_pick_order = []
        await self._switch_model_preset(preset_name)
        return True

    async def _switch_resume_session(self, target_session_id: str) -> bool:
        """Switch the active conversation history to a persisted session snapshot."""
        if target_session_id == self._conversation_session_id:
            self._resume_handler.clear_pick()
            self._console.print("[yellow]已在该会话中。[/yellow]")
            return True

        if self._session_store is None:
            self._console.print("[yellow]会话历史存储不可用，无法恢复。[/yellow]")
            return False

        meta = self._session_store.get_session(target_session_id)
        if meta is None:
            self._console.print("[red]该会话不存在，无法恢复。[/red]")
            return False

        try:
            snapshot = self._session_store.load_context_snapshot(target_session_id)
        except SessionStoreError:
            self._console.print("[red]该会话数据损坏，无法恢复。[/red]")
            return False

        if snapshot is None:
            self._console.print("[red]该会话缺少可恢复上下文，无法恢复。[/red]")
            return False

        self._persist_current_session_state()
        if self._conversation_session_created:
            self._session_store.end_session(self._conversation_session_id, reason="resumed_other")

        self._conversation_session_id = target_session_id
        self._conversation_session_created = True
        self._agent.system_prompt = meta.system_prompt
        self._agent.load_history(snapshot.messages)
        self._last_transcript_flushed_idx = len(snapshot.messages)
        self._session_store.reopen_session(target_session_id)

        self._resume_handler.clear_pick()
        self._model_pick_active = False
        self._model_pick_order = []
        self._workspace_instruction_state = WorkspaceInstructionState()
        self._foreground_delegate_depth = 0
        self._active_thinking_id = None

        await self._refresh_current_context_budget_display(trigger="resume", print_status=False)
        self._resume_handler.bind_console(self._console)
        self._resume_handler.bind_store(self._session_store)
        self._resume_handler.print_resume_result(meta, snapshot)
        return True

    def _persist_current_session_state(self) -> None:
        """Flush the current main-agent transcript delta and resumable snapshot."""
        if self._session_store is None:
            return
        try:
            messages = self._agent.messages
            if self._last_transcript_flushed_idx > len(messages):
                self._last_transcript_flushed_idx = 0
            new_messages = messages[self._last_transcript_flushed_idx :]
            if not new_messages and not self._conversation_session_created:
                return
            if not self._ensure_current_session_created():
                return
            message_count = self._session_store.append_messages(
                self._conversation_session_id,
                new_messages,
            )
            self._last_transcript_flushed_idx = len(messages)
            compacted = (
                self._agent._context.summarized_boundary > 0
                or self._agent._context._compacted_result is not None
            )
            self._session_store.upsert_context_snapshot(
                session_id=self._conversation_session_id,
                messages=messages,
                compacted=compacted,
            )
            self._session_store.touch_session(
                self._conversation_session_id,
                model=getattr(self._agent.llm, "model", None),
                message_count=message_count,
            )
        except Exception as exc:  # noqa: BLE001 - persistence must not break conversation.
            self._console.print(f"[yellow]会话历史保存失败：{exc}[/yellow]")

    def _reset_current_session_persistence_cursor(self) -> None:
        """Keep persistence state consistent after local context is cleared."""
        self._last_transcript_flushed_idx = 0
        if self._session_store is None:
            return
        if not self._conversation_session_created:
            return
        try:
            self._session_store.upsert_context_snapshot(
                session_id=self._conversation_session_id,
                messages=self._agent.messages,
                compacted=False,
            )
        except Exception as exc:  # noqa: BLE001 - reset should still succeed.
            self._console.print(f"[yellow]会话历史保存失败：{exc}[/yellow]")

    async def _switch_model_preset(self, preset_name: str, *, manual: bool = True) -> bool:
        """Switch to a configured model preset without clearing conversation context."""
        switched = await self._model_switch_service.switch_model_preset(
            preset_name,
            manual=manual,
            auto_state=self._model_auto_state,
        )
        if switched:
            await self._refresh_current_context_budget_display(
                trigger="model_switch",
                print_status=True,
            )
        return switched

    def _store_command_final_content(self, content: str | None) -> None:
        """Persist the latest non-agent command output for bridge results."""
        self._last_command_final_content = (content or "").strip()

    def _capture_console_output(self, callback: Callable[[], None]) -> str:
        """Capture synchronous Rich console output as plain text."""
        with self._console.capture() as capture:
            callback()
        return capture.get().strip()

    async def _capture_console_output_async(self, callback: Callable[[], Any]) -> str:
        """Capture async Rich console output as plain text."""
        with self._console.capture() as capture:
            await callback()
        return capture.get().strip()

    async def _capture_console_output_with_result_async(
        self,
        callback: Callable[[], Any],
    ) -> tuple[Any, str]:
        """Capture async Rich console output and return the callback result."""
        with self._console.capture() as capture:
            result = await callback()
        return result, capture.get().strip()

    def _print_remote_reset_console_output(self, final_content: str | None) -> None:
        """Print the local console hints for a remote `/reset` completion."""
        self._console.print("[yellow]会话上下文已重置。[/yellow]")
        if final_content:
            self._console.print(final_content)

    @staticmethod
    def _format_timestamp(value: object) -> str:
        if value in (None, ""):
            return "-"
        try:
            if isinstance(value, (int, float)):
                return datetime.fromtimestamp(float(value)).strftime("%Y-%m-%d %H:%M:%S")
            if isinstance(value, str):
                parsed = datetime.fromisoformat(value)
                return parsed.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return str(value)
        return str(value)

    @staticmethod
    def _task_status_label(status: str) -> str:
        mapping = {
            "running": "[cyan]running[/cyan]",
            "completed": "[green]completed[/green]",
            "failed": "[red]failed[/red]",
            "cancelled": "[yellow]cancelled[/yellow]",
        }
        return mapping.get(status, f"[white]{status}[/white]")

    def _print_tasks_summary(
        self,
        *,
        shell_tasks: list[dict[str, object]],
        subagent_tasks: dict[str, object] | None,
    ) -> None:
        self._console.print("[bold cyan]任务列表：[/bold cyan]")

        self._console.print("[bold]Shell Tasks[/bold]")
        if not shell_tasks:
            self._console.print("[dim]  (none)[/dim]")
        else:
            for task in sorted(shell_tasks, key=lambda item: float(item.get("created_at", 0) or 0)):
                task_id = str(task.get("task_id", "-"))
                status = str(task.get("status", "-"))
                command = str(task.get("command", "")).strip().replace("\n", " ")
                if len(command) > 72:
                    command = command[:69] + "..."
                self._console.print(
                    f"  {task_id} | {self._task_status_label(status)} | {command or '(empty command)'}"
                )

        self._console.print("[bold]Subagent Tasks[/bold]")
        if not subagent_tasks:
            self._console.print("[dim]  (none)[/dim]")
            return

        ordered_groups = ("running", "completed", "failed", "cancelled")
        has_any = False
        for group in ordered_groups:
            items = subagent_tasks.get(group) if isinstance(subagent_tasks, dict) else None
            if not items:
                continue
            has_any = True
            self._console.print(f"[dim]{group}[/dim]")
            for task in items:
                task_id = str(task.get("task_id", "-"))
                status = str(task.get("status", group))
                name = str(task.get("subagent_name", "-"))
                description = str(task.get("description", "")).strip().replace("\n", " ")
                if len(description) > 56:
                    description = description[:53] + "..."
                tool_calls = task.get("tool_calls")
                if isinstance(tool_calls, list):
                    tools_text = f"tools={len(tool_calls)}"
                else:
                    tools_text = f"tools={int(task.get('steps_completed', 0) or 0)}"
                total_tokens = int(task.get("total_tokens", 0) or 0)
                tokens_text = (
                    f"tokens={self._format_token_count(total_tokens)}"
                    if total_tokens > 0
                    else "tokens=-"
                )
                self._console.print(
                    f"  {task_id} | {self._task_status_label(status)} | {name} | "
                    f"{description or '(no description)'} | {tools_text} | {tokens_text}"
                )
        if not has_any:
            self._console.print("[dim]  (none)[/dim]")

    def _print_task_detail(self, task_info: dict[str, object], *, task_kind: str) -> None:
        self._console.print("[bold cyan]任务详情：[/bold cyan]")
        if task_kind == "shell":
            lines = [
                f"[bold]类型：[/] shell",
                f"[bold]任务 ID：[/] {task_info.get('task_id', '-')}",
                f"[bold]状态：[/] {self._task_status_label(str(task_info.get('status', '-')))}",
                f"[bold]命令：[/] {task_info.get('command', '-')}",
                f"[bold]目录：[/] {task_info.get('cwd', '-')}",
                f"[bold]PID：[/] {task_info.get('pid', '-')}",
                f"[bold]返回码：[/] {task_info.get('returncode', '-')}",
                f"[bold]日志：[/] {task_info.get('log_path', '-')}",
                f"[bold]创建时间：[/] {self._format_timestamp(task_info.get('created_at'))}",
                f"[bold]结束时间：[/] {self._format_timestamp(task_info.get('completed_at'))}",
            ]
            self._console.print("\n".join(lines))
            return

        tool_calls = task_info.get("tool_calls")
        recent_logs = task_info.get("recent_logs")
        tools_used = task_info.get("tools_used")
        prompt = str(task_info.get("prompt", "") or "").strip()
        final_response = str(task_info.get("final_response", "") or "").strip()
        error = str(task_info.get("error", "") or "").strip()
        status = str(task_info.get("status", "-"))
        subagent_name = str(task_info.get("subagent_name", "-"))
        subagent_type = str(task_info.get("subagent_type", "") or "").strip()
        execution_time_ms = float(task_info.get("execution_time_ms", 0) or 0)
        created_at = task_info.get("created_at")
        completed_at = task_info.get("completed_at")

        lines = [
            f"[bold]类型：[/] subagent",
            f"[bold]任务 ID：[/] {task_info.get('task_id', '-')}",
            f"[bold]状态：[/] {self._task_status_label(status)}",
            f"[bold]子代理名称：[/] {subagent_name}",
            f"[bold]子代理类型：[/] {subagent_type or '-'}",
            f"[bold]描述：[/] {task_info.get('description', '-')}",
            f"[bold]模式：[/] {'background' if task_info.get('run_in_background') else 'foreground'}",
            f"[bold]任务种类：[/] {task_info.get('task_kind', '-')}",
            f"[bold]模型：[/] {task_info.get('model', '-')}",
            f"[bold]总 Tokens：[/] {self._format_token_count(int(task_info.get('total_tokens', 0) or 0))}",
            f"[bold]Prompt Tokens：[/] {self._format_token_count(int(task_info.get('total_prompt_tokens', 0) or 0))}",
            f"[bold]Completion Tokens：[/] {self._format_token_count(int(task_info.get('total_completion_tokens', 0) or 0))}",
        ]
        if isinstance(tool_calls, list):
            lines.append(f"[bold]工具调用次数：[/] {len(tool_calls)}")
        if execution_time_ms > 0:
            lines.append(
                f"[bold]耗时：[/] {self._format_duration_compact(execution_time_ms / 1000.0)}"
            )
        if created_at:
            lines.append(f"[bold]创建时间：[/] {self._format_timestamp(created_at)}")
        if completed_at:
            lines.append(f"[bold]结束时间：[/] {self._format_timestamp(completed_at)}")
        self._console.print("\n".join(lines))

        if isinstance(tools_used, list):
            self._console.print(
                f"[bold]使用工具：[/] {', '.join(tools_used) if tools_used else '(none)'}"
            )
        if isinstance(recent_logs, list) and recent_logs:
            self._console.print("[bold]最近日志：[/]")
            for line in recent_logs:
                self._console.print(f"  - {line}")
        if prompt:
            self._console.print("[bold]Prompt：[/]")
            self._console.print(prompt)
        if final_response:
            self._console.print("[bold]结果：[/]")
            self._console.print(final_response)
        if error:
            self._console.print("[bold red]错误：[/bold red]")
            self._console.print(error)

    async def _run_slash_command_with_live_capture(
        self,
        user_input: str,
        *,
        source: str = "local",
    ) -> tuple[bool, str]:
        """Execute one slash command with live colored output and plain-text recording."""
        original_console = self._console
        original_model_console = self._model_switch_service._console
        mirror = _ConsoleMirror(original_console)
        self._console = mirror
        self._model_switch_service._console = mirror
        self._store_command_final_content("")
        try:
            handled = await self._handle_slash_command(user_input, source=source)
        finally:
            self._console = original_console
            self._model_switch_service._console = original_model_console
        final_content = self._last_command_final_content or mirror.export_text()
        return handled, final_content

    def _start_loading(self, message: str = "思考中") -> _SafeLoadingIndicator:
        """Start a loading animation."""
        if not self._interactive_terminal_ui_enabled:
            return None
        loading = _SafeLoadingIndicator(message)
        loading.start()
        time.sleep(0.02)
        return loading

    def _stop_loading(self, loading: _SafeLoadingIndicator | None):
        """Stop the loading animation."""
        if loading:
            loading.stop()

    @staticmethod
    def _is_delegate_tool(tool_name: str) -> bool:
        return tool_name in {"delegate", "delegate_parallel"}

    def _print_slash_help(self):
        """Print slash command help information."""
        self._console.print()
        self._console.print("[bold cyan]Slash 命令：[/bold cyan]")
        self._console.print("[dim]输入 / 可查看可用命令，按 Tab 可自动补全[/dim]")
        self._console.print("[bold cyan]技能命令（@）：[/bold cyan]")
        self._console.print("[dim]使用 @<skill-name> 调用技能，按 Tab 可自动补全[/dim]")
        self._console.print("[dim]图片输入可使用 @\"<path>\"<message> 或 @'<path>'<message>[/dim]")
        self._console.print()

        categories = self._slash_registry.get_by_category()
        for category, commands in sorted(categories.items()):
            self._console.print(f"[bold blue]{category}:[/bold blue]")
            for cmd in commands:
                self._console.print(f"  [cyan]/{cmd.name}[/cyan] - {cmd.description}")
        self._console.print()

    def _print_slash_command_detail(self, command_name: str):
        """Print detailed help for a specific slash command.

        Args:
            command_name: Name of the command (without /)
        """
        cmd = self._slash_registry.get(command_name)
        if not cmd:
            self._console.print(f"[red]未知命令：/{command_name}[/red]")
            return

        self._console.print()
        self._console.print(
            Panel(
                f"[bold cyan]/{cmd.name}[/bold cyan]\n\n"
                f"[dim]{cmd.description}[/dim]\n\n"
                f"[bold]用法：[/bold] {cmd.usage}\n\n"
                + (
                    f"[bold]示例：[/bold]\n" + "\n".join(f"  - {ex}" for ex in cmd.examples)
                    if cmd.examples
                    else ""
                ),
                title="[bold blue]命令详情[/bold blue]",
                border_style="bright_blue",
            )
        )
        self._console.print()

    async def _handle_slash_command(self, text: str, *, source: str = "local") -> bool:
        """Handle a slash command.

        Args:
            text: The slash command text
            source: Input source, such as ``local``, ``im``, or ``web``

        Returns:
            True if the command was handled, False otherwise
        """
        parsed_command = parse_slash_command(text)
        command_name = parsed_command.name
        args = parsed_command.args
        args_text = parsed_command.args_text

        # Handle help command
        if command_name in ("help", "h"):
            if args and args[0].startswith("/"):
                # Show details for a specific command
                self._print_slash_command_detail(args[0][1:])
            elif args:
                self._print_slash_command_detail(args[0])
            else:
                self._print_slash_help()
            return True

        # Handle exit/quit commands
        if command_name in ("exit", "quit", "q"):
            self._console.print("[yellow]再见！[/yellow]")
            raise EOFError()

        # Handle pwd command
        if command_name == "pwd":
            self._console.print(f"{self._ctx.working_dir}")
            return True

        # Handle clear command
        if command_name == "clear" or command_name == "cls":
            os.system("cls" if os.name == "nt" else "clear")
            return True

        # Handle model command
        if command_name == "model":
            if not args:
                self._start_model_pick_mode()
                return True

            if len(args) > 1:
                self._console.print("[red]用法：/model [show|list|<preset>][/red]")
                return True

            subcommand = args[0].lower()
            if subcommand == "show":
                self._print_current_model()
                return True
            if subcommand == "list":
                self._print_model_presets()
                return True

            # Convenience: /model <preset>
            await self._switch_model_preset(args[0])
            return True

        if command_name == "approval":
            if not args or args[0].lower() == "status":
                self._print_approval_status()
                return True

            if len(args) != 1 or args[0].lower() not in {"on", "off"}:
                self._console.print("[red]用法：/approval [on|off|status][/red]")
                return True

            self._agent.human_in_loop_config.enabled = args[0].lower() == "on"
            self._print_approval_status()
            return True

        # Handle reset command
        if command_name == "reset":
            reset_message = await self._handle_reset_command(source=source)
            if source in {"im", "web"}:
                self._store_command_final_content(reset_message)
            return True

        if command_name == "new":
            await self._handle_new_command()
            return True

        if command_name == "resume":
            self._resume_handler.bind_console(self._console)
            self._resume_handler.bind_store(self._session_store)
            self._resume_handler.start_pick_mode(
                current_session_id=self._conversation_session_id,
            )
            return True

        if command_name == "init":
            await self._run_init_agent()
            return True

        if command_name == "tasks":
            shell_tasks = []
            if self._ctx.shell_task_manager is not None:
                shell_tasks = [task.to_dict() for task in self._ctx.shell_task_manager.list_tasks()]

            subagent_tasks = None
            if self._ctx.subagent_executor is not None:
                raw = self._ctx.subagent_executor.list_all_runs()
                subagent_tasks = json.loads(raw) if raw else None

            self._print_tasks_summary(
                shell_tasks=shell_tasks,
                subagent_tasks=subagent_tasks,
            )
            return True

        if command_name == "task":
            if not args:
                self._console.print("[red]用法：/task <task_id>[/red]")
                return True
            task_id = args[0]

            task_info = None
            task_kind = None
            if self._ctx.shell_task_manager is not None:
                shell_task = self._ctx.shell_task_manager.get_task(task_id)
                if shell_task is not None:
                    task_info = shell_task.to_dict()
                    task_kind = "shell"

            if task_info is None and self._ctx.subagent_executor is not None:
                raw = self._ctx.subagent_executor.get_run_status(task_id)
                if raw is not None:
                    task_info = json.loads(raw)
                    task_kind = "subagent"

            if task_info is None:
                self._console.print(f"[red]未找到任务“{task_id}”。[/red]")
                return True
            self._print_task_detail(task_info, task_kind=task_kind or "unknown")
            return True

        if command_name == "task_cancel":
            if not args:
                self._console.print("[red]用法：/task_cancel <task_id>[/red]")
                return True
            task_id = args[0]

            result = None
            if self._ctx.shell_task_manager is not None:
                shell_task = self._ctx.shell_task_manager.get_task(task_id)
                if shell_task is not None:
                    result = await self._ctx.shell_task_manager.cancel(task_id)

            if result is None:
                if self._ctx.subagent_executor is None:
                    self._console.print("[yellow]没有可用的后台任务管理器。[/yellow]")
                    return True
                result = await self._ctx.subagent_executor.cancel_run(task_id)

            self._console.print(result)
            return True

        # Handle history command
        if command_name == "history":
            self._console.print("[dim]命令历史尚未实现。[/dim]")
            return True

        # Handle skills command - list available @ skills
        if command_name == "skills":
            handler = SkillSlashHandler(
                service=self._skill_runtime_service,
                console=self._console,
                review_history=self._skill_review_history,
            )
            result = await handler.handle(args)
            return result.handled

        if command_name == "memory":
            handler = MemorySlashHandler(
                store=self._memory_store,
                console=self._console,
                review_history=self._memory_review_history,
            )
            result = await handler.handle(args)
            return result.handled

        # Handle allow command - add directory to sandbox
        if command_name == "allow":
            if not args:
                self._console.print("[red]用法：/allow <path>[/red]")
                self._console.print("[dim]示例：/allow /path/to/project[/dim]")
            else:
                path_str = " ".join(args)
                try:
                    added_path = self._ctx.add_allowed_dir(path_str)
                    self._console.print(f"[green]已加入允许目录：[/] {added_path}")
                except SecurityError as e:
                    self._console.print(f"[red]{e}[/red]")
            return True

        # Handle allowed command - list allowed directories
        if command_name == "allowed":
            self._console.print()
            self._console.print("[bold cyan]允许目录：[/bold cyan]")
            for i, allowed_dir in enumerate(self._ctx.allowed_dirs, 1):
                # 标记当前工作目录
                marker = (
                    " [dim](current)[/]"
                    if str(allowed_dir.resolve()) == str(self._ctx.working_dir.resolve())
                    else ""
                )
                self._console.print(f"  {i}. {allowed_dir}{marker}")
            self._console.print()
            return True

        # Handle agents command - manage agent configurations
        if command_name == "agents":
            from cli.agents_handler import AgentSlashHandler

            handler = AgentSlashHandler(
                registry=self._agent_registry,
                console=self._console,
                workspace_root=self._ctx.working_dir,
            )
            return await handler.handle(args)

        if command_name == "team":
            from agent_core.team import team_experiment_disabled_message
            from cli.team.auto_prompt import (
                TeamAutoParseError,
                build_team_auto_prompt,
                parse_team_auto_request,
            )
            from cli.team.handler import TeamSlashHandler

            try:
                team_args = TeamSlashHandler.parse_args_text(args_text)
            except ValueError as exc:
                self._console.print(f"[red]参数解析失败：{exc}[/red]")
                return True
            if team_args and team_args[0].lower() == "auto":
                if self._ctx.team_runtime is None:
                    message = team_experiment_disabled_message()
                    self._console.print(f"[yellow]{message}[/yellow]")
                    self._store_command_final_content(message)
                    return True
                try:
                    request = parse_team_auto_request(team_args[1:])
                except TeamAutoParseError as exc:
                    message = f"用法：/team auto <goal> [--name <name>] ({exc})"
                    self._console.print(f"[red]{message}[/red]")
                    self._store_command_final_content(message)
                    return True
                self._console.print("[cyan]进入 team lead 自动编排模式...[/cyan]")
                prompt = build_team_auto_prompt(request)
                final_content = await self._run_agent(prompt, has_image=False)
                self._store_command_final_content(final_content or "")
                return True
            if team_args and team_args[0].lower() == "inbox" and "--peek" not in team_args[1:]:
                if self._ctx.team_runtime is None:
                    message = team_experiment_disabled_message()
                    self._console.print(f"[yellow]{message}[/yellow]")
                    self._store_command_final_content(message)
                    return True
                team_id = team_args[1] if len(team_args) > 1 and not team_args[1].startswith("--") else None
                team_id = team_id or self._ctx.team_runtime.get_active_team()
                if team_id is None:
                    message = "用法：/team inbox [team_id] [--peek]"
                    self._console.print(f"[red]{message}[/red]")
                    self._store_command_final_content(message)
                    return True
                messages = self._ctx.team_runtime.read_lead_inbox(team_id, ack=True)
                if not messages:
                    message = "lead inbox 为空。"
                    self._console.print(f"[dim]{message}[/dim]")
                    self._store_command_final_content(message)
                    return True
                self._console.print(
                    f"[cyan]已读取 {len(messages)} 条 lead inbox 消息，提交给 lead 处理...[/cyan]"
                )
                prompt = self._build_team_inbox_auto_trigger_prompt(
                    team_id=team_id,
                    messages=messages,
                )
                final_content = await self._run_agent(prompt, has_image=False)
                self._store_command_final_content(final_content or "")
                return True

            handler = TeamSlashHandler(
                runtime=self._ctx.team_runtime,
                console=self._console,
            )
            return await handler.handle(team_args)

        if command_name == "plugins":
            if self._plugin_manager is None:
                self._console.print("[yellow]插件管理器未配置。[/yellow]")
                return True

            handler = PluginSlashHandler(
                manager=self._plugin_manager,
                console=self._console,
            )
            result = await handler.handle(args)
            if result.reloaded:
                self._plugin_manager.reload_all()
                self._refresh_system_prompt()
            return result.handled

        if command_name == "ralph":
            if self._ralph_handler is None:
                self._ralph_handler = RalphSlashHandler(
                    workspace_root=self._ctx.working_dir,
                    console=self._console,
                )
            return await self._ralph_handler.handle(args)

        if self._plugin_manager is not None:
            plugin_command = self._plugin_manager.get_command(command_name)
            if plugin_command is not None:
                try:
                    command_output = await self._plugin_executor.execute(
                        plugin_command,
                        args=args,
                        args_text=args_text,
                        working_dir=self._ctx.working_dir,
                    )
                except PluginExecutionError as e:
                    self._console.print(f"[red]{e}[/red]")
                    return True

                if plugin_command.mode == "python":
                    if command_output:
                        self._console.print(command_output)
                    return True

                await self._run_agent(command_output, has_image=False)
                return True

        # Unknown command
        self._console.print(f"[red]未知命令：/{command_name}[/red]")
        self._console.print(f"[dim]输入 /help 查看可用命令。[/dim]")
        return True

    def _refresh_system_prompt(self) -> None:
        """Rebuild the agent system prompt after plugin registry changes."""
        if self._system_prompt_builder is None:
            return
        self._agent.system_prompt = self._system_prompt_builder()
        self._agent.clear_history()
        self._reset_current_session_persistence_cursor()
        self._console.print("[yellow]插件重载后会话上下文已重置。[/yellow]")

    def _print_available_skills(self):
        """Print all available @ skills grouped by category."""
        self._console.print()
        self._console.print("[bold cyan]可用技能（@）：[/bold cyan]")
        self._console.print("[dim]可在消息前使用 @<skill-name> 先加载技能[/dim]")
        self._console.print("[dim]Image input: @\"<path>\"<message> or @'<path>'<message>[/dim]")
        self._console.print()

        categories = self._at_registry.get_by_category()
        if not categories:
            self._console.print("[yellow]未找到技能。[/yellow]")
            self._console.print()
            return

        for category, commands in sorted(categories.items()):
            self._console.print(f"[bold blue]{category}:[/bold blue]")
            for cmd in commands:
                self._console.print(f"  [cyan]@{cmd.name}[/cyan] - {cmd.description}")
        self._console.print()

    async def _handle_at_command(self, text: str) -> bool:
        """Handle an @ command for skill invocation."""
        self._store_command_final_content("")
        skill_name, message = parse_at_command(text)
        if not skill_name:
            self._console.print("[yellow]无效的 @ 命令。[/yellow]")
            self._console.print("[dim]输入 @ 后按 Tab 可查看可用技能。[/dim]")
            self._store_command_final_content("无效的 @ 命令。\n输入 @ 后按 Tab 可查看可用技能。")
            return True

        skill = self._at_registry.get(skill_name)
        if not skill:
            self._console.print(f"[yellow]未找到技能：@{skill_name}[/yellow]")
            self._console.print("[dim]使用 /skills 查看可用技能列表。[/dim]")
            self._store_command_final_content(
                f"未找到技能：@{skill_name}\n使用 /skills 查看可用技能列表。"
            )
            return True

        self._console.print(f"[cyan]正在使用 @{skill.name}...[/cyan]")
        try:
            expanded_message = expand_at_command(skill, message)
        except (IOError, ValueError) as e:
            self._console.print(f"[red]加载技能失败：{e}[/red]")
            self._store_command_final_content(f"加载技能失败：{e}")
            return True

        self._store_command_final_content(await self._run_agent(expanded_message, has_image=False))
        return True

    async def _run_init_agent(self) -> str | None:
        """Run the dedicated `/init` agent workflow."""
        self._console.print("[dim]正在通过专用 init agent 分析仓库并生成 TGAGENTS.md...[/dim]")
        init_agent = build_init_agent(
            llm=self._agent.llm,
            workspace_root=self._ctx.working_dir,
            dependency_overrides=self._agent.dependency_overrides,
        )
        init_runtime = CLISessionRuntime.create_helper_top_level_runtime(
            self._ctx,
            helper_name="init",
        )
        init_agent.bind_session_runtime(init_runtime)
        prompt = build_init_user_prompt(self._ctx.working_dir)
        final_response = await self._run_agent(prompt, has_image=False, agent=init_agent)
        ok, error = validate_init_output(self._ctx.working_dir)
        if not ok:
            self._store_command_final_content(error)
            self._console.print(f"[red]{error}[/red]")
        else:
            self._maybe_inject_agents_md()
            success_message = "已生成并注入 TGAGENTS.md"
            self._store_command_final_content(final_response or success_message)
            self._console.print(f"[yellow]{success_message}[/yellow]")
        return final_response

    async def _run_agent(
        self,
        user_input: UserInputPayload,
        has_image: bool = False,
        agent: Agent | None = None,
        intermediate_text_callback: Callable[[str], None] | None = None,
        propagate_errors: bool = False,
        enable_interactive_terminal_ui: bool = True,
    ) -> str | None:
        """Run the agent with user input and display events."""
        self._step_number = 0
        self._console.print()
        final_response: str | None = None
        active_agent = agent or self._agent
        previous_interactive_terminal_ui_enabled = self._interactive_terminal_ui_enabled
        self._interactive_terminal_ui_enabled = enable_interactive_terminal_ui
        self._ensure_context_budget_hook(active_agent)

        # Keep workspace instructions synchronized for the main CLI agent.
        # Dedicated helper agents like /init manage their own injection timing.
        if active_agent is self._agent:
            self._maybe_inject_agents_md()

        # Start loading animation
        self._loading = self._start_loading("思考中")

        # Show cancel hint for local runs, or a static thinking hint for remote runs.
        if enable_interactive_terminal_ui:
            self._console.print("[dim]按 q 可取消本次运行[/dim]")
        else:
            self._console.print("[dim]思考中...[/dim]")

        # Create cancellation event for 'q' key
        import asyncio

        cancel_event = asyncio.Event()
        pending_intermediate_text: list[str] = []
        streamed_text_started = False

        def flush_intermediate_text() -> None:
            if intermediate_text_callback is None or not pending_intermediate_text:
                pending_intermediate_text.clear()
                return
            content = _extract_plain_server_progress_text("".join(pending_intermediate_text))
            pending_intermediate_text.clear()
            if content:
                intermediate_text_callback(content)

        def render_stream_text(text: str) -> None:
            nonlocal streamed_text_started
            if enable_interactive_terminal_ui:
                self._console.print(text, end="")
            else:
                self._write_stream_chunk(text)
            streamed_text_started = True

        # Background task to listen for 'q' key
        async def listen_for_cancel():
            """Listen for 'q' key press to cancel the current agent run."""
            if not enable_interactive_terminal_ui:
                return
            import sys
            import os

            if os.name == "nt":  # Windows
                try:
                    import msvcrt

                    while not cancel_event.is_set():
                        if msvcrt.kbhit():  # Check if key is available
                            ch = msvcrt.getwch()  # Get wide char (Unicode)
                            if ch.lower() == "q":
                                cancel_event.set()
                                break
                        await asyncio.sleep(0.05)  # Small delay to prevent busy-wait
                except (ImportError, AttributeError):
                    pass
            else:  # Unix/Linux/macOS
                try:
                    import select
                    import tty
                    import termios

                    # Save old terminal settings
                    old_settings = None
                    try:
                        old_settings = termios.tcgetattr(sys.stdin)
                        tty.setcbreak(sys.stdin.fileno())
                    except (termios.error, OSError):
                        # Not a TTY, skip
                        return

                    while not cancel_event.is_set():
                        # Check if data is available on stdin
                        if select.select([sys.stdin], [], [], 0)[0]:
                            ch = sys.stdin.read(1)
                            if ch.lower() == "q":
                                cancel_event.set()
                                break
                        await asyncio.sleep(0.05)
                except (ImportError, AttributeError, termios.error, OSError):
                    pass
                finally:
                    # Restore terminal settings
                    if old_settings is not None:
                        try:
                            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                        except (termios.error, OSError):
                            pass

        # Start the listener task
        listener_task = asyncio.create_task(listen_for_cancel())
        try:
            # Pass cancel_event to agent for immediate cancellation
            async for event in active_agent.query_stream(user_input, cancel_event=cancel_event):
                # Cancellation is now handled inside query_stream for immediate response
                # This check is for final confirmation
                if cancel_event.is_set():
                    if self._ctx.subagent_executor is not None:
                        await self._ctx.subagent_executor.cancel_running_foreground_runs()
                    self._console.print()
                    self._console.print("[yellow]用户已取消本次运行（q 键）[/yellow]")
                    break

                if isinstance(event, _CLIContextBudgetSnapshot):
                    self._last_context_budget = event
                    self._print_context_budget_status(event)
                    continue

                if isinstance(event, ToolCallEvent):
                    flush_intermediate_text()
                    self._stop_loading(self._loading)
                    self._loading = None

                    if self._is_delegate_tool(event.tool):
                        self._foreground_delegate_depth += 1

                    self._step_number += 1
                    args_str = str(event.args)[:100]
                    if len(str(event.args)) > 100:
                        args_str += "..."
                    self._console.print()
                    self._console.print(
                        f"[{self.COLOR_TOOL_CALL}]步骤 {self._step_number}：[/] "
                        f"[bold]{event.tool}[/]"
                    )
                    self._console.print(f"  [dim]参数：{args_str}[/]")
                    self._console.print("  [dim]（按 q 可取消）[/dim]")

                    if not self._is_delegate_tool(event.tool):
                        # Start loading while the tool runs.
                        self._loading = self._start_loading("执行中")

                elif isinstance(event, ToolResultEvent):
                    self._stop_loading(self._loading)
                    self._loading = None

                    result = str(event.result)
                    if event.is_error:
                        self._console.print(f"  [{self.COLOR_ERROR}]错误：{result[:200]}[/]")
                    else:
                        if len(result) > 300:
                            self._console.print(
                                f"  [{self.COLOR_TOOL_RESULT}]结果：{result[:300]}...[/]"
                            )
                        else:
                            self._console.print(f"  [{self.COLOR_TOOL_RESULT}]结果：{result}[/]")

                    if self._is_delegate_tool(event.tool):
                        self._foreground_delegate_depth = max(
                            0, self._foreground_delegate_depth - 1
                        )

                    # Restart loading for the next LLM call unless a foreground
                    # delegate run is still actively streaming its own progress.
                    if self._foreground_delegate_depth == 0:
                        self._loading = self._start_loading("思考中")

                elif isinstance(event, ThinkingEvent):
                    self._stop_loading(self._loading)
                    self._loading = None
                    if enable_interactive_terminal_ui:
                        self._console.print(f"[{self.COLOR_THINKING}]思考：{event.content}[/]")

                elif isinstance(event, ThinkingStartEvent):
                    self._stop_loading(self._loading)
                    self._loading = None
                    if enable_interactive_terminal_ui:
                        self._active_thinking_id = event.think_id
                        self._console.print(f"[{self.COLOR_THINKING}]思考：[/]", end="")

                elif isinstance(event, ThinkingDeltaEvent):
                    self._stop_loading(self._loading)
                    self._loading = None
                    if enable_interactive_terminal_ui:
                        if self._active_thinking_id != event.think_id:
                            self._active_thinking_id = event.think_id
                            self._console.print(f"[{self.COLOR_THINKING}]思考：[/]", end="")
                        self._console.print(f"[{self.COLOR_THINKING}]{event.delta}[/]", end="")

                elif isinstance(event, ThinkingEndEvent):
                    self._stop_loading(self._loading)
                    self._loading = None
                    if enable_interactive_terminal_ui:
                        if self._active_thinking_id == event.think_id:
                            self._active_thinking_id = None
                        self._console.print()

                elif isinstance(event, TextDeltaEvent):
                    if intermediate_text_callback is not None:
                        pending_intermediate_text.append(event.delta)
                    self._stop_loading(self._loading)
                    self._loading = None
                    render_stream_text(event.delta)

                elif isinstance(event, TextEvent):
                    if intermediate_text_callback is not None:
                        pending_intermediate_text.append(event.content)
                    self._stop_loading(self._loading)
                    self._loading = None
                    logger.info(event.content)
                    render_stream_text(event.content)

                elif isinstance(event, FinalResponseEvent):
                    pending_intermediate_text.clear()
                    self._stop_loading(self._loading)
                    self._loading = None
                    final_response = event.content
                    if enable_interactive_terminal_ui:
                        self._console.print()
                        self._console.print()
                    elif streamed_text_started:
                        sys.stdout.write("\n\n")
                        sys.stdout.flush()

        except Exception as e:
            self._stop_loading(self._loading)
            self._loading = None
            self._console.print(f"[{self.COLOR_ERROR}]错误：{e}[/]")
            if propagate_errors:
                raise
        finally:
            # Cancel the listener task
            cancel_event.set()
            listener_task.cancel()
            try:
                await listener_task
            except (asyncio.CancelledError, Exception):
                pass
            self._interactive_terminal_ui_enabled = previous_interactive_terminal_ui_enabled

        # Ensure loading is stopped
        self._stop_loading(self._loading)
        self._loading = None
        if enable_interactive_terminal_ui:
            self._console.print()
        elif streamed_text_started:
            sys.stdout.write("\n")
            sys.stdout.flush()
        if active_agent is self._agent:
            self._persist_current_session_state()
        return final_response

    async def _on_task_completed(self, result: Any):
        """Handle background task completion notification.

        Args:
            result: TaskResult from background subagent task manager
        """
        from agent_core.task import SubagentTaskResult as TaskResult
        from rich.panel import Panel

        if not isinstance(result, TaskResult):
            return

        # Display completion notification
        status_emoji = "✅" if result.status == "completed" else "❌"
        status_color = "green" if result.status == "completed" else "red"

        if result.status == "completed":
            self._console.print(
                f"[dim]\\[bg done] {result.subagent_name} {result.task_id} "
                f"finished in {self._format_duration_compact(result.execution_time_ms / 1000.0)} "
                f"(tokens={self._format_token_count(int(result.total_tokens or 0))}). "
                f"Use /task {result.task_id} to inspect the result.[/dim]"
            )
        elif result.status == "failed":
            self._console.print()
            self._console.print(
                Panel(
                    f"[{status_color}]{status_emoji} 任务失败：[/{status_color}]\n"
                    f"[bold]子智能体：[/] {result.subagent_name}\n"
                    f"[bold]任务 ID：[/] {result.task_id}\n"
                    f"[bold]错误：[/] {result.error}",
                    title="[bold blue]后台任务通知[/bold blue]",
                    border_style=status_color,
                )
            )
        elif result.status == "cancelled":
            self._console.print()
            self._console.print(
                Panel(
                    f"[yellow]⏹️  任务已取消：[/yellow]\n"
                    f"[bold]子智能体：[/] {result.subagent_name}\n"
                    f"[bold]任务 ID：[/] {result.task_id}",
                    title="[bold blue]后台任务通知[/bold blue]",
                    border_style="yellow",
                )
            )

    def _enqueue_local_bridge_input(self, user_input: str) -> BridgeRequest:
        """Persist one local terminal input into the file-backed bridge."""
        if self._bridge_store is None:
            raise RuntimeError("Local bridge is not configured")
        return self._bridge_store.enqueue_text(
            user_input,
            source="local",
            source_meta={"sender_id": "local-user"},
            remote_response_required=False,
        )

    @staticmethod
    def _is_immediate_local_exit_input(user_input: str) -> bool:
        """Return whether a terminal input should exit immediately instead of being enqueued."""
        normalized = user_input.strip().lower()
        return normalized in {"/exit", "/quit", "/q", "exit", "quit"}

    def _has_pending_bridge_work(self) -> bool:
        """Return whether the local bridge currently has queued work."""
        if self._bridge_store is None:
            return False
        return self._bridge_store.pending_count() > 0

    def _build_bridge_progress_callback(
        self,
        *,
        source: str,
        bridge_request: BridgeRequest | None,
    ) -> Callable[[str], None] | None:
        """Build a callback that records intermediate remote agent text."""
        if source not in {"im", "web"} or bridge_request is None or self._bridge_store is None:
            return None

        def record_progress(content: str) -> None:
            self._bridge_store.record_progress(bridge_request, content)

        return record_progress

    async def _prompt_async_with_patch_stdout(self, session: Any):
        """Read one prompt input while stdout is safely proxied above the prompt."""
        from prompt_toolkit.patch_stdout import patch_stdout

        with patch_stdout(raw=True):
            return await session.prompt_async()

    def _dismiss_active_prompt_line(self) -> None:
        """Move the terminal cursor off the current prompt line before remote output."""
        self._console.print()

    @staticmethod
    def _write_stream_chunk(text: str) -> None:
        """Append streaming text directly to stdout without Rich re-rendering."""
        if not text:
            return
        sys.stdout.write(text)
        sys.stdout.flush()

    async def _terminate_active_prompt(
        self,
        session: Any,
        prompt_task: asyncio.Task | None,
    ) -> None:
        """Best-effort terminate the active prompt_toolkit application before remote output."""
        app = getattr(session, "app", None)
        if app is not None and getattr(app, "is_running", False):
            try:
                app.exit(exception=KeyboardInterrupt())
            except Exception:
                pass
        if prompt_task is not None and not prompt_task.done():
            prompt_task.cancel()
        if prompt_task is not None:
            try:
                await prompt_task
            except (asyncio.CancelledError, EOFError, KeyboardInterrupt):
                pass

    async def _execute_input_text(
        self,
        user_input: str,
        *,
        source: str = "local",
        parsed_remote_image=None,
        bridge_request: BridgeRequest | None = None,
    ) -> _ExecutionOutcome:
        """Execute one normalized line of user input."""
        user_input = user_input.strip()
        if not user_input:
            return _ExecutionOutcome()

        if self._session_runtime is not None:
            self._session_runtime.touch()

        intermediate_text_callback = self._build_bridge_progress_callback(
            source=source,
            bridge_request=bridge_request,
        )

        # Handle numbered resume picker mode before normal command dispatch.
        if self._resume_handler.pick_active:
            self._resume_handler.bind_console(self._console)
            pick_result = self._resume_handler.handle_pick_input(user_input)
            if pick_result.selected_session_id is not None:
                await self._switch_resume_session(pick_result.selected_session_id)
            if pick_result.handled:
                return _ExecutionOutcome()

        # Handle numbered model picker mode
        if self._model_pick_active:
            if await self._handle_model_pick_input(user_input):
                return _ExecutionOutcome()

        # Handle quoted @ image command
        if is_image_command(user_input):
            try:
                parsed = parse_image_command(user_input, self._ctx)
            except ImageInputError as e:
                self._console.print(f"[red]{e}[/red]")
                self._console.print(f"[dim]{IMAGE_USAGE}[/dim]")
                return _ExecutionOutcome()
            final_content = await self._run_agent(
                parsed.content_parts,
                has_image=True,
                intermediate_text_callback=intermediate_text_callback,
                propagate_errors=source == "remote",
                enable_interactive_terminal_ui=source == "local",
            )
            return _ExecutionOutcome(final_content=final_content or "")

        if parsed_remote_image is not None:
            final_content = await self._run_agent(
                parsed_remote_image.content_parts,
                has_image=True,
                intermediate_text_callback=intermediate_text_callback,
                propagate_errors=source == "remote",
                enable_interactive_terminal_ui=source == "local",
            )
            return _ExecutionOutcome(final_content=final_content or "")

        # Handle @ commands (skill invocation)
        if is_at_command(user_input):
            try:
                handled = await self._handle_at_command(user_input)
                if handled:
                    return _ExecutionOutcome(final_content=self._last_command_final_content)
            except EOFError:
                return _ExecutionOutcome(continue_running=False, final_content="再见！")
            return _ExecutionOutcome()

        # Handle slash commands
        if is_slash_command(user_input):
            try:
                handled, final_content = await self._run_slash_command_with_live_capture(
                    user_input,
                    source=source,
                )
                if handled:
                    return _ExecutionOutcome(final_content=final_content)
            except EOFError:
                return _ExecutionOutcome(continue_running=False, final_content="再见！")
            return _ExecutionOutcome()

        # Handle legacy built-in commands (without slash)
        if user_input.lower() in ["exit", "quit"]:
            self._console.print("[yellow]再见！[/yellow]")
            return _ExecutionOutcome(continue_running=False, final_content="再见！")

        if user_input.lower() == "help":
            self._print_help()
            return _ExecutionOutcome()

        if user_input.lower() == "pwd":
            current_dir = f"{self._ctx.working_dir}"
            self._console.print(f"当前目录：{current_dir}")
            return _ExecutionOutcome(final_content=current_dir)

        # Run agent
        final_content = await self._run_agent(
            user_input,
            has_image=False,
            intermediate_text_callback=intermediate_text_callback,
            propagate_errors=source == "remote",
            enable_interactive_terminal_ui=source == "local",
        )
        return _ExecutionOutcome(final_content=final_content or "")

    async def _drain_bridge_queue(self) -> bool:
        """Consume queued bridge requests until no pending work remains."""
        if self._bridge_store is None:
            return True

        while True:
            request = self._bridge_store.claim_next_pending()
            if request is None:
                return True

            parsed_remote_image = None
            if request.source in {"im", "web"}:
                try:
                    parsed_remote_image = parse_remote_image_message(request.content)
                except ImageInputError as exc:
                    self._bridge_store.fail_request(
                        request,
                        final_content=f"执行失败：{exc}",
                        error_code="REMOTE_IMAGE_MESSAGE_INVALID",
                        error_message=str(exc),
                    )
                    self._console.print(
                        f"\n[bold red]远程图片消息无效[/bold red] "
                        f"[dim](seq={request.seq}, request_id={request.request_id})[/dim]"
                    )
                    self._console.print(f"[dim]{exc}[/dim]")
                    continue

                if parsed_remote_image is not None:
                    request.input_kind = "image"
                    preview = parsed_remote_image.user_text.replace("\n", " ")
                else:
                    preview = request.content.strip().replace("\n", " ")
                if len(preview) > 80:
                    preview = preview[:77] + "..."
                self._console.print(
                    f"\n[bold cyan]收到远程任务[/bold cyan] "
                    f"[dim](seq={request.seq}, request_id={request.request_id})[/dim]"
                )
                if preview:
                    self._console.print(f"[dim]{preview}[/dim]")
                if parsed_remote_image is not None and parsed_remote_image.invalid_images:
                    self._console.print(
                        f"[yellow]已跳过 {len(parsed_remote_image.invalid_images)} 张无效图片，继续处理其余图片。[/yellow]"
                    )

            try:
                outcome = await self._execute_input_text(
                    request.content,
                    source=request.source,
                    parsed_remote_image=parsed_remote_image,
                    bridge_request=request,
                )
            except Exception as exc:
                self._bridge_store.fail_request(
                    request,
                    final_content=f"Execution failed: {exc}",
                    error_code="LOCAL_BRIDGE_EXECUTION_ERROR",
                    error_message=str(exc),
                )
                if request.source == "remote" or request.remote_response_required:
                    continue
                raise

            if request.source in {"im", "web"} and request.content.strip() == "/reset":
                self._print_remote_reset_console_output(outcome.final_content)

            self._bridge_store.complete_request(request, final_content=outcome.final_content)
            if not outcome.continue_running:
                return False

    async def _consume_team_inbox_auto_triggers(self) -> None:
        """Watch active team lead inbox and enqueue safe automatic lead prompts."""
        while True:
            await asyncio.sleep(self.TEAM_INBOX_POLL_INTERVAL_SECONDS)
            runtime = getattr(self._ctx, "team_runtime", None)
            if runtime is None:
                continue
            try:
                team_id = runtime.get_active_team()
            except Exception:
                continue
            if not team_id:
                continue

            try:
                messages = runtime.read_lead_inbox(team_id, ack=False)
            except Exception:
                continue

            trigger_messages = [
                message
                for message in messages
                if message.type in self.TEAM_AUTO_TRIGGER_TYPES
                and message.message_id not in self._seen_team_inbox_message_ids
            ]
            if not trigger_messages:
                continue

            prompt = self._build_team_inbox_auto_trigger_prompt(
                team_id=team_id,
                messages=trigger_messages,
            )
            message_ids = [message.message_id for message in trigger_messages]
            enqueued = False
            if self._bridge_store is not None:
                request_id = self._team_auto_trigger_request_id(team_id, message_ids)
                if self._bridge_store.has_request(request_id):
                    enqueued = True
                else:
                    self._bridge_store.enqueue_text(
                        prompt,
                        source="local",
                        source_meta={
                            "kind": "team_inbox_auto_trigger",
                            "team_id": team_id,
                            "message_ids": message_ids,
                        },
                        request_id=request_id,
                        input_kind="text",
                    )
                    enqueued = True
            else:
                await self._team_auto_trigger_queue.put(prompt)
                enqueued = True

            if not enqueued:
                continue

            for message in trigger_messages:
                self._seen_team_inbox_message_ids.add(message.message_id)
            try:
                runtime.ack_lead_messages(team_id, set(message_ids))
            except Exception:
                logger.exception("Failed to ack team inbox auto trigger messages")

            preview = trigger_messages[0].body.strip().replace("\n", " ")
            if len(preview) > 80:
                preview = preview[:77] + "..."
            self._console.print(
                f"\n[bold cyan]Team inbox 自动触发[/bold cyan] "
                f"[dim]team={team_id}, messages={len(trigger_messages)}[/dim]"
            )
            if preview:
                self._console.print(f"[dim]{preview}[/dim]")

    @staticmethod
    def _team_auto_trigger_request_id(team_id: str, message_ids: list[str]) -> str:
        digest = hashlib.sha256(
            f"{team_id}:{','.join(sorted(message_ids))}".encode("utf-8")
        ).hexdigest()[:16]
        return f"team_inbox_{digest}"

    @staticmethod
    def _build_team_inbox_auto_trigger_prompt(*, team_id: str, messages: list[Any]) -> str:
        from cli.team.auto_prompt import TEAM_LANGUAGE_RULES

        payload = [message.to_dict() for message in messages]
        return (
            "[Team Inbox Auto Trigger]\n\n"
            f"Active team: {team_id}\n\n"
            f"{TEAM_LANGUAGE_RULES}\n\n"
            "New lead inbox messages:\n"
            f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
            "Instructions:\n"
            "- Treat `clarification_request` messages as coordination blockers.\n"
            "- Treat `task_done_notification` as a signal to inspect team_snapshot and decide whether to assign follow-up work, verify, or finish.\n"
            "- Treat `task_blocked_notification` as a signal to inspect the blocked task, reply with `message` or `clarification_response`, or reassign/fix the task.\n"
            "- Treat `worker_failed` as a signal to inspect member state and decide whether to reassign work or spawn a replacement.\n"
            "- Treat `idle_notification` as a signal to inspect pending tasks and either assign more work or start completion/shutdown when the team is done.\n"
            "- Answer from available context when safe.\n"
            "- If the user must decide, ask the user one concise question before unblocking the teammate.\n"
            "- Reply to the teammate with `team_send_message`, using type `message` for coordination or `clarification_response` for answers to blocker questions, and include `reply_to` with the original message_id.\n"
            "- Do not create a new team while handling this inbox trigger.\n"
        )

    async def _drain_team_auto_trigger_queue(self) -> bool:
        """Run queued team inbox auto triggers when no bridge queue is configured."""
        while True:
            try:
                prompt = self._team_auto_trigger_queue.get_nowait()
            except asyncio.QueueEmpty:
                return True
            try:
                outcome = await self._execute_input_text(prompt, source="local")
            finally:
                self._team_auto_trigger_queue.task_done()
            if not outcome.continue_running:
                return False

    async def _consume_subagent_notifications(self) -> None:
        """Render delegated agent progress and completion notifications."""
        queue = self._ctx.subagent_events
        if queue is None:
            return

        while True:
            event = await queue.get()
            try:
                from agent_core.task import SubagentProgressEvent, SubagentTaskResult

                if isinstance(event, SubagentProgressEvent):
                    self._on_subagent_progress(event)
                elif isinstance(event, SubagentTaskResult):
                    await self._on_task_completed(event)
            finally:
                queue.task_done()

    async def _run_with_bridge_session(self, session: Any) -> None:
        """Run the interactive loop while continuously draining bridge requests."""
        prompt_task: asyncio.Task | None = None

        try:
            while True:
                if prompt_task is None:
                    prompt_task = asyncio.create_task(self._prompt_async_with_patch_stdout(session))

                done, _ = await asyncio.wait(
                    {prompt_task},
                    timeout=self.BRIDGE_POLL_INTERVAL_SECONDS,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if prompt_task not in done:
                    if self._has_pending_bridge_work():
                        await self._terminate_active_prompt(session, prompt_task)
                        prompt_task = None
                        self._dismiss_active_prompt_line()
                    should_continue = await self._drain_bridge_queue()
                    if not should_continue:
                        if prompt_task is not None:
                            prompt_task.cancel()
                            try:
                                await prompt_task
                            except (asyncio.CancelledError, EOFError, KeyboardInterrupt):
                                pass
                        break
                    continue

                try:
                    user_input = await prompt_task
                except EOFError:
                    self._console.print("\n[yellow]再见！[/yellow]")
                    break
                except KeyboardInterrupt:
                    prompt_task = None
                    continue
                finally:
                    prompt_task = None

                user_input = user_input.strip()
                if not user_input:
                    continue

                if self._is_immediate_local_exit_input(user_input):
                    self._console.print("[yellow]再见！[/yellow]")
                    break

                self._enqueue_local_bridge_input(user_input)
                should_continue = await self._drain_bridge_queue()
                if not should_continue:
                    break
        finally:
            if prompt_task is not None and not prompt_task.done():
                prompt_task.cancel()
                try:
                    await prompt_task
                except (asyncio.CancelledError, EOFError, KeyboardInterrupt):
                    pass

    async def run(self):
        """Run the interactive CLI."""
        from prompt_toolkit import PromptSession
        from prompt_toolkit.styles import Style

        # Print welcome
        self._print_welcome()
        await self._refresh_empty_context_budget_display()

        notification_task = asyncio.create_task(self._consume_subagent_notifications())
        team_inbox_task = asyncio.create_task(self._consume_team_inbox_auto_triggers())

        try:
            if self._bridge_store is not None:
                should_continue = await self._drain_bridge_queue()
                if not should_continue:
                    return

            # Create key bindings
            kb = KeyBindings()

            @kb.add("c-d")
            def _exit(event):  # noqa: D401
                event.app.exit(exception=EOFError)

            @kb.add("enter")
            def _enter(event):  # noqa: D401
                """Accept @ completion with Enter instead of submitting immediately."""
                buffer = event.current_buffer
                complete_state = buffer.complete_state

                # When selecting @skill completion, Enter should apply completion
                # and keep input in the prompt for the user to continue typing.
                if complete_state and buffer.text.lstrip().startswith("@"):
                    completion = complete_state.current_completion
                    if completion is None and complete_state.completions:
                        completion = complete_state.completions[0]

                    if completion is not None:
                        buffer.apply_completion(completion)
                        skill_name, message = parse_at_command(buffer.text)
                        if skill_name and not message and not buffer.text.endswith(" "):
                            buffer.insert_text(" ")
                        return

                buffer.validate_and_handle()

            # Mark as intentionally used
            _ = _exit
            _ = _enter

            # Create slash command completer
            slash_completer = SlashCommandCompleter(self._slash_registry)
            # Create at command completer
            at_completer = AtCommandCompleter(self._at_registry)
            # Use merged completer to handle both / and @
            from prompt_toolkit.completion import merge_completers

            merged_completer = merge_completers([slash_completer, at_completer])
            threaded_completer = ThreadedCompleter(merged_completer)

            # Define style for better visual feedback
            style = Style.from_dict(
                {
                    "completion-menu.completion": "bg:#008888 #ffffff",
                    "completion-menu.completion.current": "bg:#ffffff #000000",
                    "completion-menu.meta.completion": "bg:#00aaaa #000000",
                    "completion-menu.meta.current": "bg:#00ffff #000000",
                    "completion-menu": "bg:#008888 #ffffff",
                    "bottom-toolbar": "noreverse bg:default #777777",
                    "bottom-toolbar.text": "noreverse bg:default #777777",
                }
            )

            # Create prompt session with completer
            session = PromptSession(
                message=lambda: HTML("<ansiblue>>> </ansiblue>"),
                key_bindings=kb,
                completer=threaded_completer,
                complete_while_typing=True,
                auto_suggest=AutoSuggestFromHistory(),
                style=style,
                enable_history_search=True,
                bottom_toolbar=lambda: HTML(self._render_context_budget_toolbar()),
            )

            if self._bridge_store is not None:
                await self._run_with_bridge_session(session)
                return

            prompt_task: asyncio.Task | None = None
            while True:
                try:
                    if prompt_task is None:
                        prompt_task = asyncio.create_task(session.prompt_async())

                    done, _ = await asyncio.wait(
                        {prompt_task},
                        timeout=self.BRIDGE_POLL_INTERVAL_SECONDS,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if prompt_task not in done:
                        should_continue = await self._drain_team_auto_trigger_queue()
                        if not should_continue:
                            prompt_task.cancel()
                            try:
                                await prompt_task
                            except (asyncio.CancelledError, EOFError, KeyboardInterrupt):
                                pass
                            break
                        continue

                    user_input = await prompt_task
                except EOFError:
                    self._console.print("\n[yellow]再见！[/yellow]")
                    break
                except KeyboardInterrupt:
                    prompt_task = None
                    continue
                finally:
                    prompt_task = None

                user_input = user_input.strip()
                if not user_input:
                    continue

                outcome = await self._execute_input_text(user_input)
                if not outcome.continue_running:
                    break
        finally:
            team_inbox_task.cancel()
            try:
                await team_inbox_task
            except asyncio.CancelledError:
                pass
            notification_task.cancel()
            try:
                await notification_task
            except asyncio.CancelledError:
                pass

    def _print_welcome(self):
        """Print welcome message."""
        title = Text(justify="center")
        title.append("Crab", style="bold #ff6b57")
        title.append(" CLI", style="bold #ffd166")

        subtitle = Text(
            "让任务横着走，结果稳稳落地。",
            style="italic #ffb38a",
            justify="center",
        )

        version_text = Text(f"v{get_cli_version()}", style="dim #8ecae6", justify="center")

        tips = Text()
        tips.append("Enter", style="bold white")
        tips.append(" 发送消息，", style="white")
        tips.append("/ + Tab", style="bold cyan")
        tips.append(" 查看命令\n", style="white")
        tips.append("@ + Tab", style="bold cyan")
        tips.append(" 查看技能，", style="white")
        tips.append('@"<path>"<message>', style="bold #9cd7ff")
        tips.append(" 发送图片\n", style="white")
        tips.append("Ctrl+D", style="bold #ffd166")
        tips.append(" 或 ", style="dim")
        tips.append("/exit", style="bold cyan")
        tips.append(" 退出，准备好了就直接开工。", style="dim")

        banner = Group(title, subtitle, version_text, Text(""), tips)

        self._console.print(
            Panel(
                banner,
                title="[bold #ffd166]Welcome Aboard[/bold #ffd166]",
                subtitle="[dim]scuttle mode engaged[/dim]",
                border_style="#ff7a59",
                padding=(1, 2),
            )
        )

        self._console.print()
        self._console.print(f"[dim]工作目录：[/] {self._ctx.working_dir}")
        if self._session_runtime is not None:
            self._console.print(f"[dim]CLI 会话目录：[/] {self._session_runtime.rollout_dir}")
        self._console.print(f"[dim]模型：[/] {self._agent.llm.model}")
        self._console.print(
            f"[dim]工具：[/] bash, resolve_path, read, write, edit, glob, grep, todos"
        )
        self._console.print(f"[dim]Slash 命令：[/] 按 [cyan]/[/cyan] + [cyan]Tab[/cyan] 查看全部")
        self._console.print(f"[dim]技能命令：[/] 按 [cyan]@[/cyan] + [cyan]Tab[/cyan] 查看全部")
        self._console.print(
            "[dim]审批模式：[/] 已关闭（使用 [cyan]/approval on[/cyan] 可开启逐条审批）"
        )
        if self._model_presets:
            self._console.print(f"[dim]模型预设：[/] {', '.join(self._model_presets.keys())}")
        self._console.print()

    def _print_help(self):
        """Print help information."""
        help_text = """
[bold cyan]可用命令：[/bold cyan]

  [blue]help[/blue]          - 显示帮助信息
  [blue]exit[/blue]          - 退出 CLI
  [blue]pwd[/blue]           - 显示当前工作目录
  [blue]ls [path][/blue]     - 列出目录中的文件（AI 也可以完成）
  [blue]/approval on|off|status[/blue] - 控制高风险工具的人类审批开关

[bold cyan]可用工具（供 AI 使用）：[/bold cyan]

  [blue]bash <cmd>[/blue]    - 执行 shell 命令
  [blue]resolve_path <query>[/blue] - 解析近似路径
  [blue]read <file>[/blue]   - 读取文件内容
  [blue]write <file>[/blue]  - 写入文件内容
  [blue]edit <file>[/blue]   - 编辑文件（替换文本）
  [blue]glob <pattern>[/blue]- 按模式查找文件或目录
  [blue]grep <pattern>[/blue] - 搜索文件内容
  [blue]todos[/blue]         - 管理待办事项

[bold cyan]提示：[/boldcyan]

  - 直接自然地输入你的需求，例如“列出所有 Python 文件”
  - 可使用 [blue]@"<path>"<message>[/blue] 或 [blue]@'<path>'<message>[/blue] 发送图片输入
  - AI 会自动使用工具帮助你
"""
        self._console.print(Panel(help_text, border_style="dim"))

    def _print_approval_status(self) -> None:
        status = "已开启" if self._agent.human_in_loop_config.enabled else "已关闭"
        style = "green" if self._agent.human_in_loop_config.enabled else "yellow"
        self._console.print(f"[{style}]审批模式：{status}[/{style}]")

    async def _request_human_approval(
        self,
        request: HumanApprovalRequest,
    ) -> HumanApprovalDecision:
        self._stop_loading(self._loading)
        self._loading = None

        args_preview = json.dumps(request.arguments, ensure_ascii=False, default=str)
        if len(args_preview) > 240:
            args_preview = args_preview[:237].rstrip() + "..."

        command_preview_line = (
            f"[bold]命令预览：[/bold] {request.command_preview}\n"
            if request.command_preview
            else ""
        )
        panel = Panel(
            f"[bold]工具：[/bold] {request.tool_name}\n"
            f"[bold]风险级别：[/bold] {request.risk_level}\n"
            f"[bold]审批原因：[/bold] {request.reason}\n"
            f"{command_preview_line}"
            f"[bold]参数：[/bold] {args_preview}\n"
            "[dim]提示：如果后续想关闭审批，可在本轮结束后输入 /approval off。[/dim]",
            title="[bold yellow]需要审批[/bold yellow]",
            border_style="yellow",
        )
        self._console.print()
        self._console.print(panel)

        approved = await self._prompter.prompt_yes_no(
            "是否批准这次工具调用？如需后续关闭审批，可在本轮结束后使用 /approval off",
            default=False,
        )
        if approved:
            self._console.print("[green]已批准。[/green]")
            self._console.print()
            return HumanApprovalDecision(approved=True)

        reason = await self._prompter.prompt_text("拒绝原因", optional=True)
        self._console.print("[yellow]已拒绝。[/yellow]")
        self._console.print()
        return HumanApprovalDecision(approved=False, reason=reason or None)

    def _print_slash_help(self):
        """Print slash command help information."""
        self._console.print()
        self._console.print("[bold cyan]Slash 命令：[/bold cyan]")
        self._console.print("[dim]输入 / 可查看可用命令，按 Tab 可自动补全[/dim]")
        self._console.print("[bold cyan]技能命令（@）：[/bold cyan]")
        self._console.print("[dim]使用 @<skill-name> 调用技能，按 Tab 可自动补全[/dim]")
        self._console.print("[dim]图片输入可使用 @\"<path>\"<message> 或 @'<path>'<message>[/dim]")
        self._console.print()

        categories = self._slash_registry.get_by_category()
        for category, commands in sorted(categories.items()):
            self._console.print(f"[bold blue]{category}:[/bold blue]")
            for cmd in commands:
                self._console.print(f"  [cyan]/{cmd.name}[/cyan] - {cmd.description}")
        self._console.print()

    def _print_slash_command_detail(self, command_name: str):
        """Print detailed help for a specific slash command."""
        cmd = self._slash_registry.get(command_name)
        if not cmd:
            self._console.print(f"[red]未知命令：/{command_name}[/red]")
            return

        self._console.print()
        self._console.print(
            Panel(
                f"[bold cyan]/{cmd.name}[/bold cyan]\n\n"
                f"[dim]{cmd.description}[/dim]\n\n"
                f"[bold]用法：[/bold] {cmd.usage}\n\n"
                + (
                    f"[bold]示例：[/bold]\n" + "\n".join(f"  - {ex}" for ex in cmd.examples)
                    if cmd.examples
                    else ""
                ),
                title="[bold blue]命令详情[/bold blue]",
                border_style="bright_blue",
            )
        )
        self._console.print()

    def _print_available_skills(self):
        """Print all available @ skills grouped by category."""
        self._console.print()
        self._console.print("[bold cyan]可用技能（@）：[/bold cyan]")
        self._console.print("[dim]可在消息前使用 @<skill-name> 先加载技能[/dim]")
        self._console.print("[dim]图片输入：@\"<path>\"<message> 或 @'<path>'<message>[/dim]")
        self._console.print()

        categories = self._at_registry.get_by_category()
        if not categories:
            self._console.print("[yellow]未找到技能。[/yellow]")
            self._console.print()
            return

        for category, commands in sorted(categories.items()):
            self._console.print(f"[bold blue]{category}:[/bold blue]")
            for cmd in commands:
                self._console.print(f"  [cyan]@{cmd.name}[/cyan] - {cmd.description}")
        self._console.print()

    async def _on_task_completed(self, result: Any):
        """Handle background task completion notification."""
        from agent_core.task import SubagentTaskResult as TaskResult
        from rich.panel import Panel

        if not isinstance(result, TaskResult):
            return

        if result.status == "completed":
            self._console.print(
                f"[dim]\\[bg done] {result.subagent_name} {result.task_id} "
                f"finished in {self._format_duration_compact(result.execution_time_ms / 1000.0)} "
                f"(tokens={self._format_token_count(int(result.total_tokens or 0))}). "
                f"Use /task {result.task_id} to inspect the result.[/dim]"
            )
            return

        if result.status == "failed":
            self._console.print()
            self._console.print(
                Panel(
                    f"[red]✗ 任务失败：[/red]\n"
                    f"[bold]子智能体：[/] {result.subagent_name}\n"
                    f"[bold]任务 ID：[/] {result.task_id}\n"
                    f"[bold]错误：[/] {result.error}",
                    title="[bold blue]后台任务通知[/bold blue]",
                    border_style="red",
                )
            )
            return

        if result.status == "cancelled":
            self._console.print()
            self._console.print(
                Panel(
                    f"[yellow]! 任务已取消：[/yellow]\n"
                    f"[bold]子智能体：[/] {result.subagent_name}\n"
                    f"[bold]任务 ID：[/] {result.task_id}",
                    title="[bold blue]后台任务通知[/bold blue]",
                    border_style="yellow",
                )
            )
            return

    def _on_subagent_progress(self, event: Any) -> None:
        """Render a compact delegated-agent progress line."""
        from agent_core.task import SubagentProgressEvent

        if not isinstance(event, SubagentProgressEvent):
            return
        if event.run_in_background:
            return
        elapsed = max(time.time() - event.created_at.timestamp(), 0.0)
        step = event.current_step or "Running"
        if len(step) > 72:
            step = step[:69] + "..."
        total_tokens = int(getattr(event, "total_tokens", 0) or 0)
        tokens_text = self._format_token_count(total_tokens) if total_tokens > 0 else "-"
        phase = self._classify_subagent_progress_phase(step)
        signature = (event.status, event.steps_completed, phase)
        if self._subagent_progress_signatures.get(event.task_id) == signature:
            return
        self._subagent_progress_signatures[event.task_id] = signature
        if event.status in {"completed", "failed", "cancelled"}:
            self._subagent_progress_signatures.pop(event.task_id, None)

        # Foreground delegated progress should own the terminal while it is
        # streaming updates, otherwise the parent spinner races with it and
        # causes repeated "思考中" lines and visual reordering.
        self._stop_loading(self._loading)
        self._loading = None

        self._console.print(
            f"[dim]\\[subagent] {event.task_id} | "
            f"{event.subagent_name} | {event.description} | "
            f"tools={event.steps_completed} | tokens={tokens_text} | "
            f"{self._format_duration_compact(elapsed)} | {step}"
            "[/dim]"
        )

    @staticmethod
    def _classify_subagent_progress_phase(step: str) -> str:
        if step.startswith("Calling "):
            return step
        if step in {"Completed", "Cancelled", "Failed"}:
            return step
        if step.startswith("Child agent built"):
            return "child_ready"
        if step.startswith("Building child agent"):
            return "child_building"
        return "other"
