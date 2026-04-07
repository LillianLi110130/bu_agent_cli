"""CLI Application for TG Agent.

Contains the main TGAgentCLI class and loading indicator.
Pure UI logic - receives pre-configured Agent and context.
"""

import json
import asyncio
import io
import os
import sys
import threading
import time
import hashlib
from dataclasses import dataclass
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
from agent_core.agent.registry import AgentRegistry
from agent_core.llm import ChatOpenAI
from agent_core.llm.messages import (
    ContentPartImageParam,
    ContentPartTextParam,
    ImageURL,
    SystemMessage,
    UserMessage,
)
from agent_core.plugin import (
    PluginCommandExecutor,
    PluginExecutionError,
    PluginManager,
)
from agent_core.bootstrap.session_bootstrap import (
    WorkspaceInstructionState,
    sync_workspace_agents_md,
)
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import ThreadedCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.lexers import PygmentsLexer
from pygments.lexers import BashLexer
from rich.console import Console
from rich.panel import Panel

from cli.slash_commands import (
    SlashCommand,
    SlashCommandCompleter,
    SlashCommandRegistry,
    is_slash_command,
    parse_slash_command,
)
from agent_core.skill.discovery import builtin_skills_dir, user_skills_dir
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
)
from cli.init_agent import build_init_agent, build_init_user_prompt, validate_init_output
from cli.im_bridge import BridgeRequest, FileBridgeStore
from cli.interactive_input import InteractivePrompter
from cli.model_switch_service import ModelAutoState, ModelSwitchService
from cli.plugins_handler import PluginSlashHandler
from cli.ralph_commands import RalphSlashHandler
from cli.session_runtime import CLISessionRuntime
from config.model_config import (
    ModelPreset,
    get_auto_vision_preset,
    get_default_preset,
    get_image_summary_preset,
    load_model_presets,
)
from tools import SandboxContext, SecurityError

UserInputPayload = str | list[ContentPartTextParam | ContentPartImageParam]


@dataclass
class _ExecutionOutcome:
    """Normalized outcome for one logical input execution."""

    continue_running: bool = True
    final_content: str = ""


class _CLIHumanApprovalHandler:
    """CLI-backed approval handler used by runtime hooks."""

    def __init__(self, cli: "TGAgentCLI"):
        self._cli = cli

    async def request_approval(
        self,
        request: HumanApprovalRequest,
    ) -> HumanApprovalDecision:
        return await self._cli._request_human_approval(request)


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
    """Interactive CLI for TG Agent assistant.

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
        self._workspace_instruction_state = WorkspaceInstructionState()

        if self._bridge_store is not None:
            self._bridge_store.initialize()

        if context.subagent_manager:
            context.subagent_manager.set_result_callback(self._on_task_completed)

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
            base_url = str(preset.get("base_url", "(继承当前配置)"))
            api_key_env = str(preset.get("api_key_env", "OPENAI_API_KEY"))
            vision_marker = " 视觉" if self._preset_supports_vision(name) else ""
            marker = " [green](默认)[/green]" if name == self._default_model_preset else ""
            if name == self._auto_vision_preset:
                marker += " [magenta](自动视觉)[/magenta]"
            if name == self._image_summary_preset:
                marker += " [blue](图像摘要)[/blue]"
            self._console.print(
                f"  [cyan]{name}[/cyan]{marker} -> {model} "
                f"[dim]{vision_marker}[/dim] "
                f"[dim](base_url: {base_url}, key: {api_key_env})[/dim]"
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

    async def _switch_model_preset(self, preset_name: str, *, manual: bool = True) -> bool:
        """Switch to a configured model preset without clearing conversation context."""
        return await self._model_switch_service.switch_model_preset(
            preset_name,
            manual=manual,
            auto_state=self._model_auto_state,
        )

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

    async def _run_slash_command_with_live_capture(self, user_input: str) -> tuple[bool, str]:
        """Execute one slash command with live colored output and plain-text recording."""
        original_console = self._console
        original_model_console = self._model_switch_service._console
        mirror = _ConsoleMirror(original_console)
        self._console = mirror
        self._model_switch_service._console = mirror
        self._store_command_final_content("")
        try:
            handled = await self._handle_slash_command(user_input)
        finally:
            self._console = original_console
            self._model_switch_service._console = original_model_console
        final_content = self._last_command_final_content or mirror.export_text()
        return handled, final_content

    def _start_loading(self, message: str = "思考中") -> _SafeLoadingIndicator:
        """Start a loading animation."""
        loading = _SafeLoadingIndicator(message)
        loading.start()
        time.sleep(0.02)
        return loading

    def _stop_loading(self, loading: _SafeLoadingIndicator | None):
        """Stop the loading animation."""
        if loading:
            loading.stop()

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

    async def _handle_slash_command(self, text: str) -> bool:
        """Handle a slash command.

        Args:
            text: The slash command text

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
            self._agent.clear_history()
            self._console.print("[yellow]会话上下文已重置。[/yellow]")
            return True

        if command_name == "init":
            await self._run_init_agent()
            return True

        if command_name == "tasks":
            if not self._ctx.subagent_manager:
                self._console.print("[yellow]子智能体管理器未初始化。[/yellow]")
                return True
            tasks_info = self._ctx.subagent_manager.list_all_tasks()
            self._console.print("[bold cyan]后台任务：[/bold cyan]")
            self._console.print(tasks_info)
            return True

        if command_name == "task":
            if not self._ctx.subagent_manager:
                self._console.print("[yellow]子智能体管理器未初始化。[/yellow]")
                return True
            if not args:
                self._console.print("[red]用法：/task <task_id>[/red]")
                return True
            task_id = args[0]
            task_info = self._ctx.subagent_manager.get_task_status(task_id)
            if task_info is None:
                self._console.print(f"[red]未找到任务“{task_id}”。[/red]")
                return True
            self._console.print("[bold cyan]任务详情：[/bold cyan]")
            self._console.print(task_info)
            return True

        if command_name == "task_cancel":
            if not self._ctx.subagent_manager:
                self._console.print("[yellow]子智能体管理器未初始化。[/yellow]")
                return True
            if not args:
                self._console.print("[red]用法：/task_cancel <task_id>[/red]")
                return True
            task_id = args[0]
            result = await self._ctx.subagent_manager.cancel_task(task_id)
            self._console.print(result)
            return True

        # Handle history command
        if command_name == "history":
            self._console.print("[dim]命令历史尚未实现。[/dim]")
            return True

        # Handle skills command - list available @ skills
        if command_name == "skills":
            self._print_available_skills()
            return True

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
            )
            return await handler.handle(args)

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
    ) -> str | None:
        """Run the agent with user input and display events."""
        self._step_number = 0
        self._console.print()
        final_response: str | None = None
        active_agent = agent or self._agent

        # Keep workspace instructions synchronized for the main CLI agent.
        # Dedicated helper agents like /init manage their own injection timing.
        if active_agent is self._agent:
            self._maybe_inject_agents_md()

        # Start loading animation
        self._loading = self._start_loading("思考中")

        # Show cancel hint
        self._console.print("[dim]按 q 可取消本次运行[/dim]")

        # Create cancellation event for 'q' key
        import asyncio

        cancel_event = asyncio.Event()

        # Background task to listen for 'q' key
        async def listen_for_cancel():
            """Listen for 'q' key press to cancel the current agent run."""
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
                    self._console.print()
                    self._console.print("[yellow]用户已取消本次运行（q 键）[/yellow]")
                    break

                if isinstance(event, ToolCallEvent):
                    self._stop_loading(self._loading)
                    self._loading = None

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

                    # Restart loading for the next LLM call.
                    self._loading = self._start_loading("思考中")

                elif isinstance(event, ThinkingEvent):
                    self._stop_loading(self._loading)
                    self._loading = None
                    self._console.print(f"[{self.COLOR_THINKING}]思考：{event.content}[/]")

                elif isinstance(event, TextEvent):
                    self._stop_loading(self._loading)
                    self._loading = None
                    self._console.print(event.content, end="")

                elif isinstance(event, FinalResponseEvent):
                    self._stop_loading(self._loading)
                    self._loading = None
                    final_response = event.content
                    # Check if this was a cancellation
                    if event.content == "[Cancelled by user]":
                        self._console.print()
                        self._console.print("[yellow]用户已取消本次运行（q 键）[/yellow]")
                    else:
                        self._console.print()
                        self._console.print()

        except Exception as e:
            self._stop_loading(self._loading)
            self._loading = None
            self._console.print(f"[{self.COLOR_ERROR}]错误：{e}[/]")
        finally:
            # Cancel the listener task
            cancel_event.set()
            listener_task.cancel()
            try:
                await listener_task
            except (asyncio.CancelledError, Exception):
                pass

        # Ensure loading is stopped
        self._stop_loading(self._loading)
        self._loading = None
        self._console.print()
        return final_response

    async def _on_task_completed(self, result: Any):
        """Handle background task completion notification.

        Args:
            result: TaskResult from SubagentManager
        """
        from agent_core.agent.subagent_manager import TaskResult
        from rich.panel import Panel

        if not isinstance(result, TaskResult):
            return

        # Display completion notification
        status_emoji = "✅" if result.status == "completed" else "❌"
        status_color = "green" if result.status == "completed" else "red"

        if result.status == "completed":
            self._console.print()
            self._console.print(
                Panel(
                    (
                        f"[{status_color}]{status_emoji} 任务已完成：[/{status_color}]\n"
                        f"[bold]子智能体：[/] {result.subagent_name}\n"
                        f"[bold]任务 ID：[/] {result.task_id}\n"
                        f"[bold]执行耗时：[/] {result.execution_time_ms:.0f}ms\n"
                        f"[bold]使用工具：[/] {', '.join(result.tools_used) if result.tools_used else '无'}\n"
                        f"[bold]结果：[/] {result.final_response[:500]}..."
                        if len(result.final_response) > 500
                        else "..."
                    ),
                    title="[bold blue]后台任务通知[/bold blue]",
                    border_style=status_color,
                )
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

    async def _execute_input_text(self, user_input: str) -> _ExecutionOutcome:
        """Execute one normalized line of user input."""
        user_input = user_input.strip()
        if not user_input:
            return _ExecutionOutcome()

        if self._session_runtime is not None:
            self._session_runtime.touch()

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
            final_content = await self._run_agent(parsed.content_parts, has_image=True)
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
                handled, final_content = await self._run_slash_command_with_live_capture(user_input)
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
        final_content = await self._run_agent(user_input, has_image=False)
        return _ExecutionOutcome(final_content=final_content or "")

    async def _drain_bridge_queue(self) -> bool:
        """Consume queued bridge requests until no pending work remains."""
        if self._bridge_store is None:
            return True

        while True:
            request = self._bridge_store.claim_next_pending()
            if request is None:
                return True

            if request.source == "remote":
                preview = request.content.strip().replace("\n", " ")
                if len(preview) > 80:
                    preview = preview[:77] + "..."
                self._console.print(
                    f"\n[bold cyan]收到远程任务[/bold cyan] "
                    f"[dim](seq={request.seq}, request_id={request.request_id})[/dim]"
                )
                if preview:
                    self._console.print(f"[dim]{preview}[/dim]")

            try:
                outcome = await self._execute_input_text(request.content)
            except Exception as exc:
                self._bridge_store.fail_request(
                    request,
                    final_content=f"执行失败：{exc}",
                    error_code="LOCAL_BRIDGE_EXECUTION_ERROR",
                    error_message=str(exc),
                )
                raise

            self._bridge_store.complete_request(request, final_content=outcome.final_content)
            if not outcome.continue_running:
                return False

    async def _run_with_bridge_session(self, session: Any) -> None:
        """Run the interactive loop while continuously draining bridge requests."""
        prompt_task: asyncio.Task | None = None

        try:
            while True:
                if prompt_task is None:
                    prompt_task = asyncio.create_task(session.prompt_async())

                done, _ = await asyncio.wait(
                    {prompt_task},
                    timeout=self.BRIDGE_POLL_INTERVAL_SECONDS,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if prompt_task not in done:
                    should_continue = await self._drain_bridge_queue()
                    if not should_continue:
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
        from prompt_toolkit.patch_stdout import patch_stdout
        from prompt_toolkit.styles import Style

        # Print welcome
        self._print_welcome()

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
        )

        if self._bridge_store is not None:
            with patch_stdout(raw=True):
                await self._run_with_bridge_session(session)
            return

        while True:
            try:
                user_input = await session.prompt_async()
            except EOFError:
                self._console.print("\n[yellow]再见！[/yellow]")
                break
            except KeyboardInterrupt:
                continue

            user_input = user_input.strip()
            if not user_input:
                continue

            outcome = await self._execute_input_text(user_input)
            if not outcome.continue_running:
                break

    def _print_welcome(self):
        """Print welcome message."""
        self._console.print(
            Panel(
                f"[bold cyan]TG Agent CLI[/bold cyan]\n\n"
                f"输入消息后按 Enter 发送。\n"
                f"按 [cyan]/[/cyan] 可查看可用命令。\n"
                f"按 [cyan]@[/cyan] + [cyan]Tab[/cyan] 可查看可用技能。\n"
                f'可使用 [cyan]@"<path>"<message>[/cyan] 或 '
                f"[cyan]@'<path>'<message>[/cyan] 发送图片输入。\n"
                f"按 Ctrl+D 或输入 [cyan]/exit[/cyan] 退出。\n",
                title="[bold blue]欢迎[/bold blue]",
                border_style="bright_blue",
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
  [blue]glob <pattern>[/blue]- 按模式查找文件
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
        from agent_core.agent.subagent_manager import TaskResult
        from rich.panel import Panel

        if not isinstance(result, TaskResult):
            return

        if result.status == "completed":
            status_symbol = "✓"
            status_color = "green"
            result_preview = (
                f"{result.final_response[:500]}..."
                if len(result.final_response) > 500
                else result.final_response
            )
            self._console.print()
            self._console.print(
                Panel(
                    f"[{status_color}]{status_symbol} 任务已完成：[/{status_color}]\n"
                    f"[bold]子智能体：[/] {result.subagent_name}\n"
                    f"[bold]任务 ID：[/] {result.task_id}\n"
                    f"[bold]执行耗时：[/] {result.execution_time_ms:.0f}ms\n"
                    f"[bold]使用工具：[/] {', '.join(result.tools_used) if result.tools_used else '无'}\n"
                    f"[bold]结果：[/] {result_preview}",
                    title="[bold blue]后台任务通知[/bold blue]",
                    border_style=status_color,
                )
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
