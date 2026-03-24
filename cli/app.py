"""CLI Application for Claude Code.

Contains the main ClaudeCodeCLI class and loading indicator.
Pure UI logic - receives pre-configured Agent and context.
"""

import json
import os
import sys
import threading
import time
import hashlib
from pathlib import Path
from typing import Any, Callable

from bu_agent_sdk import Agent
from bu_agent_sdk.agent import (
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
from bu_agent_sdk.agent.registry import AgentRegistry
from bu_agent_sdk.llm import ChatOpenAI
from bu_agent_sdk.llm.messages import (
    ContentPartImageParam,
    ContentPartTextParam,
    ImageURL,
    SystemMessage,
    UserMessage,
)
from bu_agent_sdk.plugin import (
    PluginCommandExecutor,
    PluginExecutionError,
    PluginManager,
)
from bu_agent_sdk.bootstrap.session_bootstrap import (
    WorkspaceInstructionState,
    sync_workspace_agents_md,
)
from bu_agent_sdk.llm.messages import SystemMessage, UserMessage
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
from cli.interactive_input import InteractivePrompter
from cli.model_switch_service import ModelAutoState, ModelSwitchService
from cli.plugins_handler import PluginSlashHandler
from cli.ralph_commands import RalphSlashHandler
from tools import SandboxContext, SecurityError

ModelPreset = dict[str, str | bool]
UserInputPayload = str | list[ContentPartTextParam | ContentPartImageParam]


class _CLIHumanApprovalHandler:
    """CLI-backed approval handler used by runtime hooks."""

    def __init__(self, cli: "ClaudeCodeCLI"):
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

    def __init__(self, message: str = "Thinking"):
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


# =============================================================================
# CLI Application
# =============================================================================


class ClaudeCodeCLI:
    """Interactive CLI for Claude Code assistant.

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
        self._loading: _LoadingIndicator | None = None
        self._prompter = InteractivePrompter(self._console)
        self._slash_registry = slash_registry or SlashCommandRegistry()
        self._at_registry = at_registry or AtCommandRegistry(
            Path(__file__).resolve().parent.parent / "bu_agent_sdk" / "skills"
        )
        self._agent_registry = agent_registry
        self._plugin_manager = plugin_manager
        self._plugin_executor = PluginCommandExecutor()
        self._system_prompt_builder = system_prompt_builder
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
        self._model_pick_active = False
        self._model_pick_order: list[str] = []
        self._agents_md_hash: str | None = None
        self._agents_md_content: str | None = None
        self._ralph_handler: RalphSlashHandler | None = None
        self._approval_handler = _CLIHumanApprovalHandler(self)
        self._agent.human_in_loop_handler = self._approval_handler
        self._agent.human_in_loop_config = HumanInLoopConfig(enabled=True)
        self._agent.register_hook(
            ModelRoutingHook(
                service=self._model_switch_service,
                auto_state=self._model_auto_state,
            )
        )
        self._workspace_instruction_state = WorkspaceInstructionState()

        if context.subagent_manager:
            context.subagent_manager.set_result_callback(self._on_task_completed)

    def _load_model_presets(self) -> dict[str, ModelPreset]:
        """Load model presets from config/model_presets.json."""
        if not self._model_presets_path.exists():
            return {}

        try:
            raw = self._model_presets_path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except Exception as e:
            self._console.print(
                f"[yellow]Failed to load model presets: {e}[/yellow]"
            )
            return {}

        if not isinstance(data, dict):
            self._console.print(
                "[yellow]model_presets.json must be a JSON object.[/yellow]"
            )
            return {}

        default_name = data.get("default")
        if isinstance(default_name, str) and default_name.strip():
            self._default_model_preset = default_name.strip()

        auto_vision_name = data.get("auto_vision_preset")
        if isinstance(auto_vision_name, str) and auto_vision_name.strip():
            self._auto_vision_preset = auto_vision_name.strip()

        image_summary_name = data.get("image_summary_preset")
        if isinstance(image_summary_name, str) and image_summary_name.strip():
            self._image_summary_preset = image_summary_name.strip()

        preset_data = data.get("presets")
        if not isinstance(preset_data, dict):
            return {}

        presets: dict[str, ModelPreset] = {}
        for name, config in preset_data.items():
            if not isinstance(name, str) or not isinstance(config, dict):
                continue

            model = config.get("model")
            if not isinstance(model, str) or not model.strip():
                continue

            cleaned: ModelPreset = {"model": model.strip()}

            base_url = config.get("base_url")
            if isinstance(base_url, str) and base_url.strip():
                cleaned["base_url"] = base_url.strip()

            api_key_env = config.get("api_key_env", "OPENAI_API_KEY")
            if isinstance(api_key_env, str) and api_key_env.strip():
                cleaned["api_key_env"] = api_key_env.strip()
            else:
                cleaned["api_key_env"] = "OPENAI_API_KEY"

            cleaned["vision"] = bool(config.get("vision", False))
            presets[name.strip()] = cleaned

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
        """Inject workspace AGENTS.md into context if present."""
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

        # Tree (depth 4)
        tree_lines: list[str] = []
        for current, dirs, files in os.walk(root):
            rel = os.path.relpath(current, root)
            depth = 0 if rel == "." else rel.count(os.sep) + 1
            if depth > 4:
                dirs[:] = []
                continue
            dirs[:] = [d for d in dirs if not is_ignored_dir(d)]
            indent = "  " * depth
            tree_lines.append(f"{indent}{os.path.basename(current)}/")
            for f in sorted(files):
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
            if path.exists():
                content = path.read_text(encoding="utf-8")[:4000]
                file_snippets.append(f"## {name}\n{content}")

        # Files (depth-limited, read all with truncation)
        file_snippets_all: list[str] = []
        for current, dirs, files in os.walk(root):
            rel = os.path.relpath(current, root)
            depth = 0 if rel == "." else rel.count(os.sep) + 1
            if depth > 4:
                dirs[:] = []
                continue
            dirs[:] = [d for d in dirs if not is_ignored_dir(d)]
            for f in sorted(files):
                path = Path(current) / f
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
        base_url_display = str(base_url) if base_url else "(default)"
        preset_name = self._resolve_current_preset_name()
        preset_line = preset_name or "(unmatched)"
        if preset_name and self._preset_supports_vision(preset_name):
            preset_line += " [vision]"
        self._console.print(f"Current model: [cyan]{model}[/cyan]")
        self._console.print(f"Preset: [dim]{preset_line}[/dim]")
        self._console.print(f"Base URL: [dim]{base_url_display}[/dim]")
        self._console.print(f"Context messages: [dim]{len(self._agent.messages)}[/dim]")

    def _print_model_presets(self):
        """Print configured model presets."""
        if not self._model_presets:
            self._console.print(
                f"[yellow]No model presets found at {self._model_presets_path}[/yellow]"
            )
            return

        self._console.print("[bold cyan]Model presets:[/bold cyan]")
        for name, preset in self._model_presets.items():
            model = str(preset["model"])
            base_url = str(preset.get("base_url", "(inherit current)"))
            api_key_env = str(preset.get("api_key_env", "OPENAI_API_KEY"))
            vision_marker = " vision" if self._preset_supports_vision(name) else ""
            marker = (
                " [green](default)[/green]"
                if name == self._default_model_preset
                else ""
            )
            if name == self._auto_vision_preset:
                marker += " [magenta](auto-vision)[/magenta]"
            if name == self._image_summary_preset:
                marker += " [blue](image-summary)[/blue]"
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
            self._console.print("[yellow]No model presets configured.[/yellow]")
            self._model_pick_active = False
            self._model_pick_order = []
            return

        self._model_pick_order = list(self._model_presets.keys())
        self._model_pick_active = True
        current_preset = self._resolve_current_preset_name()

        self._console.print()
        self._console.print("[bold cyan]Select a model preset:[/bold cyan]")
        for idx, name in enumerate(self._model_pick_order, 1):
            preset = self._model_presets[name]
            model = str(preset["model"])
            markers: list[str] = []
            if name == current_preset:
                markers.append("current")
            if name == self._default_model_preset:
                markers.append("default")
            if self._preset_supports_vision(name):
                markers.append("vision")
            if name == self._auto_vision_preset:
                markers.append("auto-vision")
            if name == self._image_summary_preset:
                markers.append("image-summary")
            marker_text = f" [dim]({', '.join(markers)})[/dim]" if markers else ""
            self._console.print(f"  {idx}. [cyan]{name}[/cyan] -> {model}{marker_text}")
        self._console.print(
            "[dim]Type the number and press Enter to switch, or 'q' to cancel.[/dim]"
        )

    async def _handle_model_pick_input(self, user_input: str) -> bool:
        """Handle one line of input while in numbered model-pick mode."""
        if not self._model_pick_active:
            return False

        value = user_input.strip()
        if not value:
            self._console.print("[dim]Enter a number, or 'q' to cancel.[/dim]")
            return True

        if value.lower() in {"q", "quit", "cancel", "exit"}:
            self._model_pick_active = False
            self._model_pick_order = []
            self._console.print("[yellow]Model selection cancelled.[/yellow]")
            return True

        if not value.isdigit():
            self._console.print("[red]Invalid selection. Please enter a number.[/red]")
            return True

        index = int(value)
        if index < 1 or index > len(self._model_pick_order):
            self._console.print(
                f"[red]Selection out of range. Choose 1-{len(self._model_pick_order)}.[/red]"
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

    def _start_loading(self, message: str = "Thinking") -> _LoadingIndicator:
        """Start a loading animation."""
        loading = _LoadingIndicator(message)
        loading.start()
        time.sleep(0.02)
        return loading

    def _stop_loading(self, loading: _LoadingIndicator | None):
        """Stop the loading animation."""
        if loading:
            loading.stop()

    def _print_welcome(self):
        """Print welcome message."""
        self._console.print(
            Panel(
                f"[bold cyan]Claude Code CLI[/bold cyan]\n\n"
                f"Type your message and press Enter to send.\n"
                f"Press [cyan]/[/cyan] to see available commands.\n"
                f"Press [cyan]@[/cyan] + [cyan]Tab[/cyan] to see available skills.\n"
                f"Use [cyan]@\"<path>\"<message>[/cyan] or "
                f"[cyan]@'<path>'<message>[/cyan] for image input.\n"
                f"Press Ctrl+D or type [cyan]/exit[/cyan] to quit.\n",
                title="[bold blue]Welcome[/bold blue]",
                border_style="bright_blue",
            )
        )

        # Show sandbox info
        self._console.print()
        self._console.print(f"[dim]Working directory:[/] {self._ctx.working_dir}")
        self._console.print(f"[dim]Model:[/] {self._agent.llm.model}")
        self._console.print(f"[dim]Tools:[/] bash, read, write, edit, glob, grep, todos")
        self._console.print(f"[dim]Slash Commands:[/] Press [cyan]/[/cyan] + [cyan]Tab[/cyan] to see all")
        self._console.print(f"[dim]Skill Commands:[/] Press [cyan]@[/cyan] + [cyan]Tab[/cyan] to see all")
        self._console.print(
            "[dim]审批模式：[/]已开启（使用 [cyan]/approval off[/cyan] 可关闭逐条审批）"
        )
        if self._model_presets:
            self._console.print(
                f"[dim]Model presets:[/] {', '.join(self._model_presets.keys())}"
            )
        self._console.print()

    def _print_help(self):
        """Print help information."""
        help_text = """
[bold cyan]Available Commands:[/bold cyan]

  [blue]help[/blue]          - Show this help message
  [blue]exit[/blue]          - Exit the CLI
  [blue]pwd[/blue]           - Print current working directory
  [blue]ls [path][/blue]     - List files in directory (AI can do this too)
  [blue]/approval on|off|status[/blue] - 控制高风险工具的人类审批开关

[bold cyan]Available Tools (for AI):[/bold cyan]

  [blue]bash <cmd>[/blue]    - Run shell commands
  [blue]read <file>[/blue]   - Read file contents
  [blue]write <file>[/blue]  - Write content to file
  [blue]edit <file>[/blue]   - Edit file (replace text)
  [blue]glob <pattern>[/blue]- Find files by pattern
  [blue]grep <pattern>[/blue] - Search file contents
  [blue]todos[/blue]         - Manage todo list

[bold cyan]Tips:[/boldcyan]

  - Just type your request naturally, e.g., "List all Python files"
  - Send image input with [blue]@"<path>"<message>[/blue] or [blue]@'<path>'<message>[/blue]
  - The AI will use tools automatically to help you
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
            "[dim]提示：如需关闭后续审批，可在本轮结束后输入 /approval off。[/dim]",
            title="[bold yellow]需要审批[/bold yellow]",
            border_style="yellow",
        )
        self._console.print()
        self._console.print(panel)

        approved = await self._prompter.prompt_yes_no(
            "是否批准这次工具调用？如需关闭后续审批，可在本轮结束后使用 /approval off",
            default=False,
        )
        if approved:
            self._console.print("[green]已批准。[/green]")
            self._console.print()
            return HumanApprovalDecision(approved=True)

        reason = await self._prompter.prompt_text(
            "拒绝原因",
            optional=True,
        )
        self._console.print("[yellow]已拒绝。[/yellow]")
        self._console.print()
        return HumanApprovalDecision(approved=False, reason=reason or None)

    def _print_slash_help(self):
        """Print slash command help information."""
        self._console.print()
        self._console.print("[bold cyan]Slash Commands:[/bold cyan]")
        self._console.print("[dim]Press / to see available commands, Tab to autocomplete[/dim]")
        self._console.print("[bold cyan]Skill Commands (@):[/bold cyan]")
        self._console.print("[dim]Use @<skill-name> to invoke skills, Tab to autocomplete[/dim]")
        self._console.print(
            "[dim]Use @\"<path>\"<message> or @'<path>'<message> for image input[/dim]"
        )
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
            self._console.print(f"[red]Unknown command: /{command_name}[/red]")
            return

        self._console.print()
        self._console.print(Panel(
            f"[bold cyan]/{cmd.name}[/bold cyan]\n\n"
            f"[dim]{cmd.description}[/dim]\n\n"
            f"[bold]Usage:[/bold] {cmd.usage}\n\n"
            + (f"[bold]Examples:[/bold]\n" + "\n".join(f"  • {ex}" for ex in cmd.examples) if cmd.examples else ""),
            title="[bold blue]Command Details[/bold blue]",
            border_style="bright_blue",
        ))
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
            self._console.print("[yellow]Goodbye![/yellow]")
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
                self._console.print("[red]Usage: /model [show|list|<preset>][/red]")
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
            self._console.print("[yellow]Conversation context reset.[/yellow]")
            return True

        if command_name == "init":
            # Generate docs/PROJECT.md with an AI summary of the project
            out_path = self._ctx.working_dir / "AGENTS.md"

            snapshot = self._build_project_snapshot()
            system = SystemMessage(
                content=(
                    "The user just ran `/init`.\n"
                    "Generate AGENTS.md based on the project snapshot.\n"
                    "Keep it concise and useful for onboarding.\n"
                    "内容要求中文"
                )
            )
            user = UserMessage(
                content=(
                    "Based on the project snapshot below, write AGENTS.md with:\n"
                    "1) Overview\n2) Project Structure\n3) How It Works\n"
                    "4) Constraints/Assumptions\n\n"
                    f"{snapshot}"
                )
            )

            response = await self._agent.llm.ainvoke(
                messages=[system, user],
                tools=None,
                tool_choice=None,
            )
            content = response.content or ""
            # Strip any hidden thinking blocks from the output
            if content:
                import re
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE)
                content = re.sub(r"<analysis>.*?</analysis>", "", content, flags=re.DOTALL | re.IGNORECASE)
            out_path.write_text(content, encoding="utf-8")
            self._console.print("[yellow]Generated AGENTS.md[/yellow]")
            return True

        if command_name == "tasks":
            if not self._ctx.subagent_manager:
                self._console.print("[yellow]Subagent manager not initialized.[/yellow]")
                return True
            tasks_info = self._ctx.subagent_manager.list_all_tasks()
            self._console.print("[bold cyan]Background Tasks:[/bold cyan]")
            self._console.print(tasks_info)
            return True

        if command_name == "task":
            if not self._ctx.subagent_manager:
                self._console.print("[yellow]Subagent manager not initialized.[/yellow]")
                return True
            if not args:
                self._console.print("[red]Usage: /task <task_id>[/red]")
                return True
            task_id = args[0]
            task_info = self._ctx.subagent_manager.get_task_status(task_id)
            if task_info is None:
                self._console.print(f"[red]Task '{task_id}' not found.[/red]")
                return True
            self._console.print("[bold cyan]Task Details:[/bold cyan]")
            self._console.print(task_info)
            return True

        if command_name == "task_cancel":
            if not self._ctx.subagent_manager:
                self._console.print("[yellow]Subagent manager not initialized.[/yellow]")
                return True
            if not args:
                self._console.print("[red]Usage: /task_cancel <task_id>[/red]")
                return True
            task_id = args[0]
            result = await self._ctx.subagent_manager.cancel_task(task_id)
            self._console.print(result)
            return True

        # Handle history command
        if command_name == "history":
            self._console.print("[dim]Command history not implemented yet.[/dim]")
            return True

        # Handle skills command - list available @ skills
        if command_name == "skills":
            self._print_available_skills()
            return True

        # Handle allow command - add directory to sandbox
        if command_name == "allow":
            if not args:
                self._console.print("[red]Usage: /allow <path>[/red]")
                self._console.print("[dim]Example: /allow /path/to/project[/dim]")
            else:
                path_str = " ".join(args)
                try:
                    added_path = self._ctx.add_allowed_dir(path_str)
                    self._console.print(f"[green]Added to allowed directories:[/] {added_path}")
                except SecurityError as e:
                    self._console.print(f"[red]{e}[/red]")
            return True

        # Handle allowed command - list allowed directories
        if command_name == "allowed":
            self._console.print()
            self._console.print("[bold cyan]Allowed Directories:[/bold cyan]")
            for i, allowed_dir in enumerate(self._ctx.allowed_dirs, 1):
                # 标记当前工作目录
                marker = " [dim](current)[/]" if str(allowed_dir.resolve()) == str(self._ctx.working_dir.resolve()) else ""
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
                self._console.print("[yellow]Plugin manager not configured.[/yellow]")
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
        self._console.print(f"[red]Unknown command: /{command_name}[/red]")
        self._console.print(f"[dim]Type /help for available commands.[/dim]")
        return True

    def _refresh_system_prompt(self) -> None:
        """Rebuild the agent system prompt after plugin registry changes."""
        if self._system_prompt_builder is None:
            return
        self._agent.system_prompt = self._system_prompt_builder()
        self._agent.clear_history()
        self._console.print("[yellow]Conversation context reset after plugin reload.[/yellow]")

    def _print_available_skills(self):
        """Print all available @ skills grouped by category."""
        self._console.print()
        self._console.print("[bold cyan]Available Skills (@):[/bold cyan]")
        self._console.print(
            "[dim]Use @<skill-name> to load a skill before your message[/dim]"
        )
        self._console.print(
            "[dim]Image input: @\"<path>\"<message> or @'<path>'<message>[/dim]"
        )
        self._console.print()

        categories = self._at_registry.get_by_category()
        if not categories:
            self._console.print("[yellow]No skills found.[/yellow]")
            self._console.print()
            return

        for category, commands in sorted(categories.items()):
            self._console.print(f"[bold blue]{category}:[/bold blue]")
            for cmd in commands:
                self._console.print(f"  [cyan]@{cmd.name}[/cyan] - {cmd.description}")
        self._console.print()

    async def _handle_at_command(self, text: str) -> bool:
        """Handle an @ command for skill invocation."""
        skill_name, message = parse_at_command(text)
        if not skill_name:
            self._console.print("[yellow]Invalid @ command.[/yellow]")
            self._console.print("[dim]Type @ and press Tab to see available skills.[/dim]")
            return True

        skill = self._at_registry.get(skill_name)
        if not skill:
            self._console.print(f"[yellow]Skill not found: @{skill_name}[/yellow]")
            self._console.print("[dim]Use /skills to list available skills.[/dim]")
            return True

        self._console.print(f"[cyan]Using @{skill.name}...[/cyan]")
        try:
            expanded_message = expand_at_command(skill, message)
        except (IOError, ValueError) as e:
            self._console.print(f"[red]Failed to load skill: {e}[/red]")
            return True

        await self._run_agent(expanded_message, has_image=False)
        return True

    async def _run_agent(self, user_input: UserInputPayload, has_image: bool = False):
        """Run the agent with user input and display events."""
        self._step_number = 0
        self._console.print()

        # Inject AGENTS.md (if present) before each user query
        # self._maybe_inject_agents_md()

        # Start loading animation
        self._loading = self._start_loading("Thinking")

        # Show cancel hint
        self._console.print("[dim]Press 'q' to cancel this run[/dim]")

        # Create cancellation event for 'q' key
        import asyncio
        cancel_event = asyncio.Event()

        # Background task to listen for 'q' key
        async def listen_for_cancel():
            """Listen for 'q' key press to cancel the current agent run."""
            import sys
            import os

            if os.name == 'nt':  # Windows
                try:
                    import msvcrt
                    while not cancel_event.is_set():
                        if msvcrt.kbhit():  # Check if key is available
                            ch = msvcrt.getwch()  # Get wide char (Unicode)
                            if ch.lower() == 'q':
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
                            if ch.lower() == 'q':
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
            async for event in self._agent.query_stream(user_input, cancel_event=cancel_event):
                # Cancellation is now handled inside query_stream for immediate response
                # This check is for final confirmation
                if cancel_event.is_set():
                    self._console.print()
                    self._console.print("[yellow]Agent run cancelled by user (q key)[/yellow]")
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
                        f"[{self.COLOR_TOOL_CALL}]Step {self._step_number}:[/] "
                        f"[bold]{event.tool}[/]"
                    )
                    self._console.print(f"  [dim]Args: {args_str}[/]")
                    self._console.print("  [dim](Press 'q' to cancel)[/dim]")

                    # Start "Executing" loading while tool runs
                    self._loading = self._start_loading("Executing")

                elif isinstance(event, ToolResultEvent):
                    self._stop_loading(self._loading)
                    self._loading = None

                    result = str(event.result)
                    if event.is_error:
                        self._console.print(f"  [{self.COLOR_ERROR}]Error: {result[:200]}[/]")
                    else:
                        if len(result) > 300:
                            self._console.print(
                                f"  [{self.COLOR_TOOL_RESULT}]Result: {result[:300]}...[/]"
                            )
                        else:
                            self._console.print(f"  [{self.COLOR_TOOL_RESULT}]Result: {result}[/]")

                    # Restart "Thinking" loading for next LLM call
                    self._loading = self._start_loading("Thinking")

                elif isinstance(event, ThinkingEvent):
                    self._stop_loading(self._loading)
                    self._loading = None
                    self._console.print(f"[{self.COLOR_THINKING}]Thinking: {event.content}[/]")

                elif isinstance(event, TextEvent):
                    self._stop_loading(self._loading)
                    self._loading = None
                    self._console.print(event.content, end="")

                elif isinstance(event, FinalResponseEvent):
                    self._stop_loading(self._loading)
                    self._loading = None
                    # Check if this was a cancellation
                    if event.content == "[Cancelled by user]":
                        self._console.print()
                        self._console.print("[yellow]Agent run cancelled by user (q key)[/yellow]")
                    else:
                        self._console.print()
                        self._console.print()

        except Exception as e:
            self._stop_loading(self._loading)
            self._loading = None
            self._console.print(f"[{self.COLOR_ERROR}]Error: {e}[/]")
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

    async def _on_task_completed(self, result: Any):
        """Handle background task completion notification.

        Args:
            result: TaskResult from SubagentManager
        """
        from bu_agent_sdk.agent.subagent_manager import TaskResult
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
                    f"[{status_color}]{status_emoji} Task Completed:[/{status_color}]\n"
                    f"[bold]Subagent:[/] {result.subagent_name}\n"
                    f"[bold]Task ID:[/] {result.task_id}\n"
                    f"[bold]Execution Time:[/] {result.execution_time_ms:.0f}ms\n"
                    f"[bold]Tools Used:[/] {', '.join(result.tools_used) if result.tools_used else 'None'}\n"
                    f"[bold]Result:[/] {result.final_response[:500]}..."
                    if len(result.final_response) > 500 else "...",
                    title="[bold blue]Background Task Notification[/bold blue]",
                    border_style=status_color,
                )
            )
        elif result.status == "failed":
            self._console.print()
            self._console.print(
                Panel(
                    f"[{status_color}]{status_emoji} Task Failed:[/{status_color}]\n"
                    f"[bold]Subagent:[/] {result.subagent_name}\n"
                    f"[bold]Task ID:[/] {result.task_id}\n"
                    f"[bold]Error:[/] {result.error}",
                    title="[bold blue]Background Task Notification[/bold blue]",
                    border_style=status_color,
                )
            )
        elif result.status == "cancelled":
            self._console.print()
            self._console.print(
                Panel(
                    f"[yellow]⏹️  Task Cancelled:[/yellow]\n"
                    f"[bold]Subagent:[/] {result.subagent_name}\n"
                    f"[bold]Task ID:[/] {result.task_id}",
                    title="[bold blue]Background Task Notification[/bold blue]",
                    border_style="yellow",
                )
            )

    async def run(self):
        """Run the interactive CLI."""
        from prompt_toolkit import PromptSession
        from prompt_toolkit.styles import Style

        # Print welcome
        self._print_welcome()

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
        style = Style.from_dict({
            "completion-menu.completion": "bg:#008888 #ffffff",
            "completion-menu.completion.current": "bg:#ffffff #000000",
            "completion-menu.meta.completion": "bg:#00aaaa #000000",
            "completion-menu.meta.current": "bg:#00ffff #000000",
            "completion-menu": "bg:#008888 #ffffff",
        })

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

        while True:
            try:
                user_input = await session.prompt_async()
            except EOFError:
                self._console.print("\n[yellow]Goodbye![/yellow]")
                break
            except KeyboardInterrupt:
                continue

            user_input = user_input.strip()
            if not user_input:
                continue

            # Handle numbered model picker mode
            if self._model_pick_active:
                if await self._handle_model_pick_input(user_input):
                    continue

            # Handle quoted @ image command
            if is_image_command(user_input):
                try:
                    parsed = parse_image_command(user_input, self._ctx)
                except ImageInputError as e:
                    self._console.print(f"[red]{e}[/red]")
                    self._console.print(f"[dim]{IMAGE_USAGE}[/dim]")
                    continue
                await self._run_agent(parsed.content_parts, has_image=True)
                continue

            # Handle @ commands (skill invocation)
            if is_at_command(user_input):
                try:
                    if await self._handle_at_command(user_input):
                        continue
                except EOFError:
                    break
                continue

            # Handle slash commands
            if is_slash_command(user_input):
                try:
                    if await self._handle_slash_command(user_input):
                        continue
                except EOFError:
                    break
                continue

            # Handle legacy built-in commands (without slash)
            if user_input.lower() in ["exit", "quit"]:
                self._console.print("[yellow]Goodbye![/yellow]")
                break

            if user_input.lower() == "help":
                self._print_help()
                continue

            if user_input.lower() == "pwd":
                self._console.print(f"Current directory: {self._ctx.working_dir}")
                continue

            # Run agent
            await self._run_agent(user_input, has_image=False)
