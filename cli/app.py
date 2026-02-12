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
from typing import Any

from bu_agent_sdk import Agent
from bu_agent_sdk.agent import (
    FinalResponseEvent,
    TextEvent,
    ThinkingEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from bu_agent_sdk.llm import ChatOpenAI
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
from tools import SandboxContext, SecurityError


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

    def __init__(self, agent: Agent, context: SandboxContext):
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
        self._slash_registry = SlashCommandRegistry()
        self._at_registry = AtCommandRegistry(
            Path(__file__).resolve().parent.parent / "bu_agent_sdk" / "skills"
        )
        self._model_presets_path = (
            Path(__file__).resolve().parent.parent
            / "config"
            / "model_presets.json"
        )
        self._default_model_preset: str | None = None
        self._model_presets = self._load_model_presets()
        self._model_pick_active = False
        self._model_pick_order: list[str] = []
        self._agents_md_hash: str | None = None
        self._agents_md_content: str | None = None

        if context.subagent_manager:
            context.subagent_manager.set_result_callback(self._on_task_completed)

    def _load_model_presets(self) -> dict[str, dict[str, str]]:
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

        preset_data = data.get("presets")
        if not isinstance(preset_data, dict):
            return {}

        presets: dict[str, dict[str, str]] = {}
        for name, config in preset_data.items():
            if not isinstance(name, str) or not isinstance(config, dict):
                continue

            model = config.get("model")
            if not isinstance(model, str) or not model.strip():
                continue

            cleaned: dict[str, str] = {"model": model.strip()}

            base_url = config.get("base_url")
            if isinstance(base_url, str) and base_url.strip():
                cleaned["base_url"] = base_url.strip()

            api_key_env = config.get("api_key_env", "OPENAI_API_KEY")
            if isinstance(api_key_env, str) and api_key_env.strip():
                cleaned["api_key_env"] = api_key_env.strip()
            else:
                cleaned["api_key_env"] = "OPENAI_API_KEY"

            presets[name.strip()] = cleaned

        return presets

    def _maybe_inject_agents_md(self) -> None:
        """Inject AGENTS.md into context once per content hash."""
        config_path = self._ctx.working_dir / "AGENTS.md"
        if not config_path.exists():
            return
        # Ensure system prompt is present before injecting AGENTS.md
        if not self._agent._context and self._agent.system_prompt:
            self._agent._context.add_message(
                SystemMessage(content=self._agent.system_prompt)
            )
        content = config_path.read_text(encoding="utf-8").strip()
        if not content:
            return
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if self._agents_md_hash == content_hash and self._agents_md_content:
            # If context was cleared, re-inject even if content unchanged
            for msg in self._agent._context.get_messages():
                if msg.role == "developer" and getattr(msg, "content", "") == self._agents_md_content:
                    return
        # Remove previously injected AGENTS.md content (if any)
        if self._agents_md_content:
            messages = self._agent._context.get_messages()
            for i in range(len(messages) - 1, -1, -1):
                msg = messages[i]
                if msg.role == "developer" and getattr(msg, "content", "") == self._agents_md_content:
                    self._agent._context.remove_message_at(i)
        self._agents_md_hash = content_hash
        self._agents_md_content = content
        self._agent._context.inject_message(UserMessage(content=content), pinned=True)

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
        self._console.print(f"Current model: [cyan]{model}[/cyan]")
        self._console.print(f"Base URL: [dim]{base_url_display}[/dim]")
        self._console.print(
            f"Context messages: [dim]{len(self._agent.messages)}[/dim]"
        )

    def _print_model_presets(self):
        """Print configured model presets."""
        if not self._model_presets:
            self._console.print(
                f"[yellow]No model presets found at {self._model_presets_path}[/yellow]"
            )
            return

        self._console.print("[bold cyan]Model presets:[/bold cyan]")
        for name, preset in self._model_presets.items():
            model = preset["model"]
            base_url = preset.get("base_url", "(inherit current)")
            api_key_env = preset.get("api_key_env", "OPENAI_API_KEY")
            marker = (
                " [green](default)[/green]"
                if name == self._default_model_preset
                else ""
            )
            self._console.print(
                f"  [cyan]{name}[/cyan]{marker} -> {model} "
                f"[dim](base_url: {base_url}, key: {api_key_env})[/dim]"
            )

    def _resolve_current_preset_name(self) -> str | None:
        """Best-effort preset match for current model/base URL."""
        exact_match = self._resolve_exact_current_preset_name()
        if exact_match is not None:
            return exact_match

        if self._default_model_preset in self._model_presets:
            return self._default_model_preset

        return next(iter(self._model_presets.keys()), None)

    def _resolve_exact_current_preset_name(self) -> str | None:
        """Exact preset match for current model/base URL, without fallback."""
        if not self._model_presets:
            return None

        current_model = str(self._agent.llm.model)
        current_base_url = getattr(self._agent.llm, "base_url", None)
        current_base_url_str = str(current_base_url) if current_base_url else None

        for name, preset in self._model_presets.items():
            if preset.get("model") != current_model:
                continue
            preset_base_url = preset.get("base_url")
            if preset_base_url is None or preset_base_url == current_base_url_str:
                return name

        return None

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
            model = preset["model"]
            markers: list[str] = []
            if name == current_preset:
                markers.append("current")
            if name == self._default_model_preset:
                markers.append("default")
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

    async def _switch_model_preset(self, preset_name: str) -> bool:
        """Switch to a configured model preset without clearing conversation context."""
        preset = self._model_presets.get(preset_name)
        if not preset:
            self._console.print(f"[red]Unknown preset: {preset_name}[/red]")
            self._console.print("[dim]Use /model list to see available presets.[/dim]")
            return False

        current_preset = self._resolve_exact_current_preset_name()
        if current_preset == preset_name:
            self._console.print(
                f"[dim]Already using preset [cyan]{preset_name}[/cyan].[/dim]"
            )
            return True

        model = preset["model"]
        api_key_env = preset.get("api_key_env", "OPENAI_API_KEY")
        api_key = os.getenv(api_key_env)
        if not api_key:
            self._console.print(
                f"[red]Missing API key env var: {api_key_env}. Switch aborted.[/red]"
            )
            return False

        preflight = await self._agent.preflight_model_switch(model)
        if not preflight.ok:
            self._console.print(
                f"[red]Model switch preflight failed: {preflight.reason or 'context is too large'}[/red]"
            )
            self._console.print(
                f"[dim]Estimated tokens: {preflight.estimated_tokens}, "
                f"threshold: {preflight.threshold}, "
                f"utilization: {preflight.threshold_utilization:.0%}[/dim]"
            )
            return False

        old_llm = self._agent.llm
        old_model = str(old_llm.model)
        old_base_url = getattr(old_llm, "base_url", None)
        new_base_url = preset.get("base_url")
        if new_base_url is None and old_base_url is not None:
            new_base_url = str(old_base_url)

        try:
            self._agent.llm = ChatOpenAI(
                model=model,
                api_key=api_key,
                base_url=new_base_url,
            )
        except Exception as e:
            self._agent.llm = old_llm
            self._console.print(f"[red]Failed to switch model: {e}[/red]")
            return False

        if preflight.compacted:
            self._console.print(
                "[yellow]Context was compacted before switching to fit target model.[/yellow]"
            )

        self._console.print(
            f"[green]Model switched:[/] [dim]{old_model}[/dim] -> [cyan]{model}[/cyan]"
        )
        self._console.print(
            f"[dim]Context preserved ({len(self._agent.messages)} messages).[/dim]"
        )
        return True

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
                f"Press [cyan]@[/cyan] to see available skills.\n"
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
  - The AI will use tools automatically to help you
"""
        self._console.print(Panel(help_text, border_style="dim"))

    def _print_slash_help(self):
        """Print slash command help information."""
        self._console.print()
        self._console.print("[bold cyan]Slash Commands:[/bold cyan]")
        self._console.print("[dim]Press / to see available commands, Tab to autocomplete[/dim]")
        self._console.print("[bold cyan]Skill Commands (@):[/bold cyan]")
        self._console.print("[dim]Use @<skill-name> to invoke skills, Tab to autocomplete[/dim]")
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
        command_name, args = parse_slash_command(text)

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

        # Unknown command
        self._console.print(f"[red]Unknown command: /{command_name}[/red]")
        self._console.print(f"[dim]Type /help for available commands.[/dim]")
        return True

    def _print_available_skills(self):
        """Print all available @ skills grouped by category."""
        self._console.print()
        self._console.print("[bold cyan]Available Skills (@):[/bold cyan]")
        self._console.print(
            "[dim]Use @<skill-name> to load a skill before your message[/dim]"
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

        await self._run_agent(expanded_message)
        return True

    async def _run_agent(self, user_input: str):
        """Run the agent with user input and display events."""
        self._step_number = 0
        self._console.print()

        # Inject AGENTS.md (if present) before each user query
        self._maybe_inject_agents_md()

        # Start loading animation
        self._loading = self._start_loading("Thinking")

        try:
            async for event in self._agent.query_stream(user_input):
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
                    self._console.print()
                    self._console.print()

        except Exception as e:
            self._stop_loading(self._loading)
            self._loading = None
            self._console.print(f"[{self.COLOR_ERROR}]Error: {e}[/]")

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
            await self._run_agent(user_input)
