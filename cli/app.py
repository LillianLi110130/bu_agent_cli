"""CLI Application for Claude Code.

Contains the main ClaudeCodeCLI class and loading indicator.
Pure UI logic - receives pre-configured Agent and context.
"""

import sys
import threading
import time
from pathlib import Path

from bu_agent_sdk import Agent
from bu_agent_sdk.agent import (
    FinalResponseEvent,
    TextEvent,
    ThinkingEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.panel import Panel

from tools import SandboxContext


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
                f"Press Ctrl+D or type 'exit' to quit.\n"
                f"Type 'help' for available commands.\n",
                title="[bold blue]Welcome[/bold blue]",
                border_style="bright_blue",
            )
        )

        # Show sandbox info
        self._console.print()
        self._console.print(f"[dim]Working directory:[/] {self._ctx.working_dir}")
        self._console.print(f"[dim]Model:[/] {self._agent.llm.model}")
        self._console.print(f"[dim]Tools:[/] bash, read, write, edit, glob, grep, todos")
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

    async def _run_agent(self, user_input: str):
        """Run the agent with user input and display events."""
        self._step_number = 0
        self._console.print()

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

    async def run(self):
        """Run the interactive CLI."""
        from prompt_toolkit import PromptSession

        # Print welcome
        self._print_welcome()

        # Create key bindings
        kb = KeyBindings()

        @kb.add("c-d")
        def _exit(event):  # noqa: D401
            event.app.exit(exception=EOFError)

        # Mark as intentionally used
        _ = _exit

        # Create prompt session
        session = PromptSession(
            message=lambda: HTML("<ansiblue>>> </ansiblue>"),
            key_bindings=kb,
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

            # Handle built-in commands
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
