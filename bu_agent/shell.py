"""
Shell UI implementation for BU Agent CLI.
"""

import asyncio
import sys
import threading
import time
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from rich.live import Live
from rich.spinner import Spinner

from bu_agent.core import BUAgent


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
        # Use ANSI escape sequence to clear entire line
        sys.stdout.write("\r\033[K")  # Carriage return + clear to end of line
        sys.stdout.flush()

    def start(self):
        """Start the loading animation in a separate thread."""
        self._stop_event.clear()

        # Show first frame immediately
        self._show_frame(0)

        def _run():
            frame = 1
            while not self._stop_event.is_set():
                self._show_frame(frame % len(self._frames))
                frame += 1
                time.sleep(0.08)
            # Clear the line when stopping
            self._clear()

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the loading animation."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=0.5)
        # Ensure cleared (do it twice to be sure)
        self._clear()
        self._clear()


# Colors
_COLOR_TOOL_CALL = "bright_blue"
_COLOR_TOOL_RESULT = "green"
_COLOR_ERROR = "red"
_COLOR_THINKING = "dim cyan"
_COLOR_FINAL = "bold green"


class Shell:
    """
    A simple interactive shell for BU Agent.
    """

    def __init__(self, agent: BUAgent, welcome_info: list[dict] | None = None):
        self.agent = agent
        self._welcome_info = list(welcome_info or [])
        self._console = Console()
        self._step_number = 0
        self._current_tool_output: list[str] = []
        self._loading: _LoadingIndicator | None = None

    async def run(self, command: str | None = None) -> bool:
        """Run the shell."""

        _print_welcome(self._console, self.agent.name, self._welcome_info)

        # Create key bindings
        kb = KeyBindings()
        kb.add("c-d")(self._exit_handler)

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

            if not user_input.strip():
                continue

            if user_input.strip().lower() in ["exit", "quit"]:
                self._console.print("[yellow]Goodbye![/yellow]")
                break

            await self._run_agent(user_input)

        return True

    def _exit_handler(self, event):
        """Handle Ctrl+D to exit."""
        event.app.exit(exception=EOFError)

    def _start_loading(self, message: str = "Thinking") -> _LoadingIndicator:
        """Start a loading animation."""
        loading = _LoadingIndicator(message)
        loading.start()
        # Small delay to ensure the first frame is rendered
        time.sleep(0.02)
        return loading

    def _stop_loading(self, loading: _LoadingIndicator | None):
        """Stop the loading animation."""
        if loading:
            loading.stop()
            # Print newline to move past the cleared line
            sys.stdout.write("\n")
            sys.stdout.flush()

    async def _run_agent(self, user_input: str):
        """Run the agent with user input."""
        self._step_number = 0
        self._console.print()

        # Start loading animation
        self._loading = self._start_loading("Thinking")

        def on_event(event_type: str, data: Any):
            """Handle agent events."""
            if event_type == "step_start":
                # Step is about to start, restart loading with Executing message
                self._stop_loading(self._loading)
                self._loading = self._start_loading("Executing")

            elif event_type == "step_complete":
                # Step completed, loading will continue for next LLM call
                pass

            elif event_type == "text":
                self._stop_loading(self._loading)
                self._loading = None
                self._console.print(data, end="")

            elif event_type == "thinking":
                self._stop_loading(self._loading)
                self._loading = None
                self._console.print(f"[{_COLOR_THINKING}]Thinking: {data}[/]")

            elif event_type == "tool_call":
                # 🔴 兜底清除 Executing...
                self._stop_loading(self._loading)
                self._loading = None

                self._step_number += 1
                tool_data = data
                self._console.print()
                self._console.print(
                    f"[{_COLOR_TOOL_CALL}]Step {self._step_number}:[/] "
                    f"[bold]{tool_data['name']}[/]"
                )
                self._console.print(f"  [dim]Args: {tool_data['arguments'][:100]}...[/]")
                self._current_tool_output = []

            elif event_type == "tool_result":
                result_data = data
                result = result_data["result"]
                is_error = result_data["is_error"]

                if is_error:
                    self._console.print(f"  [{_COLOR_ERROR}]Error: {result[:200]}[/]")
                else:
                    # Show truncated result
                    if len(result) > 300:
                        self._console.print(f"  [{_COLOR_TOOL_RESULT}]Result: {result[:300]}...[/]")
                    else:
                        self._console.print(f"  [{_COLOR_TOOL_RESULT}]Result: {result}[/]")
                # Restart loading after tool result (for next LLM call)
                self._stop_loading(self._loading)
                self._loading = self._start_loading("Thinking")

            elif event_type == "final":
                self._stop_loading(self._loading)
                self._loading = None
                self._console.print()

            elif event_type == "error":
                self._stop_loading(self._loading)
                self._loading = None
                self._console.print(f"[{_COLOR_ERROR}]Error: {data}[/]")

        try:
            await self.agent.run(user_input, on_event)
        except Exception as e:
            # Stop loading if still running
            self._stop_loading(self._loading)
            self._loading = None
            self._console.print(f"[{_COLOR_ERROR}]Error: {e}[/]")

        # Ensure loading is stopped
        self._stop_loading(self._loading)
        self._loading = None
        self._console.print()


def _print_welcome(console: Console, name: str, info_items: list[dict]):
    """Print welcome message."""

    logo = """
    ╔═══════════════════╗
    ║   BU Agent CLI   ║
    ╚═══════════════════╝
    """

    console.print(
        Panel(
            f"[bold cyan]{name}[/bold cyan]\n\n"
            f"Type your message and press Enter to send.\n"
            f"Press Ctrl+D or type 'exit' to quit.\n",
            title="[bold blue]Welcome[/bold blue]",
            border_style="bright_blue",
        )
    )

    if info_items:
        console.print()
        for item in info_items:
            level = item.get("level", "info")
            color = {
                "info": "dim",
                "warn": "yellow",
                "error": "red",
            }.get(level, "dim")
            console.print(f"[{color}]{item['name']}: {item['value']}[/]")
