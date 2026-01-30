"""
Claude Code CLI - An interactive coding assistant with file operations.

Includes bash, file operations (read/write/edit), search (glob/grep),
todo management, and task completion - all with dependency injection
for secure filesystem access.

Usage:
    py -3.10 claude_code.py
    py -3.10 claude_code.py --model gpt-4o
    py -3.10 claude_code.py --root-dir ./other-project

Environment Variables:
    LLM_MODEL: Model to use (default: GLM-4.7)
    LLM_BASE_URL: LLM API base URL (default: https://open.bigmodel.cn/api/coding/paas/v4)
    OPENAI_API_KEY: API key for OpenAI-compatible APIs
"""

import asyncio
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

from bu_agent_sdk import Agent
from bu_agent_sdk.agent import (
    FinalResponseEvent,
    TextEvent,
    ThinkingEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from bu_agent_sdk.llm import ChatOpenAI
from bu_agent_sdk.tools import Depends, tool

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax


# =============================================================================
# Sandbox Context - Filesystem management with security
# =============================================================================


class SecurityError(Exception):
    """Raised when a path escapes the sandbox."""

    pass


@dataclass
class SandboxContext:
    """Sandboxed filesystem context. All file operations are restricted to root_dir."""

    root_dir: Path  # 沙盒的根目录（边界）
    working_dir: Path  # 当前工作目录（在root_dir内）
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    @classmethod
    def create(cls, root_dir: Path | str | None = None) -> "SandboxContext":
        """Create a new sandbox context, defaulting to current directory."""
        session_id = str(uuid.uuid4())[:8]
        if root_dir is None:
            # 默认使用当前工作目录作为沙箱根目录
            root = Path.cwd().resolve()
        else:
            root = Path(root_dir).resolve()
        if not root.exists():
            root.mkdir(parents=True, exist_ok=True)
        return cls(root_dir=root, working_dir=root, session_id=session_id)

    def resolve_path(self, path: str | Path) -> Path:
        """Resolve and validate a path is within the sandbox."""
        path_obj = Path(path)
        if path_obj.is_absolute():
            resolved = path_obj.resolve()
        else:
            resolved = (self.working_dir / path_obj).resolve()

        # Security check: ensure path is within sandbox
        # Windows 兼容：使用绝对路径字符串比较，处理大小写不一致问题
        resolved_str = str(resolved).lower()
        root_str = str(self.root_dir.resolve()).lower()

        # 确保路径在沙箱内，且不是沙箱的兄弟目录（如 D:\project 匹配 D:\project-other）
        if not resolved_str.startswith(root_str):
            raise SecurityError(
                f"Path escapes sandbox: {path} -> {resolved} (root: {self.root_dir})"
            )
        # 检查是否恰好是 root 或其子目录
        if resolved_str != root_str and not resolved_str[len(root_str) :].startswith(("\\", "/")):
            raise SecurityError(
                f"Path escapes sandbox: {path} -> {resolved} (root: {self.root_dir})"
            )
        return resolved


def get_sandbox_context() -> SandboxContext:
    """Dependency injection marker. Override this in the agent."""
    raise RuntimeError("get_sandbox_context() must be overridden via dependency_overrides")


# =============================================================================
# Bash Tool
# =============================================================================


@tool("Execute a shell command and return output")
async def bash(
    command: str,
    ctx: Annotated[SandboxContext, Depends(get_sandbox_context)],
    timeout: int = 30,
) -> str:
    """Run a bash command in the sandbox working directory."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(ctx.working_dir),
        )
        output = result.stdout + result.stderr
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout}s"
    except Exception as e:
        return f"Error: {e}"


# =============================================================================
# File Operations
# =============================================================================


@tool("Read contents of a file")
async def read(
    file_path: str,
    ctx: Annotated[SandboxContext, Depends(get_sandbox_context)],
) -> str:
    """Read a file and return its contents with line numbers."""
    try:
        path = ctx.resolve_path(file_path)
    except SecurityError as e:
        return f"Security error: {e}"

    if not path.exists():
        return f"File not found: {file_path}"
    if path.is_dir():
        return f"Path is a directory: {file_path}"
    try:
        lines = path.read_text().splitlines()
        numbered = [f"{i + 1:4d}  {line}" for i, line in enumerate(lines)]
        return "\n".join(numbered)
    except Exception as e:
        return f"Error reading file: {e}"


@tool("Write content to a file")
async def write(
    file_path: str,
    content: str,
    ctx: Annotated[SandboxContext, Depends(get_sandbox_context)],
) -> str:
    """Write content to a file, creating directories if needed."""
    try:
        path = ctx.resolve_path(file_path)
    except SecurityError as e:
        return f"Security error: {e}"

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return f"Wrote {len(content)} bytes to {file_path}"
    except Exception as e:
        return f"Error writing file: {e}"


@tool("Replace text in a file")
async def edit(
    file_path: str,
    old_string: str,
    new_string: str,
    ctx: Annotated[SandboxContext, Depends(get_sandbox_context)],
) -> str:
    """Replace old_string with new_string in a file."""
    try:
        path = ctx.resolve_path(file_path)
    except SecurityError as e:
        return f"Security error: {e}"

    if not path.exists():
        return f"File not found: {file_path}"
    try:
        content = path.read_text()
        if old_string not in content:
            return f"String not found in {file_path}"
        count = content.count(old_string)
        new_content = content.replace(old_string, new_string)
        path.write_text(new_content)
        return f"Replaced {count} occurrence(s) in {file_path}"
    except Exception as e:
        return f"Error editing file: {e}"


# =============================================================================
# Search Tools
# =============================================================================


@tool("Find files matching a glob pattern")
async def glob_search(
    pattern: str,
    ctx: Annotated[SandboxContext, Depends(get_sandbox_context)],
    path: str | None = None,
) -> str:
    """Find files matching a glob pattern like **/*.py"""
    try:
        search_dir = ctx.resolve_path(path) if path else ctx.working_dir
    except SecurityError as e:
        return f"Security error: {e}"

    try:
        matches = list(search_dir.glob(pattern))
        files = [str(m.relative_to(ctx.root_dir)) for m in matches if m.is_file()][:50]
        if not files:
            return f"No files match pattern: {pattern}"
        return f"Found {len(files)} file(s):\n" + "\n".join(files)
    except Exception as e:
        return f"Error: {e}"


@tool("Search file contents with regex")
async def grep(
    pattern: str,
    ctx: Annotated[SandboxContext, Depends(get_sandbox_context)],
    path: str | None = None,
) -> str:
    """Search for pattern in files recursively."""
    try:
        search_dir = ctx.resolve_path(path) if path else ctx.working_dir
    except SecurityError as e:
        return f"Security error: {e}"

    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"Invalid regex: {e}"

    results = []
    for file_path in search_dir.rglob("*"):
        if not file_path.is_file():
            continue
        try:
            for i, line in enumerate(file_path.read_text().splitlines(), 1):
                if regex.search(line):
                    rel_path = file_path.relative_to(ctx.root_dir)
                    results.append(f"{rel_path}:{i}: {line[:100]}")
                    if len(results) >= 50:
                        return "\n".join(results) + "\n... (truncated)"
        except Exception:
            pass  # Skip binary/unreadable files
    return "\n".join(results) if results else f"No matches for: {pattern}"


# =============================================================================
# Todo Tools (session-scoped via context)
# =============================================================================

_todos: dict[str, list[dict]] = {}  # session_id -> todos


@tool("Read current todo list")
async def todo_read(
    ctx: Annotated[SandboxContext, Depends(get_sandbox_context)],
) -> str:
    """Get the current todo list."""
    todos = _todos.get(ctx.session_id, [])
    if not todos:
        return "Todo list is empty"
    lines = []
    for i, t in enumerate(todos, 1):
        status = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}[t["status"]]
        lines.append(f"{i}. {status} {t['content']}")
    return "\n".join(lines)


@tool("Update the todo list")
async def todo_write(
    todos: list[dict],
    ctx: Annotated[SandboxContext, Depends(get_sandbox_context)],
) -> str:
    """Set the todo list. Each item needs: content, status, activeForm"""
    _todos[ctx.session_id] = todos
    stats = {
        "pending": sum(1 for t in todos if t.get("status") == "pending"),
        "in_progress": sum(1 for t in todos if t.get("status") == "in_progress"),
        "completed": sum(1 for t in todos if t.get("status") == "completed"),
    }
    return f"Updated todos: {stats['pending']} pending, {stats['in_progress']} in progress, {stats['completed']} completed"


# =============================================================================
# Done Tool
# =============================================================================


@tool("Signal that the task is complete")
async def done(message: str) -> str:
    """Call this when the task is finished."""
    return f"TASK COMPLETE: {message}"


# =============================================================================
# All Tools
# =============================================================================

ALL_TOOLS = [bash, read, write, edit, glob_search, grep, todo_read, todo_write, done]


# =============================================================================
# CLI Application
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


class ClaudeCodeCLI:
    """Interactive CLI for Claude Code assistant."""

    # Color scheme
    COLOR_TOOL_CALL = "bright_blue"
    COLOR_TOOL_RESULT = "green"
    COLOR_ERROR = "red"
    COLOR_THINKING = "dim cyan"
    COLOR_FINAL = "bold green"

    def __init__(self, root_dir: Path | str | None = None, model: str | None = None):
        self._console = Console()
        self._ctx = SandboxContext.create(root_dir)
        self._agent: Agent | None = None
        self._model = model
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

    def _create_llm(self) -> ChatOpenAI:
        """Create LLM instance based on environment or model parameter."""
        model = self._model or os.getenv("LLM_MODEL", "GLM-4.7")
        base_url = os.getenv("LLM_BASE_URL", "https://open.bigmodel.cn/api/coding/paas/v4")
        api_key = os.getenv("OPENAI_API_KEY", "your_key")

        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
        )

    def _create_agent(self):
        """Create the agent with all tools."""
        llm = self._create_llm()
        self._agent = Agent(
            llm=llm,
            tools=ALL_TOOLS,
            system_prompt=(
                "You are a coding assistant. You can read, write, and edit files, "
                "run shell commands, search for files and content, and manage todos. "
                f"Working directory: {self._ctx.working_dir}"
            ),
            dependency_overrides={get_sandbox_context: lambda: self._ctx},
        )

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
        self._console.print(f"[dim]Model:[/] {self._agent.llm.model if self._agent else 'N/A'}")
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
        # Create agent
        self._create_agent()

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


# =============================================================================
# Main Entry Point
# =============================================================================


def parse_args():
    """Parse command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(description="Claude Code CLI - Interactive coding assistant")
    parser.add_argument(
        "--model",
        "-m",
        help="LLM model to use (default: from LLM_MODEL env var or gpt-4o)",
    )
    parser.add_argument(
        "--root-dir",
        "-r",
        help="Root directory for sandbox (default: current working directory)",
    )
    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_args()

    cli = ClaudeCodeCLI(root_dir=args.root_dir, model=args.model)
    try:
        await cli.run()
    except KeyboardInterrupt:
        print("\n[yellow]Goodbye![/yellow]")


if __name__ == "__main__":
    asyncio.run(main())
