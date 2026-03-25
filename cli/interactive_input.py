"""
Interactive Input Helper

Provides utilities for interactive user input in the CLI.
"""

import os
import sys
from pathlib import Path
from typing import Any, Awaitable, Callable, List, Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.shortcuts import confirm
from rich.console import Console

from tools.text_encoding import read_text_with_fallback


class InteractivePrompter:
    """Interactive input helper for CLI."""

    def __init__(self, console: Optional[Console] = None):
        """Initialize the prompter.

        Args:
            console: Rich console instance
        """
        self.console = console or Console()
        self._session = PromptSession()

    async def prompt_text(
        self,
        prompt: str,
        default: str = "",
        optional: bool = False,
        placeholder: str = "",
    ) -> str:
        """Prompt for text input.

        Args:
            prompt: Prompt message
            default: Default value
            optional: Whether input is optional
            placeholder: Placeholder text

        Returns:
            User input
        """
        if default:
            prompt_display = f"{prompt} [{default}]: "
        elif optional:
            prompt_display = f"{prompt}（可选）: "
        else:
            prompt_display = f"{prompt}: "

        try:
            result = await self._session.prompt_async(
                HTML(f"<ansiblue>{prompt_display}</ansiblue>"),
                default=default,
            )

            if not result and optional:
                return default

            return result
        except (EOFError, KeyboardInterrupt):
            return ""

    async def prompt_choice(
        self,
        prompt: str,
        choices: List[str],
        default: Optional[str] = None,
    ) -> str:
        """Prompt for choice selection.

        Args:
            prompt: Prompt message
            choices: Available choices
            default: Default choice

        Returns:
            Selected choice
        """
        if not choices:
            raise ValueError("choices cannot be empty")

        # Show choices
        self.console.print(f"\n{prompt}")

        completer = WordCompleter(choices)

        for i, choice in enumerate(choices, 1):
            marker = "（默认）" if choice == default else ""
            self.console.print(f"  {i}. {choice}{marker}")

        self.console.print()

        while True:
            try:
                # Allow typing choice directly
                result = await self._session.prompt_async(
                    HTML("<ansiblue>请选择：</ansiblue> "),
                    completer=completer,
                )

                # Check if input is a number
                if result.isdigit():
                    idx = int(result) - 1
                    if 0 <= idx < len(choices):
                        return choices[idx]

                # Check if input matches a choice
                result_lower = result.lower()
                for choice in choices:
                    if choice.lower() == result_lower:
                        return choice

                self.console.print(
                    f"[red]无效选项。请输入 1-{len(choices)}，或直接输入选项名称。[/red]"
                )

            except (EOFError, KeyboardInterrupt):
                return default if default else choices[0]

    async def prompt_yes_no(
        self,
        prompt: str,
        default: bool = True,
    ) -> bool:
        """Prompt for yes/no confirmation.

        Args:
            prompt: Prompt message
            default: Default value

        Returns:
            True for yes, False for no
        """
        default_str = "回车默认是" if default else "回车默认否"
        prompt_display = f"{prompt} [{default_str}，输入 y/n 或 是/否]: "

        while True:
            try:
                result = await self._session.prompt_async(
                    HTML(f"<ansiblue>{prompt_display}</ansiblue>")
                )

                if not result:
                    return default

                result_lower = result.lower()
                if result_lower in ("y", "yes", "是"):
                    return True
                elif result_lower in ("n", "no", "否"):
                    return False

                self.console.print("[red]请输入 y/n 或 是/否。[/red]")

            except (EOFError, KeyboardInterrupt):
                return default

    async def prompt_multiselect(
        self,
        prompt: str,
        choices: List[str],
        default: Optional[List[str]] = None,
    ) -> List[str]:
        """Prompt for multiple selections.

        Args:
            prompt: Prompt message
            choices: Available choices
            default: Default selections

        Returns:
            List of selected choices
        """
        default = default or []

        self.console.print(f"\n{prompt}")
        self.console.print(
            "[dim]请输入多个选项编号或名称，用逗号分隔；输入 all 表示全选，输入 none 表示不选[/dim]"
        )

        for i, choice in enumerate(choices, 1):
            marker = "[green]+[/green]" if choice in default else "[dim]-[/dim]"
            self.console.print(f"  {marker} {i}. {choice}")

        self.console.print()

        try:
            result = await self._session.prompt_async(
                HTML("<ansiblue>已选：</ansiblue> "),
                completer=WordCompleter(choices + ["all", "none"]),
            )

            if not result:
                return default

            result_lower = result.lower().strip()

            if result_lower == "all":
                return choices.copy()
            if result_lower == "none":
                return []

            # Parse comma-separated selections
            selected = []
            for item in result.split(","):
                item = item.strip()

                # Check if it's a number
                if item.isdigit():
                    idx = int(item) - 1
                    if 0 <= idx < len(choices):
                        selected.append(choices[idx])
                else:
                    # Check if it matches a choice
                    for choice in choices:
                        if choice.lower() == item.lower():
                            selected.append(choice)
                            break

            return selected if selected else default.copy()

        except (EOFError, KeyboardInterrupt):
            return default.copy()

    async def prompt_multiline(
        self,
        prompt: str,
        default: str = "",
    ) -> str:
        """Prompt for multiline text input.

        Args:
            prompt: Prompt message
            default: Default value

        Returns:
            Multiline text
        """
        from prompt_toolkit.key_binding import KeyBindings

        kb = KeyBindings()

        @kb.add("c-q")
        def _(event):
            """Finish input on Ctrl-Q."""
            event.app.exit(result=event.app.current_buffer.text)

        self.console.print(f"\n{prompt}")
        self.console.print("[dim]请输入内容（按 Ctrl+Q，或按 Esc 后回车结束）：[/dim]\n")

        try:
            result = await self._session.prompt_async(
                HTML("<ansiblue>></ansiblue> "),
                multiline=True,
                key_bindings=kb,
                default=default,
            )

            return result

        except (EOFError, KeyboardInterrupt):
            return default

    async def prompt_file_content(
        self,
        prompt: str,
    ) -> Optional[str]:
        """Prompt for file path and return content.

        Args:
            prompt: Prompt message

        Returns:
            File content or None
        """
        self.console.print(f"\n{prompt}")
        self.console.print("[dim]请输入文件路径，留空可取消[/dim]")

        try:
            file_path = await self._session.prompt_async(
                HTML("<ansiblue>文件路径：</ansiblue> ")
            )

            if not file_path:
                return None

            path = Path(file_path.strip())

            if not path.exists():
                self.console.print(f"[red]未找到文件：{path}[/red]")
                return None

            content, _ = read_text_with_fallback(path)
            return content

        except (EOFError, KeyboardInterrupt):
            return None

    async def prompt_edit(
        self,
        prompt: str,
        current_value: str,
    ) -> str:
        """Prompt for editing a value.

        Args:
            prompt: Prompt message
            current_value: Current value to edit

        Returns:
            Edited value
        """
        self.console.print(f"\n{prompt}")
        self.console.print("[dim]当前值：[/dim]")
        self.console.print(current_value[:200] + "..." if len(current_value) > 200 else current_value)
        self.console.print()
        self.console.print("[dim]输入新值，或直接回车保留当前值[/dim]")

        try:
            result = await self._session.prompt_async(
                HTML("<ansiblue>新值：</ansiblue> "),
                default=current_value,
            )

            return result

        except (EOFError, KeyboardInterrupt):
            return current_value

    async def prompt_number(
        self,
        prompt: str,
        default: float,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> float:
        """Prompt for numeric input.

        Args:
            prompt: Prompt message
            default: Default value
            min_value: Minimum allowed value
            max_value: Maximum allowed value

        Returns:
            Numeric value
        """
        range_str = ""
        if min_value is not None and max_value is not None:
            range_str = f" [{min_value}-{max_value}]"
        elif min_value is not None:
            range_str = f" (>= {min_value})"
        elif max_value is not None:
            range_str = f" (<= {max_value})"

        prompt_display = f"{prompt}{range_str} [{default}]: "

        while True:
            try:
                result = await self._session.prompt_async(
                    HTML(f"<ansiblue>{prompt_display}</ansiblue>")
                )

                if not result:
                    return default

                try:
                    value = float(result)

                    if min_value is not None and value < min_value:
                        self.console.print(f"[red]数值必须 >= {min_value}[/red]")
                        continue
                    if max_value is not None and value > max_value:
                        self.console.print(f"[red]数值必须 <= {max_value}[/red]")
                        continue

                    return value

                except ValueError:
                    self.console.print("[red]请输入有效数字[/red]")

            except (EOFError, KeyboardInterrupt):
                return default


def get_editor_command() -> Optional[str]:
    """Get the editor command from environment.

    Returns:
        Editor command or None
    """
    # Check EDITOR variable
    editor = os.getenv("EDITOR")
    if editor:
        return editor

    # Platform-specific defaults
    if sys.platform == "win32":
        # Try code, then notepad
        for cmd in ["code", "notepad"]:
            try:
                import shutil
                if shutil.which(cmd):
                    return cmd
            except Exception:
                pass

    return None


async def open_in_editor(file_path: Path, editor: Optional[str] = None) -> bool:
    """Open a file in an external editor.

    Args:
        file_path: File to edit
        editor: Editor command (uses EDITOR env var if not specified)

    Returns:
        True if successful
    """
    if editor is None:
        editor = get_editor_command()

    if not editor:
        return False

    try:
        import subprocess
        process = await asyncio.create_subprocess_exec(
            editor,
            str(file_path),
        )
        await process.wait()
        return True

    except Exception:
        return False


# Import asyncio for open_in_editor
import asyncio
