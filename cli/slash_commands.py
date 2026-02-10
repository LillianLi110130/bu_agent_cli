"""Slash command system for Claude Code CLI.

Provides autocomplete for slash commands starting with '/', including:
- Automatic suggestion popup when '/' is pressed
- Tab completion for commands
- Command descriptions and metadata
"""

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable

from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document


# =============================================================================
# Command Metadata
# =============================================================================


@dataclass
class SlashCommand:
    """Metadata for a slash command."""

    name: str
    description: str
    usage: str = ""
    handler: Callable[[], str] | Callable[[str], str] | None = None
    is_builtin: bool = True
    category: str = "General"
    examples: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.usage:
            self.usage = f"/{self.name}"


# =============================================================================
# Command Registry
# =============================================================================


class SlashCommandRegistry:
    """Registry for slash commands with autocomplete support."""

    def __init__(self):
        self._commands: OrderedDict[str, SlashCommand] = OrderedDict()
        self._register_default_commands()

    def _register_default_commands(self):
        """Register default built-in commands."""
        default_commands = [
            SlashCommand(
                name="help",
                description="Show available commands and help information",
                usage="/help",
                examples=["/help"],
                category="General",
            ),
            SlashCommand(
                name="exit",
                description="Exit the CLI application",
                usage="/exit",
                examples=["/exit", "/quit"],
                category="General",
            ),
            SlashCommand(
                name="quit",
                description="Exit the CLI application (alias for /exit)",
                usage="/quit",
                examples=["/quit"],
                category="General",
            ),
            SlashCommand(
                name="pwd",
                description="Print current working directory",
                usage="/pwd",
                examples=["/pwd"],
                category="File System",
            ),
            SlashCommand(
                name="clear",
                description="Clear the terminal screen",
                usage="/clear",
                examples=["/clear"],
                category="General",
            ),
            SlashCommand(
                name="model",
                description="Show or change the current LLM model",
                usage="/model [model-name]",
                examples=["/model", "/model gpt-4o"],
                category="Settings",
            ),
            SlashCommand(
                name="history",
                description="Show command history",
                usage="/history",
                examples=["/history"],
                category="General",
            ),
            SlashCommand(
                name="reset",
                description="Reset the conversation context",
                usage="/reset",
                examples=["/reset"],
                category="Session",
            ),
            SlashCommand(
                name="allow",
                description="Add a directory to the sandbox allowed list",
                usage="/allow <path>",
                examples=["/allow /path/to/project", "/allow .."],
                category="File System",
            ),
            SlashCommand(
                name="allowed",
                description="List all allowed directories in the sandbox",
                usage="/allowed",
                examples=["/allowed"],
                category="File System",
            ),
        ]

        for cmd in default_commands:
            self.register(cmd)

    def register(self, command: SlashCommand) -> None:
        """Register a new slash command.

        Args:
            command: SlashCommand instance to register
        """
        self._commands[command.name] = command

    def unregister(self, name: str) -> None:
        """Unregister a slash command.

        Args:
            name: Command name to unregister
        """
        if name in self._commands:
            del self._commands[name]

    def get(self, name: str) -> SlashCommand | None:
        """Get a command by name.

        Args:
            name: Command name to retrieve

        Returns:
            SlashCommand if found, None otherwise
        """
        return self._commands.get(name)

    def get_all(self) -> list[SlashCommand]:
        """Get all registered commands.

        Returns:
            List of all SlashCommand instances
        """
        return list(self._commands.values())

    def get_by_category(self) -> dict[str, list[SlashCommand]]:
        """Get commands grouped by category.

        Returns:
            Dictionary mapping category names to lists of commands
        """
        categories: dict[str, list[SlashCommand]] = {}
        for cmd in self._commands.values():
            if cmd.category not in categories:
                categories[cmd.category] = []
            categories[cmd.category].append(cmd)
        return categories

    def match_prefix(self, prefix: str) -> list[SlashCommand]:
        """Get all commands matching a prefix.

        Args:
            prefix: Prefix string to match (without leading /)

        Returns:
            List of matching SlashCommand instances
        """
        prefix_lower = prefix.lower()
        return [
            cmd for cmd in self._commands.values()
            if cmd.name.lower().startswith(prefix_lower)
        ]


# =============================================================================
# Slash Command Completer
# =============================================================================


class SlashCommandCompleter(Completer):
    """Completer for slash commands.

    Provides autocompletion for commands starting with '/'.
    Shows command descriptions in the completion menu.
    """

    def __init__(self, registry: SlashCommandRegistry):
        self._registry = registry

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> iter:
        """Get completions for the current document.

        Args:
            document: The current Document being edited
            complete_event: The CompleteEvent that triggered this completion

        Yields:
            Completion objects for matching commands
        """
        text = document.text_before_cursor

        # Only trigger on '/' or when text starts with '/'
        if not text or not text[0] == "/":
            return

        # Extract the command part (without arguments)
        # Remove the leading '/' and get the first word
        command_part = text[1:].split()[0] if len(text) > 1 else ""

        # Find matching commands
        matching_commands = self._registry.match_prefix(command_part)

        # Create completions with rich display
        for cmd in matching_commands:
            # Calculate the text to be inserted
            # If the command is already fully typed, add a space
            if command_part == cmd.name:
                insert_text = cmd.name + " "
            else:
                # Insert the full command name to replace what user typed
                insert_text = cmd.name

            # Create display with description
            display = f"/{cmd.name}"
            display_meta = cmd.description

            # Find the position of '/' in the current text
            slash_pos = text.find("/")

            # Find where the command word starts (after '/' and any spaces)
            cmd_word_start = slash_pos + 1
            while cmd_word_start < len(text) and text[cmd_word_start] == " ":
                cmd_word_start += 1

            # Find the end of the command word (end of first word after spaces)
            cmd_word_end = cmd_word_start
            while cmd_word_end < len(text) and text[cmd_word_end] != " ":
                cmd_word_end += 1

            # Calculate start_position relative to cursor
            # start_position is negative: how many chars before cursor to start replacing
            cursor_pos = len(text)
            start_position = cmd_word_start - cursor_pos

            yield Completion(
                insert_text,
                start_position=start_position,
                display=display,
                display_meta=display_meta,
            )


# =============================================================================
# Utilities
# =============================================================================


def is_slash_command(text: str) -> bool:
    """Check if text is a slash command.

    Args:
        text: Text to check

    Returns:
        True if text starts with '/', False otherwise
    """
    return text.strip().startswith("/")


def parse_slash_command(text: str) -> tuple[str, list[str]]:
    """Parse a slash command into name and arguments.

    Args:
        text: Slash command text (e.g., "/model gpt-4o")

    Returns:
        Tuple of (command_name, arguments_list)
    """
    parts = text.strip()[1:].split()
    if not parts:
        return "", []

    command_name = parts[0]
    args = parts[1:] if len(parts) > 1 else []

    return command_name, args
