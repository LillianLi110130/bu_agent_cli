"""@ command system for skill invocation.

Provides auto-discovery of skills, autocomplete for @ commands, and skill
content expansion.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document


# =============================================================================
# AtCommand - Represents a single skill that can be invoked with @
# =============================================================================


@dataclass
class AtCommand:
    """Metadata and content loader for a skill invoked via @.

    Skills are stored in bu_agent_sdk/skills/*/skill.md files with YAML frontmatter.
    """

    name: str
    description: str
    path: Path
    category: str = "General"

    @classmethod
    def from_file(cls, path: Path) -> "AtCommand":
        """Create AtCommand by parsing a skill.md file.

        Extracts metadata from YAML frontmatter at the top of the file.

        Args:
            path: Path to the skill.md file

        Returns:
            AtCommand instance with parsed metadata

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file has no valid frontmatter
        """
        content = path.read_text(encoding="utf-8")

        # Parse YAML frontmatter (between --- markers)
        frontmatter_match = re.match(
            r"\A---\s*\r?\n(.*?)\r?\n---\s*(?:\r?\n|$)", content, re.DOTALL
        )
        if not frontmatter_match:
            raise ValueError(f"No valid frontmatter found in {path}")

        frontmatter_text = frontmatter_match.group(1)

        try:
            import yaml

            metadata = yaml.safe_load(frontmatter_text) or {}
        except ImportError:
            # Fallback: simple parsing if yaml not available
            metadata = {}
            for line in frontmatter_text.split("\n"):
                if ":" in line:
                    key, value = line.split(":", 1)
                    metadata[key.strip()] = value.strip()

        return cls(
            name=metadata.get("name", path.parent.name),
            description=metadata.get("description", ""),
            path=path,
            category=metadata.get("category", "General"),
        )

    def load_content(self) -> str:
        """Read and return the full skill.md content.

        Returns:
            The complete markdown content of the skill.md file

        Raises:
            FileNotFoundError: If the skill.md file doesn't exist
            OSError: If the file cannot be read
        """
        return self.path.read_text(encoding="utf-8")


# =============================================================================
# AtCommandRegistry - Auto-discovers and manages all available skills
# =============================================================================


class AtCommandRegistry:
    """Registry for auto-discovering and managing @ commands.

    Scans skills directory for skill.md files and caches them for quick lookup.
    """

    def __init__(self, skills_dir: Path | None = None):
        self.commands: dict[str, AtCommand] = {}
        self._skills_dir = skills_dir or (
            Path(__file__).resolve().parent.parent / "bu_agent_sdk" / "skills"
        )
        self.discover_skills(self._skills_dir)

    def discover_skills(self, skills_dir: Path | None = None) -> None:
        """Auto-discover skills from a directory.

        Looks for subdirectories containing skill.md files.

        Args:
            skills_dir: Path to the skills directory (e.g., bu_agent_sdk/skills)
        """
        if skills_dir is not None:
            self._skills_dir = skills_dir

        self.commands = {}
        if not self._skills_dir.exists():
            return

        for skill_path in sorted(self._skills_dir.glob("*/skill.md")):
            try:
                cmd = AtCommand.from_file(skill_path)
                self.commands[cmd.name] = cmd
            except (ValueError, FileNotFoundError, OSError):
                # Skip invalid skill files silently
                continue

    def get(self, name: str) -> AtCommand | None:
        """Get an AtCommand by name."""
        return self.commands.get(name)

    def get_command(self, name: str) -> AtCommand | None:
        """Get an AtCommand by name.

        Args:
            name: The skill name (without @ prefix)

        Returns:
            AtCommand instance, or None if not found
        """
        return self.get(name)

    def get_all(self) -> list[AtCommand]:
        """Get all discovered commands."""
        return list(self.commands.values())

    def get_by_category(self) -> dict[str, list[AtCommand]]:
        """Get all commands grouped by category."""
        return self.list_commands()

    def list_commands(self) -> dict[str, list[AtCommand]]:
        """List all commands grouped by category.

        Returns:
            Dictionary mapping category names to lists of AtCommand instances
        """
        categories: dict[str, list[AtCommand]] = {}
        for cmd in self.commands.values():
            categories.setdefault(cmd.category, []).append(cmd)
        for cmds in categories.values():
            cmds.sort(key=lambda cmd: cmd.name)
        return categories

    def match_prefix(self, prefix: str) -> list[AtCommand]:
        """Get commands matching a name prefix (case-insensitive)."""
        prefix_lower = prefix.lower()
        return [
            cmd
            for cmd in self.commands.values()
            if cmd.name.lower().startswith(prefix_lower)
        ]


# =============================================================================
# AtCommandCompleter - Provides Tab autocomplete for @ commands
# =============================================================================


class AtCommandCompleter(Completer):
    """Completer for @ commands.

    Provides autocompletion for commands starting with '@'.
    Shows command descriptions in the completion menu.
    """

    def __init__(self, registry: AtCommandRegistry):
        self._registry = registry

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> iter[Completion]:
        """Get completions for the current document.

        Args:
            document: The current Document being edited
            complete_event: The CompleteEvent that triggered this completion

        Yields:
            Completion objects for matching commands
        """
        text = document.text_before_cursor

        # Only trigger on '@' or when text starts with '@'
        if not text or text[0] != "@":
            return

        # Extract the command part (without arguments)
        # Remove the leading '@' and get the first word
        command_part = text[1:].split()[0] if len(text) > 1 else ""
        command_end = 1 + len(command_part)
        if document.cursor_position > command_end:
            return

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
            display = f"@{cmd.name}"
            display_meta = cmd.description

            completion_start_position = 1

            yield Completion(
                text=insert_text,
                start_position=completion_start_position - document.cursor_position,
                display=display,
                display_meta=display_meta,
            )


# =============================================================================
# Utility functions for @ command handling
# =============================================================================


def extract_at_command(message: str):
    """Extract the @ command name from a user message.

    Finds the first @ command in the message and returns its name.
    The @ command is identified as '@' followed by a word (alphanumeric + underscore).

    Args:
        message: The user message text

    Returns:
        The command name (without @ prefix), or None if no @ command found
    """
    # Match @ followed by common skill-name characters.
    match = re.search(r"@([A-Za-z0-9][A-Za-z0-9_-]*)", message)
    return match.group(1) if match else None


def is_at_command(text: str) -> bool:
    """Return True when input is a valid top-level @ command."""
    skill_name, _ = parse_at_command(text)
    return bool(skill_name)


def parse_at_command(text: str) -> tuple[str, str]:
    """Parse text into (<skill_name>, <message_without_skill_prefix>)."""
    stripped = text.strip()
    match = re.match(r"^@([A-Za-z0-9][A-Za-z0-9_-]*)\s*(.*)$", stripped, re.DOTALL)
    if not match:
        return "", ""
    return match.group(1), match.group(2)


def load_skill_content(path: Path) -> str:
    """Load the content of a skill from its file path.

    Args:
        path: Path to the skill.md file

    Returns:
        The skill content as a string

    Raises:
        FileNotFoundError: If the skill.md file doesn't exist
        OSError: If the file cannot be read
    """
    return path.read_text(encoding="utf-8")


def expand_at_command(skill: AtCommand, user_message: str) -> str:
    """Expand @ command by prepending skill content to user input."""
    return prepend_skill_to_message(skill.load_content(), user_message)


def prepend_skill_to_message(skill_content: str, user_message: str) -> str:
    """Prepend skill content to a user message.

    This is used to provide context from skills before the user's actual message.

    Args:
        skill_content: The markdown content from the skill.md file
        user_message: The original user message

    Returns:
        A string with skill content prepended to the user message
    """
    if not skill_content:
        return user_message

    # Add separator between skill and message for clarity
    skill_body = skill_content.rstrip()
    if not user_message:
        return skill_body
    return f"{skill_body}\n\n---\n\n{user_message}"
