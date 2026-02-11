"""@ command system for skill invocation.

Provides auto-discovery of skills, autocomplete for @ commands, and skill content loading.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path


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
    def from_file(cls, path: Path):
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
        frontmatter_match = re.match(r"^---\n(.*?)\n---\n", content, re.DOTALL)
        if not frontmatter_match:
            raise ValueError(f"No valid frontmatter found in {path}")

        frontmatter_text = frontmatter_match.group(1)

        try:
            import yaml
            metadata = yaml.safe_load(frontmatter_text) or {}
        except ImportError:
            # Fallback: simple parsing if yaml not available
            metadata = {}
            for line in frontmatter_text.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    metadata[key.strip()] = value.strip()

        return cls(
            name=metadata.get('name', path.parent.name),
            description=metadata.get('description', ''),
            path=path,
            category=metadata.get('category', 'General')
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

    def __init__(self):
        self.commands = {}  # type: dict[str, AtCommand]

    def discover_skills(self, skills_dir: Path) -> None:
        """Auto-discover skills from a directory.

        Looks for subdirectories containing skill.md files.

        Args:
            skills_dir: Path to the skills directory (e.g., bu_agent_sdk/skills)
        """
        if not skills_dir.exists():
            return

        for skill_path in skills_dir.glob("*/skill.md"):
            try:
                cmd = AtCommand.from_file(skill_path)
                self.commands[cmd.name] = cmd
            except (ValueError, FileNotFoundError, OSError):
                # Skip invalid skill files silently
                continue

    def get_command(self, name: str):
        """Get an AtCommand by name.

        Args:
            name: The skill name (without @ prefix)

        Returns:
            AtCommand instance, or None if not found
        """
        return self.commands.get(name)

    def list_commands(self):
        """List all commands grouped by category.

        Returns:
            Dictionary mapping category names to lists of AtCommand instances
        """
        categories = {}  # type: dict[str, list[AtCommand]]
        for cmd in self.commands.values():
            categories.setdefault(cmd.category, []).append(cmd)
        return categories
