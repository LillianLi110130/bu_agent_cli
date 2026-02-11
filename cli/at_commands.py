"""@ command system for skill invocation.

Provides auto-discovery of skills, autocomplete for @ commands, and skill content loading.
"""

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

    def load_content(self) -> str:
        """Read and return the full skill.md content.

        Returns:
            The complete markdown content of the skill.md file

        Raises:
            FileNotFoundError: If the skill.md file doesn't exist
            OSError: If the file cannot be read
        """
        return self.path.read_text(encoding="utf-8")
