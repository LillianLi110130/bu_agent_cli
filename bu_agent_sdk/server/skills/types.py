"""
Type definitions for the skills system.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class Skill:
    """Represents a skill that can be invoked to enhance agent capabilities.

    Attributes:
        name: Unique identifier for the skill (e.g., "calculator", "code_reviewer")
        display_name: Human-readable display name
        description: Brief description of what the skill does
        content: The skill content in markdown format (prompt/instructions)
        category: Category for organizing skills (e.g., "Math", "Development")
        source: Where the skill was loaded from (config, database, or API)
        enabled: Whether the skill is currently enabled
        version: Version string for the skill
        tags: Optional list of tags for filtering
    """

    name: str
    display_name: str
    description: str
    content: str
    category: str = "General"
    source: Literal["config", "database", "api"] = "config"
    enabled: bool = True
    version: str = "1.0"
    tags: list[str] | None = None

    def format(self, user_input: str = "") -> str:
        """Format the skill content with optional user input.

        Args:
            user_input: The user's input/request to process

        Returns:
            The formatted prompt with skill content prepended
        """
        if not user_input:
            return self.content

        return f"""# Skill: {self.display_name}

{self.content}

---

## User Request
{user_input}
"""

    def to_dict(self) -> dict:
        """Convert skill to dictionary for serialization."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "content": self.content,
            "category": self.category,
            "source": self.source,
            "enabled": self.enabled,
            "version": self.version,
            "tags": self.tags or [],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Skill":
        """Create a Skill from a dictionary."""
        return cls(
            name=data["name"],
            display_name=data["display_name"],
            description=data["description"],
            content=data["content"],
            category=data.get("category", "General"),
            source=data.get("source", "config"),
            enabled=data.get("enabled", True),
            version=data.get("version", "1.0"),
            tags=data.get("tags"),
        )


# Type alias for source
SkillSource = Literal["config", "database", "api"]
