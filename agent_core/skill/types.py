from dataclasses import dataclass


@dataclass
class Skill:
    """Represents a mini-OpenCode skill.

    Attributes:
        name: The name of the skill.
        description: A description of what the skill does.
        path: The absolute path to the skill directory.
    """

    name: str
    description: str
    path: str
