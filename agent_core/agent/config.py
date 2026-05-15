from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any
import yaml


def _parse_tool_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        parsed = [str(item).strip() for item in value if str(item).strip()]
        return parsed or None
    return None


@dataclass
class AgentConfig:
    """Agent配置数据类"""
    name: str
    description: str
    source_path: Path | None = None
    source_scope: str = "builtin"
    source_priority: int = 100
    model: Optional[str] = None
    temperature: Optional[float] = None
    tools: list[str] | None = None
    disallowed_tools: list[str] = field(default_factory=list)
    max_turns: int | None = None
    background: bool = False
    skills: list[str] = field(default_factory=list)
    system_prompt: str = ""


def parse_agent_config(
    md_file: Path,
    *,
    source_scope: str = "builtin",
    source_priority: int = 100,
) -> Optional[AgentConfig]:
    content = md_file.read_text(encoding="utf-8")
    if not content.startswith("---"):
        return None

    parts = content.split("---", 2)
    if len(parts) < 3:
        return None

    metadata = yaml.safe_load(parts[1])
    if not metadata:
        return None

    return AgentConfig(
        name=metadata.get("name", md_file.stem),
        description=metadata.get("description", ""),
        source_path=md_file,
        source_scope=source_scope,
        source_priority=source_priority,
        model=metadata.get("model"),
        temperature=metadata.get("temperature"),
        tools=_parse_tool_list(metadata.get("tools")),
        disallowed_tools=_parse_tool_list(
            metadata.get("disallowedTools", metadata.get("disallowed_tools"))
        )
        or [],
        max_turns=metadata.get("maxTurns", metadata.get("max_turns")),
        background=bool(metadata.get("background", False)),
        skills=[str(item).strip() for item in metadata.get("skills", []) if str(item).strip()],
        system_prompt=parts[2].strip(),
    )
