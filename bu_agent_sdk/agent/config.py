from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class AgentConfig:
    """Agent配置数据类"""
    name: str
    description: str
    mode: str
    model: Optional[str] = None
    temperature: Optional[float] = None
    tools: dict[str, bool] | None = None
    system_prompt: str = ""


def parse_agent_config(md_file: Path) -> Optional[AgentConfig]:
    content = md_file.read_text(encoding="utf-8")
    if not content.startswith("---"):
        return None

    parts = content.split("---", 2)
    if len(parts) < 3:
        return None

    metadata = yaml.safe_load(parts[1])
    if not metadata:
        return None

    mode = metadata.get("mode", "subagent")

    return AgentConfig(
        name=md_file.stem,
        description=metadata.get("description", ""),
        mode=mode,
        model=metadata.get("model"),
        temperature=metadata.get("temperature"),
        tools=metadata.get("tools"),
        system_prompt=parts[2].strip(),
    )
