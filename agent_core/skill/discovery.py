from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class DiscoveredSkill:
    name: str
    description: str
    path: Path
    category: str = "General"


def default_skill_dirs(workspace_root: Path, builtin_skills_dir: Path) -> list[Path]:
    return [
        builtin_skills_dir,
        Path("~/.tg_agent/skills").expanduser(),
        workspace_root / ".tg_agent" / "skills",
    ]


def discover_skill_files(skill_dirs: Sequence[Path]) -> list[DiscoveredSkill]:
    merged: dict[str, DiscoveredSkill] = {}
    for root_dir in skill_dirs:
        if not root_dir.exists() or not root_dir.is_dir():
            continue

        for skill_path in _iter_skill_paths(root_dir):
            try:
                skill = parse_discovered_skill(skill_path)
            except (ValueError, FileNotFoundError, OSError):
                continue
            merged[skill.name] = skill

    return sorted(merged.values(), key=lambda item: item.name)


def parse_discovered_skill(path: Path) -> DiscoveredSkill:
    content = path.read_text(encoding="utf-8")
    frontmatter_match = re.match(r"\A---\s*\r?\n(.*?)\r?\n---\s*(?:\r?\n|$)", content, re.DOTALL)
    if not frontmatter_match:
        raise ValueError(f"No valid frontmatter found in {path}")

    metadata = _parse_frontmatter(frontmatter_match.group(1))
    name = metadata.get("name", path.parent.name)
    if not name:
        raise ValueError(f"Skill name is required for {path}")

    return DiscoveredSkill(
        name=name,
        description=metadata.get("description", ""),
        path=path,
        category=metadata.get("category", "General"),
    )


def _parse_frontmatter(frontmatter_text: str) -> dict[str, str]:
    try:
        import yaml
    except ImportError:
        metadata: dict[str, str] = {}
        for line in frontmatter_text.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip()
        return metadata

    try:
        metadata = yaml.safe_load(frontmatter_text) or {}
    except Exception as exc:  # pragma: no cover - exact yaml error type is implementation-specific
        raise ValueError("Invalid skill frontmatter") from exc

    if not isinstance(metadata, dict):
        return {}
    return metadata


def _iter_skill_paths(skills_dir: Path) -> Iterable[Path]:
    discovered_paths: set[Path] = set()
    for pattern in ("*/skill.md", "*/SKILL.md"):
        discovered_paths.update(skills_dir.glob(pattern))
    yield from sorted(discovered_paths)
