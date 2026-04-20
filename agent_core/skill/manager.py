from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from agent_core.skill.discovery import (
    is_user_writable_skill_path,
    parse_discovered_skill,
    user_skills_dir,
)


class SkillManagementError(ValueError):
    """Raised when a skill management operation is invalid or unsafe."""


@dataclass(frozen=True, slots=True)
class SkillManageResult:
    action: str
    name: str
    path: Path
    message: str


class SkillManager:
    """Write manager for user-level skills under ``~/.tg_agent/skills`` only."""

    _VALID_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,80}$")
    _SUPPORT_DIRS = {"references", "templates", "scripts", "assets"}

    def __init__(self, skills_root: Path | None = None) -> None:
        self.skills_root = (skills_root or user_skills_dir()).expanduser().resolve()

    def create(self, *, name: str, content: str) -> SkillManageResult:
        skill_name = self._validate_name(name)
        try:
            existing = self._find_existing_skill_file(skill_name)
        except SkillManagementError:
            existing = None
        if existing is not None:
            raise SkillManagementError(f"Skill already exists: {skill_name}")

        target = self._skill_file_for_new_skill(skill_name)
        if target.exists():
            raise SkillManagementError(f"Skill already exists: {skill_name}")

        self._validate_target_path(target)
        target.parent.mkdir(parents=True, exist_ok=False)
        try:
            target.write_text(content, encoding="utf-8")
            self._validate_skill_file(target, expected_name=skill_name)
        except Exception:
            shutil.rmtree(target.parent, ignore_errors=True)
            raise

        return SkillManageResult(
            action="create",
            name=skill_name,
            path=target,
            message=f"Skill created: {skill_name}",
        )

    def patch(
        self,
        *,
        name: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> SkillManageResult:
        skill_name = self._validate_name(name)
        target = self._find_existing_skill_file(skill_name)
        original = target.read_text(encoding="utf-8")
        if old_string not in original:
            raise SkillManagementError("old_string not found in user-level skill")

        count = original.count(old_string)
        if count > 1 and not replace_all:
            raise SkillManagementError(
                "old_string appears multiple times; set replace_all=true or provide more context"
            )

        updated = original.replace(old_string, new_string, count if replace_all else 1)
        self._write_with_rollback(target, updated, expected_name=skill_name)
        return SkillManageResult(
            action="patch",
            name=skill_name,
            path=target,
            message=f"Skill patched: {skill_name}",
        )

    def edit(self, *, name: str, content: str) -> SkillManageResult:
        skill_name = self._validate_name(name)
        target = self._find_existing_skill_file(skill_name)
        self._write_with_rollback(target, content, expected_name=skill_name)
        return SkillManageResult(
            action="edit",
            name=skill_name,
            path=target,
            message=f"Skill edited: {skill_name}",
        )

    def write_file(self, *, name: str, file_path: str, content: str) -> SkillManageResult:
        skill_name = self._validate_name(name)
        skill_file = self._find_existing_skill_file(skill_name)
        target = self._resolve_support_file(skill_file.parent, file_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        original_exists = target.exists()
        original = target.read_text(encoding="utf-8") if original_exists else None
        try:
            target.write_text(content, encoding="utf-8")
        except Exception:
            if original_exists and original is not None:
                target.write_text(original, encoding="utf-8")
            elif target.exists():
                target.unlink()
            raise

        return SkillManageResult(
            action="write_file",
            name=skill_name,
            path=target,
            message=f"Skill support file written: {skill_name}",
        )

    def remove_file(self, *, name: str, file_path: str) -> SkillManageResult:
        skill_name = self._validate_name(name)
        skill_file = self._find_existing_skill_file(skill_name)
        target = self._resolve_support_file(skill_file.parent, file_path)
        if not target.exists():
            raise SkillManagementError(f"Support file not found: {file_path}")
        if target.is_dir():
            raise SkillManagementError("remove_file only supports files")
        target.unlink()
        return SkillManageResult(
            action="remove_file",
            name=skill_name,
            path=target,
            message=f"Skill support file removed: {skill_name}",
        )

    def _skill_file_for_new_skill(self, name: str) -> Path:
        return self.skills_root / name / "SKILL.md"

    def _find_existing_skill_file(self, name: str) -> Path:
        self.skills_root.mkdir(parents=True, exist_ok=True)
        for path in sorted(self.skills_root.glob("*/SKILL.md")) + sorted(
            self.skills_root.glob("*/skill.md")
        ):
            if not is_user_writable_skill_path(path):
                continue
            try:
                skill = parse_discovered_skill(path, root_dir=self.skills_root)
            except (OSError, ValueError):
                continue
            if skill.name == name:
                self._validate_target_path(path)
                return path
        raise SkillManagementError(f"User-level skill not found: {name}")

    def _validate_target_path(self, path: Path) -> None:
        if not is_user_writable_skill_path(path):
            raise SkillManagementError(
                f"Refusing to modify non-user skill path: {path.resolve()}"
            )

    def _validate_skill_file(self, path: Path, *, expected_name: str) -> None:
        self._validate_target_path(path)
        parsed = parse_discovered_skill(path, root_dir=self.skills_root)
        if parsed.name != expected_name:
            raise SkillManagementError(
                f"Skill frontmatter name must be '{expected_name}', got '{parsed.name}'"
            )

    def _write_with_rollback(self, path: Path, content: str, *, expected_name: str) -> None:
        self._validate_target_path(path)
        original = path.read_text(encoding="utf-8")
        try:
            path.write_text(content, encoding="utf-8")
            self._validate_skill_file(path, expected_name=expected_name)
        except Exception:
            path.write_text(original, encoding="utf-8")
            raise

    def _resolve_support_file(self, skill_dir: Path, relative_path: str) -> Path:
        rel = Path(relative_path)
        if rel.is_absolute():
            raise SkillManagementError("Support file path must be relative to the skill directory")
        if not rel.parts:
            raise SkillManagementError("Support file path is required")
        if rel.parts[0] not in self._SUPPORT_DIRS:
            allowed = ", ".join(sorted(self._SUPPORT_DIRS))
            raise SkillManagementError(f"Support files must be under one of: {allowed}")

        target = (skill_dir / rel).resolve()
        self._validate_target_path(target)
        try:
            target.relative_to(skill_dir.resolve())
        except ValueError as exc:
            raise SkillManagementError("Support file path escapes the skill directory") from exc
        if target.name.lower() == "skill.md":
            raise SkillManagementError("Use patch or edit for SKILL.md")
        return target

    def _validate_name(self, name: str) -> str:
        skill_name = (name or "").strip()
        if not self._VALID_NAME.match(skill_name):
            raise SkillManagementError(
                "Skill name must use letters, numbers, underscores or hyphens only"
            )
        return skill_name
