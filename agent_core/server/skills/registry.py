"""
Skill registry for managing and resolving skills.

The registry supports multiple loaders with priority-based resolution.
"""

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .types import Skill
    from .loaders import SkillLoader

logger = logging.getLogger("agent_core.server.skills")


# Global registry instance
_global_registry: "SkillRegistry | None" = None


def get_global_registry() -> "SkillRegistry":
    """Get the global skill registry instance.

    Returns:
        The global SkillRegistry instance

    Raises:
        RuntimeError: If the registry has not been initialized
    """
    if _global_registry is None:
        raise RuntimeError("SkillRegistry not initialized. Call set_global_registry() first.")
    return _global_registry


def set_global_registry(registry: "SkillRegistry") -> None:
    """Set the global skill registry instance.

    Args:
        registry: The SkillRegistry instance to use globally
    """
    global _global_registry
    _global_registry = registry


class SkillRegistry:
    """Registry for managing skills with multiple loaders.

    The registry supports:
    - Multiple loaders with priority-based resolution
    - In-memory caching of loaded skills
    - Hot-reload of skills from all loaders
    - Category-based listing

    Example:
        registry = SkillRegistry()
        registry.register_loader(ConfigSkillLoader(builtin_skills))
        registry.register_loader(DatabaseSkillLoader(db_url))

        await registry.reload()

        skill = await registry.get("calculator")
    """

    def __init__(self):
        self._loaders: list["SkillLoader"] = []
        self._skills: dict[str, "Skill"] = {}
        self._lock = asyncio.Lock()

    def register_loader(self, loader: "SkillLoader") -> None:
        """Register a skill loader.

        Loaders are checked in the order they are registered.

        Args:
            loader: The SkillLoader instance to register
        """
        self._loaders.append(loader)
        logger.debug(f"Registered skill loader: {loader.__class__.__name__}")

    async def reload(self) -> None:
        """Reload all skills from all loaders.

        This clears the current cache and loads all skills again.
        """
        async with self._lock:
            self._skills.clear()

            for loader in self._loaders:
                try:
                    skills = await loader.load()
                    for skill in skills:
                        if skill.enabled:
                            self._skills[skill.name] = skill
                            logger.debug(f"Loaded skill: {skill.name}")
                except Exception as e:
                    logger.error(f"Error loading from {loader.__class__.__name__}: {e}")

            logger.info(f"Reloaded {len(self._skills)} skills from {len(self._loaders)} loaders")

    async def get(self, name: str) -> "Skill | None":
        """Get a skill by name.

        Checks cache first, then queries loaders in order if not found.

        Args:
            name: The skill name to retrieve

        Returns:
            The Skill if found, None otherwise
        """
        async with self._lock:
            # Check cache first
            if name in self._skills:
                return self._skills[name]

            # Query loaders in order
            for loader in self._loaders:
                try:
                    skill = await loader.get(name)
                    if skill and skill.enabled:
                        self._skills[name] = skill
                        return skill
                except Exception as e:
                    logger.warning(f"Error querying {loader.__class__.__name__} for '{name}': {e}")

            return None

    def list_all(self) -> list["Skill"]:
        """List all loaded skills.

        Returns:
            List of all Skill instances
        """
        return list(self._skills.values())

    def list_by_category(self) -> dict[str, list["Skill"]]:
        """List skills grouped by category.

        Returns:
            Dictionary mapping category names to lists of skills
        """
        categories: dict[str, list["Skill"]] = {}
        for skill in self._skills.values():
            if skill.category not in categories:
                categories[skill.category] = []
            categories[skill.category].append(skill)

        # Sort skills within each category
        for skills in categories.values():
            skills.sort(key=lambda s: s.name)

        return categories

    async def invalidate(self, name: str) -> None:
        """Remove a skill from cache.

        The skill will be reloaded from loaders on next access.

        Args:
            name: The skill name to invalidate
        """
        async with self._lock:
            self._skills.pop(name, None)
            logger.debug(f"Invalidated skill cache: {name}")

    async def add(self, skill: "Skill") -> None:
        """Directly add a skill to the registry.

        This bypasses loaders and adds the skill directly to cache.

        Args:
            skill: The Skill to add
        """
        async with self._lock:
            self._skills[skill.name] = skill
            logger.debug(f"Directly added skill: {skill.name}")

    def has(self, name: str) -> bool:
        """Check if a skill exists in the registry.

        Args:
            name: The skill name to check

        Returns:
            True if the skill exists, False otherwise
        """
        return name in self._skills

    def __len__(self) -> int:
        """Return the number of loaded skills."""
        return len(self._skills)

    def __contains__(self, name: str) -> bool:
        """Check if a skill name is in the registry."""
        return name in self._skills
