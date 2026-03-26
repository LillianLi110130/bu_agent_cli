"""
Server-side Skills module for agent_core.

This module provides a skill system for online/server environments where
file system access is not available. Skills are loaded from configuration
or database and can be invoked to enhance agent capabilities.

Example:
    from agent_core.server.skills import SkillRegistry, ConfigSkillLoader

    # Create registry with built-in skills
    registry = SkillRegistry()
    loader = ConfigSkillLoader(builtin_skills)
    registry.register_loader(loader)

    # Get a skill
    skill = await registry.get("calculator")
    enhanced_prompt = skill.format(user_input="2 + 2")

    # Or use the injector for API handling
    from agent_core.server.skills import prepare_message_with_skill
    enhanced_msg, skill_name = await prepare_message_with_skill("2+2", "calculator")
"""

from .types import Skill, SkillSource
from .registry import SkillRegistry, get_global_registry, set_global_registry
from .loaders import (
    SkillLoader,
    ConfigSkillLoader,
    DatabaseSkillLoader,
    RemoteAPISkillLoader,
)
from .injector import prepare_message_with_skill, get_available_skills
from .builtin import BUILTIN_SKILLS

__all__ = [
    # Types
    "Skill",
    "SkillSource",
    # Registry
    "SkillRegistry",
    "get_global_registry",
    "set_global_registry",
    # Loaders
    "SkillLoader",
    "ConfigSkillLoader",
    "DatabaseSkillLoader",
    "RemoteAPISkillLoader",
    # Injector (for server API usage)
    "prepare_message_with_skill",
    "get_available_skills",
    # Built-in
    "BUILTIN_SKILLS",
]
