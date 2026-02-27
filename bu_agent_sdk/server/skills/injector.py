"""
Skill injection utilities for server-side skill handling.

This module provides utilities to inject skill content into user messages
before sending to the agent, without using tool calls.
"""

import logging

logger = logging.getLogger("bu_agent_sdk.server.skills.injector")


async def prepare_message_with_skill(
    message: str,
    skill_name: str | None,
) -> tuple[str, str | None]:
    """Prepare a message with optional skill content prepended.

    Args:
        message: The original user message
        skill_name: Optional skill name to invoke

    Returns:
        A tuple of (processed_message, applied_skill_name)
        - processed_message: The message with skill content prepended (if skill found)
        - applied_skill_name: The name of the applied skill, or None if no skill

    Example:
        message, skill = await prepare_message_with_skill("2+2", "calculator")
        # message will have calculator skill content prepended
        # skill will be "calculator"
    """
    if not skill_name:
        return message, None

    try:
        from .registry import get_global_registry

        registry = get_global_registry()
    except RuntimeError:
        logger.warning("Skill registry not initialized, proceeding without skill")
        return message, None

    skill = await registry.get(skill_name)

    if not skill:
        logger.warning(f"Skill '{skill_name}' not found, proceeding without skill")
        return message, None

    if not skill.enabled:
        logger.warning(f"Skill '{skill_name}' is disabled, proceeding without skill")
        return message, None

    # Format the message with skill content
    enhanced = skill.format(message)
    logger.info(f"Applied skill '{skill_name}' to message")
    logger.debug(f"Enhanced message preview (first 200 chars): {enhanced[:200]}...")

    return enhanced, skill.name


def get_available_skills() -> list[dict]:
    """Get list of all available skills for API responses.

    Returns:
        List of skill dictionaries with basic info
    """
    try:
        from .registry import get_global_registry

        registry = get_global_registry()
        skills = registry.list_all()
        return [
            {
                "name": s.name,
                "display_name": s.display_name,
                "description": s.description,
                "category": s.category,
            }
            for s in skills
        ]
    except RuntimeError:
        return []
