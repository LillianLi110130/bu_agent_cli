"""
Example: Using Skills in Server Environment

This example demonstrates how to initialize and use the skills system
in a server/online environment without file system access.

Skills are invoked by passing the skill name as a parameter to the chat API,
not as a tool call.
"""

import asyncio
from bu_agent_sdk import Agent
from bu_agent_sdk.llm import ChatOpenAI
from bu_agent_sdk.tools import tool
from bu_agent_sdk.server import create_server
from bu_agent_sdk.server.skills import (
    SkillRegistry,
    ConfigSkillLoader,
    BUILTIN_SKILLS,
    set_global_registry,
)


# 1. Define your tools
@tool("Add two numbers")
async def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


@tool("Multiply two numbers")
async def multiply(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b


# 2. Create and initialize the skill registry
async def init_skills():
    """Initialize the global skill registry."""
    registry = SkillRegistry()

    # Add built-in skills from config
    config_loader = ConfigSkillLoader(BUILTIN_SKILLS)
    registry.register_loader(config_loader)

    # Optionally: Add custom skills
    from bu_agent_sdk.server.skills.types import Skill

    custom_skills = [
        Skill(
            name="my_custom_skill",
            display_name="My Custom Skill",
            description="A custom skill for my specific use case",
            content="""
# My Custom Skill

This is a custom skill that provides specific instructions
for my application domain.

## Guidelines

- Always be polite
- Provide detailed explanations
- Include code examples when relevant
            """,
            category="Custom",
            source="config",
        )
    ]
    custom_loader = ConfigSkillLoader([s.to_dict() for s in custom_skills])
    registry.register_loader(custom_loader)

    # Load all skills
    await registry.reload()

    # Set as global registry
    set_global_registry(registry)

    # Log loaded skills
    by_category = registry.list_by_category()
    for category, skills in by_category.items():
        print(f"Category '{category}': {', '.join([s.name for s in skills])}")


# 3. Create agent factory (no need for use_skill tool)
def create_agent_factory():
    """Create an agent factory."""

    def factory():
        return Agent(
            llm=ChatOpenAI(model="gpt-4o"),
            tools=[add, multiply],  # Just your regular tools
        )

    return factory


# 4. Run the server
async def main():
    """Main entry point."""
    # Initialize skills first
    await init_skills()

    # Create server with agent factory
    agent_factory = create_agent_factory()
    app = create_server(
        agent_factory=agent_factory,
        session_timeout_minutes=30,
        max_sessions=100,
    )

    return app


# Minimal example for quick testing
def minimal_example():
    """
    Minimal example - just the essentials.
    """
    import uvicorn

    # Quick setup
    registry = SkillRegistry()
    registry.register_loader(ConfigSkillLoader(BUILTIN_SKILLS))

    # Create app
    from bu_agent_sdk import Agent
    from bu_agent_sdk.llm import ChatOpenAI
    from bu_agent_sdk.server import create_server

    def factory():
        return Agent(
            llm=ChatOpenAI(model="gpt-4o"),
            tools=[],
        )

    # Set registry before creating app
    set_global_registry(registry)

    # Create server
    app = create_server(agent_factory=factory)

    # Run
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    # Run the minimal example
    minimal_example()


# ========================================
# API Usage Examples
# ========================================

"""
# Example 1: Normal query without skill
POST /agent/query
{
    "message": "What is 2 + 2?",
    "session_id": "optional-session-id"
}

# Example 2: Query with calculator skill
POST /agent/query
{
    "message": "What is 123 + 456?",
    "skill": "calculator",
    "session_id": "optional-session-id"
}

# Example 3: Stream query with brainstorming skill
POST /agent/query-stream
{
    "message": "I want to add a user authentication feature",
    "skill": "brainstorming",
    "session_id": "optional-session-id"
}

# Example 4: List available skills
GET /skills

# Example 5: Get skills by category
GET /skills/by-category
"""
