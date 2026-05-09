"""End-to-end test for @ command functionality.

This test simulates the complete flow:
1. User types "@<skill-name> message"
2. Extract skill name
3. Get skill from registry
4. Load skill content
5. Prepend to user message
6. Send to LLM (simulated)
"""

from pathlib import Path
from cli.at_commands import (
    AtCommandRegistry,
    extract_at_command,
    prepend_skill_to_message,
)
import re


def test_calculator_at_command_flow():
    """Test complete @calculator command flow."""
    # Simulate user input
    user_input = "@calculator What is 123 + 456?"

    # Step 1: Extract @ command name
    skill_name = extract_at_command(user_input)
    assert skill_name == "calculator", f"Expected 'calculator', got '{skill_name}'"

    # Step 2: Get skill from registry
    registry = AtCommandRegistry()
    skills_dir = Path(__file__).resolve().parent.parent / "bu_agent_sdk" / "skills"
    registry.discover_skills(skills_dir)

    skill_cmd = registry.get_command(skill_name)
    assert skill_cmd is not None, f"Calculator skill not found in registry"
    assert skill_cmd.name == "calculator"

    # Step 3: Extract the actual user message (without @<skill>)
    user_message = re.sub(rf"@{re.escape(skill_name)}\s*", "", user_input, count=1)
    assert user_message == "What is 123 + 456?"

    # Step 4: Load skill content
    skill_content = skill_cmd.load_content()
    assert len(skill_content) > 0
    assert "calculator" in skill_content.lower()

    # Step 5: Prepend skill to message
    final_message = prepend_skill_to_message(skill_content, user_message)

    # Step 6: Verify final message
    # Should contain skill content
    assert skill_content in final_message
    # Should contain user message
    assert user_message in final_message
    # Skill should come before user message
    assert final_message.index(skill_content) < final_message.index(user_message)
    # Should have a separator
    assert "\n\n" in final_message

    print("✓ @calculator command flow works correctly")
    print(f"  - Original input: {user_input}")
    print(f"  - Skill name: {skill_name}")
    print(f"  - User message: {user_message}")
    print(f"  - Final message length: {len(final_message)} chars")
    print(f"  - Skill content length: {len(skill_content)} chars")


def test_python_at_command_flow():
    """Test complete @python command flow if it exists."""
    # Simulate user input
    user_input = "@python create a hello world function"

    # Step 1: Extract @ command name
    skill_name = extract_at_command(user_input)
    assert skill_name == "python"

    # Step 2: Get skill from registry
    registry = AtCommandRegistry()
    skills_dir = Path(__file__).resolve().parent.parent / "bu_agent_sdk" / "skills"
    registry.discover_skills(skills_dir)

    skill_cmd = registry.get_command(skill_name)
    if skill_cmd is None:
        # @python skill may not exist - that's okay
        print("Note: @python skill not found in registry")
        return

    # Step 3: Extract the actual user message
    user_message = re.sub(rf"@{re.escape(skill_name)}\s*", "", user_input, count=1)
    assert user_message == "create a hello world function"

    # Step 4: Load skill content
    skill_content = skill_cmd.load_content()
    assert len(skill_content) > 0

    # Step 5: Prepend skill to message
    final_message = prepend_skill_to_message(skill_content, user_message)

    # Step 6: Verify
    assert skill_content in final_message
    assert user_message in final_message

    print("✓ @python command flow works correctly")


def test_nonexistent_at_command():
    """Test handling of non-existent @ command."""
    user_input = "@nonexistent do something"

    skill_name = extract_at_command(user_input)
    assert skill_name == "nonexistent"

    registry = AtCommandRegistry()
    skills_dir = Path(__file__).resolve().parent.parent / "bu_agent_sdk" / "skills"
    registry.discover_skills(skills_dir)

    skill_cmd = registry.get_command(skill_name)
    assert skill_cmd is None

    print("✓ Non-existent @ command handled correctly (returns None)")


def test_all_skills_are_discoverable():
    """Test that all skills in bu_agent_sdk/skills can be discovered and loaded."""
    registry = AtCommandRegistry()
    skills_dir = Path(__file__).resolve().parent.parent / "bu_agent_sdk" / "skills"
    registry.discover_skills(skills_dir)

    all_skills = list(registry.commands.values())
    print(f"\n✓ Discovered {len(all_skills)} skills:")

    for skill in sorted(all_skills, key=lambda s: s.name):
        # Load content to verify it's accessible
        content = skill.load_content()
        assert len(content) > 0, f"Skill {skill.name} has no content"
        print(f"  - @{skill.name}: {skill.description[:60]}...")

    assert len(all_skills) > 0


if __name__ == "__main__":
    test_calculator_at_command_flow()
    print()
    test_python_at_command_flow()
    print()
    test_nonexistent_at_command()
    print()
    test_all_skills_are_discoverable()
    print("\n✅ All @ command e2e tests passed!")
