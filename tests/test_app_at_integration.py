"""Integration tests for @ command handling.

Tests the integration between AtCommandRegistry, AtCommandCompleter,
and verifies the workflow for @ commands without importing app.py
(which has Python 3.10+ dependencies).
"""
import pytest
from pathlib import Path
from unittest.mock import Mock, patch
import re

# Import classes we need to test
from cli.at_commands import (
    AtCommand,
    AtCommandRegistry,
    AtCommandCompleter,
    extract_at_command,
    prepend_skill_to_message,
)


class TestAtCommandWorkflow:
    """Test the complete @ command workflow."""

    def test_registry_discovers_real_skills(self):
        """Test that AtCommandRegistry discovers skills from bu_agent_sdk/skills."""
        skills_dir = (
            Path(__file__).resolve().parent.parent / "bu_agent_sdk" / "skills"
        )
        assert skills_dir.exists(), f"Skills directory should exist: {skills_dir}"

        registry = AtCommandRegistry()
        registry.discover_skills(skills_dir)

        # Should find at least some skills
        all_commands = registry.list_commands()
        assert len(all_commands) > 0, "Should have discovered at least one skill"

        # Verify calculator skill exists (it should be there)
        if "calculator" in registry.commands:
            calc = registry.commands["calculator"]
            assert calc.name == "calculator"
            assert calc.path == skills_dir / "calculator" / "skill.md"

    def test_at_command_preprocessing_workflow(self):
        """Test the complete workflow of @ command preprocessing."""
        # User types: "@calculator compute 123 + 456"
        user_input = "@calculator compute 123 + 456"

        # Step 1: Extract @ command name
        skill_name = extract_at_command(user_input)
        assert skill_name is not None
        assert skill_name == "calculator"

        # Step 2: Get skill from registry
        registry = AtCommandRegistry()
        skills_dir = Path(__file__).resolve().parent.parent / "bu_agent_sdk" / "skills"
        registry.discover_skills(skills_dir)

        at_cmd = registry.get_command(skill_name)
        if at_cmd is None:
            pytest.skip("Calculator skill not found")

        # Step 3: Extract message part (what comes after @<skill-name>)
        message_pattern = re.sub(rf"@{re.escape(skill_name)}\s*", "", user_input, count=1)
        assert message_pattern == "compute 123 + 456"

        # Step 4: Load skill content
        skill_content = at_cmd.load_content()
        assert skill_content is not None
        assert "calculator" in skill_content.lower()

        # Step 5: Prepend skill to message
        final_message = prepend_skill_to_message(skill_content, message_pattern)

        # Step 6: Verify final message structure
        assert final_message.index("calculator") < final_message.index("compute 123 + 456")
        assert "compute 123 + 456" in final_message

        # The final message is ready to send to the LLM
        assert len(final_message) > len(user_input)

    def test_at_command_completer_with_real_skills(self):
        """Test that AtCommandCompleter works with discovered skills."""
        registry = AtCommandRegistry()
        skills_dir = Path(__file__).resolve().parent.parent / "bu_agent_sdk" / "skills"
        registry.discover_skills(skills_dir)

        completer = AtCommandCompleter(registry)

        # Test completions for "@" (show all skills)
        from prompt_toolkit.document import Document
        doc = Document("@", 1)
        completions = list(completer.get_completions(doc, None))

        # Should have completions if skills exist
        assert len(completions) == len(registry.commands), \
            f"Should have {len(registry.commands)} completions"

        # Check that completions have expected format
        for comp in completions:
            assert comp.text  # Should have text
            assert comp.display  # Should have display text
            # display is FormattedText, convert to string for check
            display_str = str(comp.display)
            assert "@" in display_str  # Display should start with @

    def test_at_command_completer_partial_match(self):
        """Test that AtCommandCompleter handles partial matches."""
        registry = AtCommandRegistry()
        skills_dir = Path(__file__).resolve().parent.parent / "bu_agent_sdk" / "skills"
        registry.discover_skills(skills_dir)

        completer = AtCommandCompleter(registry)

        # Test for partial match like "@cal"
        from prompt_toolkit.document import Document
        doc = Document("@cal", 4)
        completions = list(completer.get_completions(doc, None))

        # Should find matching skills that start with "cal"
        matching_skills = [
            cmd for cmd in registry.commands.values()
            if cmd.name.startswith("cal")
        ]

        assert len(completions) == len(matching_skills)

    def test_at_command_completer_with_at_sign_only(self):
        """Test that AtCommandCompleter shows all commands when just @ is typed."""
        registry = AtCommandRegistry()
        skills_dir = Path(__file__).resolve().parent.parent / "bu_agent_sdk" / "skills"
        registry.discover_skills(skills_dir)

        completer = AtCommandCompleter(registry)

        # Test completions for "@" (show all skills)
        from prompt_toolkit.document import Document
        doc = Document("@", 1)
        completions = list(completer.get_completions(doc, None))

        # Should have completions if skills exist
        if len(registry.commands) > 0:
            assert len(completions) > 0

        # Each completion should have text
        for comp in completions:
            assert comp.text
            # text should be the skill name (without @)
            assert comp.text in registry.commands

    def test_extract_at_command_edge_cases(self):
        """Test edge cases for @ command extraction."""
        # @ at start with message
        assert extract_at_command("@skill do this") == "skill"

        # @ in middle
        assert extract_at_command("use @skill to do this") == "skill"

        # No @
        assert extract_at_command("just a message") is None

        # Just @
        assert extract_at_command("@") is None

        # @ with space after
        assert extract_at_command("@ something") is None

        # Multiple @, should find first
        assert extract_at_command("@first @second") == "first"

    def test_prepend_preserves_user_message(self):
        """Test that prepending skill content preserves user message."""
        skill_content = "# Skill\n\nSkill content here."
        user_message = "User request here."

        result = prepend_skill_to_message(skill_content, user_message)

        assert result.startswith(skill_content)
        assert user_message in result
        assert result.endswith(user_message)

    def test_prepend_empty_skill(self):
        """Test that empty skill content just returns user message."""
        skill_content = ""
        user_message = "User request here."

        result = prepend_skill_to_message(skill_content, user_message)

        assert result == user_message

    def test_skills_listing_format(self):
        """Test that skills can be listed in a formatted way."""
        registry = AtCommandRegistry()
        skills_dir = Path(__file__).resolve().parent.parent / "bu_agent_sdk" / "skills"
        registry.discover_skills(skills_dir)

        commands = registry.list_commands()

        # Should be organized by category
        assert isinstance(commands, dict)

        # Each category should have a list of commands
        for category, cmd_list in commands.items():
            assert isinstance(cmd_list, list)
            for cmd in cmd_list:
                assert isinstance(cmd, AtCommand)
                assert cmd.name
                assert cmd.description
                assert cmd.path

    def test_complete_user_scenario(self):
        """Test a complete user scenario with @ command."""
        # Scenario: User wants to use the calculator skill
        user_input = "@calculator What is 15 * 3 + 5?"

        # 1. Extract skill name
        skill_name = extract_at_command(user_input)
        if skill_name is None:
            pytest.skip("No @ command found in input")

        # 2. Get skill from registry
        registry = AtCommandRegistry()
        skills_dir = Path(__file__).resolve().parent.parent / "bu_agent_sdk" / "skills"
        registry.discover_skills(skills_dir)

        skill_cmd = registry.get_command(skill_name)
        if skill_cmd is None:
            pytest.skip(f"Skill '{skill_name}' not found")

        # 3. Extract the actual user message (without @<skill>)
        user_message = re.sub(rf"@{re.escape(skill_name)}\s*", "", user_input, count=1)

        # 4. Load and prepend skill content
        skill_content = skill_cmd.load_content()
        final_message = prepend_skill_to_message(skill_content, user_message)

        # 5. Verify the final message
        # Should contain skill context
        assert len(skill_content) > 0
        # Should contain the user's request
        assert "What is 15 * 3 + 5?" in final_message
        # Skill content should come before user message
        assert skill_content in final_message
        assert final_message.index(skill_content) < final_message.index("What is 15 * 3 + 5?")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
