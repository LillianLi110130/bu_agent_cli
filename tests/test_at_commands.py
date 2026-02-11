import tempfile
from pathlib import Path
from prompt_toolkit.document import Document
from prompt_toolkit.completion import CompleteEvent
import yaml
from cli.at_commands import (
    AtCommand,
    AtCommandRegistry,
    AtCommandCompleter,
    extract_at_command,
    load_skill_content,
    prepend_skill_to_message,
)

def test_at_command_creation():
    skill_path = Path("/fake/path/skill.md")
    cmd = AtCommand(
        name="test-skill",
        description="A test skill",
        path=skill_path,
        category="General"
    )

    assert cmd.name == "test-skill"
    assert cmd.description == "A test skill"
    assert cmd.path == skill_path
    assert cmd.category == "General"

def test_at_command_load_content_with_real_file():
    # This test will be updated in a later task
    # For now, just test that the method exists
    skill_path = Path(__file__).parent.parent / "bu_agent_sdk" / "skills" / "calculator" / "skill.md"
    cmd = AtCommand(
        name="calculator",
        description="Test",
        path=skill_path
    )

    content = cmd.load_content()
    assert isinstance(content, str)
    assert "Calculator" in content

def test_parse_skill_frontmatter():
    """Test that skill.md frontmatter is parsed correctly"""

    # Create a temporary skill.md file
    skill_content = """---
name: test-skill
description: A test skill
category: Test Category
---

# Test Skill Content

This is the content of the skill.
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write(skill_content)
        temp_path = Path(f.name)

    try:
        cmd = AtCommand.from_file(temp_path)

        assert cmd.name == "test-skill"
        assert cmd.description == "A test skill"
        assert cmd.category == "Test Category"
        assert cmd.path == temp_path
    finally:
        temp_path.unlink()

def test_registry_discovery_from_skills_directory():
    """Test that AtCommandRegistry auto-discovers skills"""
    skills_dir = Path(__file__).parent.parent / "bu_agent_sdk" / "skills"

    registry = AtCommandRegistry()
    registry.discover_skills(skills_dir)

    # Should find at least some known skills
    assert len(registry.commands) > 0

    # Check calculator is found
    assert "calculator" in registry.commands
    calc = registry.commands["calculator"]
    assert calc.name == "calculator"
    assert calc.path == skills_dir / "calculator" / "skill.md"

def test_registry_get_command():
    """Test getting a command by name"""
    skills_dir = Path(__file__).parent.parent / "bu_agent_sdk" / "skills"

    registry = AtCommandRegistry()
    registry.discover_skills(skills_dir)

    cmd = registry.get_command("calculator")
    assert cmd is not None
    assert cmd.name == "calculator"

    # Non-existent command returns None
    cmd = registry.get_command("nonexistent")
    assert cmd is None

def test_registry_list_commands():
    """Test listing all commands grouped by category"""
    skills_dir = Path(__file__).parent.parent / "bu_agent_sdk" / "skills"

    registry = AtCommandRegistry()
    registry.discover_skills(skills_dir)

    commands = registry.list_commands()
    assert len(commands) > 0

    # Should be a dict with categories as keys
    assert isinstance(commands, dict)

def test_at_command_completer():
    """Test AtCommandCompleter provides completions for @ commands"""
    skills_dir = Path(__file__).parent.parent / "bu_agent_sdk" / "skills"

    registry = AtCommandRegistry()
    registry.discover_skills(skills_dir)

    completer = AtCommandCompleter(registry)

    # Test completion for "@cal"
    doc = Document("@cal", cursor_position=4)
    event = CompleteEvent()
    completions = list(completer.get_completions(doc, event))

    # Should have at least 'calculator' in completions
    assert len(completions) > 0
    texts = [c.text for c in completions]
    assert "calculator" in texts

def test_at_command_completer_no_trigger_on_at_sign():
    """Test completer doesn't trigger when not starting with @"""
    skills_dir = Path(__file__).parent.parent / "bu_agent_sdk" / "skills"

    registry = AtCommandRegistry()
    registry.discover_skills(skills_dir)

    completer = AtCommandCompleter(registry)

    # Test completion for "cal" (without @) - should not trigger
    doc = Document("cal", cursor_position=3)
    event = CompleteEvent()
    completions = list(completer.get_completions(doc, event))

    # Should have no completions
    assert len(completions) == 0

def test_extract_at_command():
    """Test extracting @ command name from user message"""
    # Simple @ command
    cmd = extract_at_command("@calculator help me with math")
    assert cmd == "calculator"

    # @ command at start with space after
    cmd = extract_at_command("@calculator 2 + 2")
    assert cmd == "calculator"

    # No @ command
    cmd = extract_at_command("help me with math")
    assert cmd is None

    # @ in middle of text (should be detected)
    cmd = extract_at_command("I need @calculator help")
    assert cmd == "calculator"

    # Just @ with nothing
    cmd = extract_at_command("@")
    assert cmd is None

    # @ followed by space
    cmd = extract_at_command("@ something")
    assert cmd is None

def test_load_skill_content():
    """Test loading skill content via registry"""
    skills_dir = Path(__file__).parent.parent / "bu_agent_sdk" / "skills"

    registry = AtCommandRegistry()
    registry.discover_skills(skills_dir)

    content = load_skill_content(registry, "calculator")
    assert content is not None
    assert len(content) > 0

    # Should contain markdown from skill file
    assert "calculator" in content.lower()

def test_load_skill_content_nonexistent():
    """Test loading content for non-existent skill"""
    registry = AtCommandRegistry()

    content = load_skill_content(registry, "nonexistent")
    assert content is None

def test_prepend_skill_to_message():
    """Test prepending skill content to user message"""
    skill_content = """# Calculator Skill

This is a calculator skill.
"""
    user_message = "What is 2 + 2?"

    result = prepend_skill_to_message(skill_content, user_message)

    assert result.startswith(skill_content)
    assert user_message in result

def test_prepend_skill_to_message_with_separator():
    """Test that separator is added between skill and message"""
    skill_content = "# Calculator Skill"
    user_message = "What is 2 + 2?"

    result = prepend_skill_to_message(skill_content, user_message)

    # Should have separator
    assert "\n\n" in result or "\n" in result

def test_prepend_skill_to_message_empty_skill():
    """Test prepending with empty skill content"""
    skill_content = ""
    user_message = "What is 2 + 2?"

    result = prepend_skill_to_message(skill_content, user_message)

    # Should just return the user message
    assert result == user_message

