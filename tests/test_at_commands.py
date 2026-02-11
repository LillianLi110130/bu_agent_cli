import tempfile
from pathlib import Path
from prompt_toolkit.document import Document
from prompt_toolkit.completion import CompleteEvent
import yaml
from cli.at_commands import AtCommand, AtCommandRegistry, AtCommandCompleter

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

