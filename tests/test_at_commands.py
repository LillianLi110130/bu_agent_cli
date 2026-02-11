from pathlib import Path
from cli.at_commands import AtCommand

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
