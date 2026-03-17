import shutil
import uuid
from pathlib import Path

from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document

from cli.at_commands import (
    AtCommand,
    AtCommandCompleter,
    AtCommandRegistry,
    extract_at_command,
    load_skill_content,
    parse_at_command,
    prepend_skill_to_message,
)


def make_workspace(root: Path) -> Path:
    workspace = root / f"case-{uuid.uuid4().hex[:8]}"
    workspace.mkdir(parents=True, exist_ok=False)
    return workspace


def test_at_command_creation():
    skill_path = Path("/fake/path/SKILL.md")
    cmd = AtCommand(
        name="test-skill",
        description="A test skill",
        path=skill_path,
        category="General",
    )

    assert cmd.name == "test-skill"
    assert cmd.description == "A test skill"
    assert cmd.path == skill_path
    assert cmd.category == "General"


def test_at_command_load_content_with_real_file():
    skill_path = (
        Path(__file__).parent.parent
        / "bu_agent_sdk"
        / "skills"
        / "calculator"
        / "SKILL.md"
    )
    cmd = AtCommand(
        name="calculator",
        description="Test",
        path=skill_path,
    )

    content = cmd.load_content()
    assert isinstance(content, str)
    assert "Calculator" in content


def test_parse_skill_frontmatter():
    repo_root = Path(__file__).resolve().parent.parent
    temp_root = repo_root / ".pytest_tmp"
    temp_root.mkdir(exist_ok=True)
    workspace = make_workspace(temp_root)
    skill_path = workspace / "SKILL.md"
    try:
        skill_path.write_text(
            """---
name: test-skill
description: A test skill
category: Test Category
---

# Test Skill Content
""",
            encoding="utf-8",
        )

        cmd = AtCommand.from_file(skill_path)

        assert cmd.name == "test-skill"
        assert cmd.description == "A test skill"
        assert cmd.category == "Test Category"
        assert cmd.path == skill_path
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_registry_discovery_from_skills_directory():
    skills_dir = Path(__file__).parent.parent / "bu_agent_sdk" / "skills"

    registry = AtCommandRegistry(skills_dir)

    assert len(registry.commands) > 0
    assert "calculator" in registry.commands
    calc = registry.commands["calculator"]
    assert calc.name == "calculator"
    assert calc.path == skills_dir / "calculator" / "SKILL.md"


def test_registry_register_and_unregister():
    repo_root = Path(__file__).resolve().parent.parent
    temp_root = repo_root / ".pytest_tmp"
    temp_root.mkdir(exist_ok=True)
    workspace = make_workspace(temp_root)
    try:
        registry = AtCommandRegistry(workspace / "missing-skills")
        command = AtCommand(
            name="custom:skill",
            description="Custom skill",
            path=workspace / "custom" / "SKILL.md",
        )

        registry.register(command)
        assert registry.get_command("custom:skill") is command

        registry.unregister("custom:skill")
        assert registry.get_command("custom:skill") is None
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_at_command_completer_supports_namespaced_commands():
    repo_root = Path(__file__).resolve().parent.parent
    temp_root = repo_root / ".pytest_tmp"
    temp_root.mkdir(exist_ok=True)
    workspace = make_workspace(temp_root)
    try:
        registry = AtCommandRegistry(workspace / "missing-skills")
        registry.register(
            AtCommand(
                name="review-kit:code-review",
                description="Review helper",
                path=workspace / "review-kit" / "SKILL.md",
            )
        )

        completer = AtCommandCompleter(registry)
        doc = Document("@review-kit:c", cursor_position=len("@review-kit:c"))
        completions = list(completer.get_completions(doc, CompleteEvent()))

        assert len(completions) == 1
        assert completions[0].text == "review-kit:code-review"
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_extract_and_parse_at_command_support_namespaces():
    command = extract_at_command("@review-kit:code-review focus on tests")
    assert command == "review-kit:code-review"

    parsed_name, parsed_message = parse_at_command(
        "@review-kit:code-review focus on tests"
    )
    assert parsed_name == "review-kit:code-review"
    assert parsed_message == "focus on tests"


def test_load_skill_content_reads_from_path():
    repo_root = Path(__file__).resolve().parent.parent
    temp_root = repo_root / ".pytest_tmp"
    temp_root.mkdir(exist_ok=True)
    workspace = make_workspace(temp_root)
    skill_path = workspace / "SKILL.md"
    try:
        skill_path.write_text("# Calculator Skill", encoding="utf-8")
        assert load_skill_content(skill_path) == "# Calculator Skill"
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_prepend_skill_to_message():
    skill_content = """# Calculator Skill

This is a calculator skill.
"""
    user_message = "What is 2 + 2?"

    result = prepend_skill_to_message(skill_content, user_message)

    assert result.startswith(skill_content.rstrip())
    assert user_message in result


def test_prepend_skill_to_message_empty_skill():
    user_message = "What is 2 + 2?"
    result = prepend_skill_to_message("", user_message)
    assert result == user_message
