import shutil
import uuid
from pathlib import Path

from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document

from agent_core.skill.discovery import default_skill_dirs, discover_skill_files
from tg_crab_main import create_runtime_registries
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


def write_skill(
    skills_root: Path,
    directory_name: str,
    *,
    name: str | None = None,
    description: str = "A test skill",
    category: str = "General",
    filename: str = "SKILL.md",
) -> Path:
    skill_dir = skills_root / directory_name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_path = skill_dir / filename
    skill_path.write_text(
        "\n".join(
            [
                "---",
                f"name: {name or directory_name}",
                f"description: {description}",
                f"category: {category}",
                "---",
                "",
                f"# {name or directory_name}",
            ]
        ),
        encoding="utf-8",
    )
    return skill_path


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
    skill_path = Path(__file__).parent.parent / "skills" / "calculator" / "SKILL.md"
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
    skills_dir = Path(__file__).parent.parent / "skills"

    registry = AtCommandRegistry(skills_dir)

    assert len(registry.commands) > 0
    assert "calculator" in registry.commands
    calc = registry.commands["calculator"]
    assert calc.name == "calculator"
    assert calc.path == skills_dir / "calculator" / "SKILL.md"


def test_registry_discovers_builtin_user_and_project_skills(tmp_path: Path):
    builtin_skills = tmp_path / "builtin_skills"
    user_skills = tmp_path / "user_skills"
    project_skills = tmp_path / "project_skills"

    write_skill(builtin_skills, "builtin-only", description="builtin skill")
    write_skill(user_skills, "user-only", description="user skill")
    write_skill(project_skills, "project-only", description="project skill")

    registry = AtCommandRegistry(skill_dirs=[builtin_skills, user_skills, project_skills])

    assert set(registry.commands) >= {"builtin-only", "user-only", "project-only"}


def test_registry_prefers_project_over_user_over_builtin_for_same_skill_name(tmp_path: Path):
    builtin_skills = tmp_path / "builtin_skills"
    user_skills = tmp_path / "user_skills"
    project_skills = tmp_path / "project_skills"

    write_skill(builtin_skills, "builtin-shared", name="shared", description="builtin override")
    write_skill(user_skills, "user-shared", name="shared", description="user override")
    project_path = write_skill(
        project_skills,
        "project-shared",
        name="shared",
        description="project override",
    )

    registry = AtCommandRegistry(skill_dirs=[builtin_skills, user_skills, project_skills])

    command = registry.get("shared")
    assert command is not None
    assert command.description == "project override"
    assert command.path == project_path


def test_registry_prefers_user_over_builtin_when_project_missing(tmp_path: Path):
    builtin_skills = tmp_path / "builtin_skills"
    user_skills = tmp_path / "user_skills"

    write_skill(builtin_skills, "builtin-shared", name="shared", description="builtin override")
    user_path = write_skill(user_skills, "user-shared", name="shared", description="user override")

    registry = AtCommandRegistry(skill_dirs=[builtin_skills, user_skills])

    command = registry.get("shared")
    assert command is not None
    assert command.description == "user override"
    assert command.path == user_path


def test_discover_skill_files_returns_highest_priority_version(tmp_path: Path):
    builtin_skills = tmp_path / "builtin_skills"
    user_skills = tmp_path / "user_skills"
    project_skills = tmp_path / "project_skills"

    write_skill(builtin_skills, "builtin-shared", name="shared", description="builtin override")
    write_skill(user_skills, "user-shared", name="shared", description="user override")
    project_path = write_skill(
        project_skills,
        "project-shared",
        name="shared",
        description="project override",
    )

    discovered = discover_skill_files([builtin_skills, user_skills, project_skills])

    assert [skill.name for skill in discovered] == ["shared"]
    assert discovered[0].description == "project override"
    assert discovered[0].path == project_path


def test_default_skill_dirs_syncs_packaged_skills_to_user_builtin_root(tmp_path: Path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    packaged_skills = tmp_path / "packaged_skills"
    home_dir = tmp_path / "home"

    write_skill(packaged_skills, "builtin-only", description="builtin skill")

    monkeypatch.setenv("HOME", str(home_dir))
    resolved_dirs = default_skill_dirs(workspace, packaged_skills)

    builtin_root = home_dir / ".tg_agent" / "skills" / ".builtin"
    assert resolved_dirs[0] == builtin_root
    assert resolved_dirs[1] == home_dir / ".tg_agent" / "skills"
    assert resolved_dirs[2] == workspace / "skills"
    assert (builtin_root / "builtin-only" / "SKILL.md").exists()


def test_create_runtime_registries_loads_workspace_and_user_skills(
    tmp_path: Path,
    monkeypatch,
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    builtin_skills = tmp_path / "builtin_skills"
    home_dir = tmp_path / "home"
    user_skills = home_dir / ".tg_agent" / "skills"
    project_skills = workspace / "skills"
    empty_plugins = tmp_path / "plugins"
    empty_agents = tmp_path / "agents"
    empty_plugins.mkdir()
    empty_agents.mkdir()

    write_skill(builtin_skills, "builtin-only", description="builtin skill")
    write_skill(user_skills, "user-only", description="user skill")
    write_skill(project_skills, "project-only", description="project skill")

    monkeypatch.setenv("HOME", str(home_dir))
    runtime = create_runtime_registries(
        workspace_root=workspace,
        plugin_dirs=[("builtin", empty_plugins)],
        skills_dir=builtin_skills,
        agents_dir=empty_agents,
    )

    assert runtime.skill_registry.get("builtin-only") is not None
    assert runtime.skill_registry.get("user-only") is not None
    assert runtime.skill_registry.get("project-only") is not None
    builtin_path = runtime.skill_registry.get("builtin-only").path
    assert builtin_path.parent == (home_dir / ".tg_agent" / "skills" / ".builtin" / "builtin-only")
    assert builtin_path.name.lower() == "skill.md"


def test_registry_skips_invalid_yaml_skill_files(tmp_path: Path):
    skills_root = tmp_path / "skills"
    invalid_skill_dir = skills_root / "broken"
    invalid_skill_dir.mkdir(parents=True, exist_ok=True)
    (invalid_skill_dir / "SKILL.md").write_text(
        "---\nname: broken\ndescription: [oops\n---\n\n# broken\n",
        encoding="utf-8",
    )
    valid_path = write_skill(skills_root, "valid", description="valid skill")

    registry = AtCommandRegistry(skill_dirs=[skills_root])

    assert registry.get("valid") is not None
    assert registry.get("valid").path == valid_path
    assert "broken" not in registry.commands


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

    parsed_name, parsed_message = parse_at_command("@review-kit:code-review focus on tests")
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
