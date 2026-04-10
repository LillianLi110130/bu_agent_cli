import importlib
from pathlib import Path

import pytest


class DummyLLM:
    def __init__(self, model: str = "dummy-model") -> None:
        self.model = model

    @property
    def provider(self) -> str:
        return "dummy"

    @property
    def name(self) -> str:
        return self.model

    async def ainvoke(self, messages, tools=None, tool_choice=None, **kwargs):
        raise NotImplementedError

    async def astream(self, messages, tools=None, tool_choice=None, **kwargs):
        if False:
            yield None


def _load_module(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        pytest.fail(f"Expected module '{module_name}' to exist: {exc}")


def _write_skill(
    skills_root: Path,
    directory_name: str,
    *,
    name: str | None = None,
    description: str = "A test skill",
) -> Path:
    skill_dir = skills_root / directory_name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_path = skill_dir / "SKILL.md"
    skill_path.write_text(
        "\n".join(
            [
                "---",
                f"name: {name or directory_name}",
                f"description: {description}",
                "---",
                "",
                f"# {name or directory_name}",
            ]
        ),
        encoding="utf-8",
    )
    return skill_path


def test_build_system_prompt_uses_packaged_skills_outside_workspace(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_module("agent_core.bootstrap.agent_factory")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    prompt = module.build_system_prompt(tmp_path)

    assert "brainstorming" in prompt
    assert str(tmp_path) in prompt


def test_build_system_prompt_includes_user_and_workspace_skills(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_module("agent_core.bootstrap.agent_factory")

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    builtin_skills = tmp_path / "builtin_skills"
    home_dir = tmp_path / "home"
    user_skills = home_dir / ".tg_agent" / "skills"
    project_skills = workspace / "skills"

    _write_skill(builtin_skills, "builtin-only", description="builtin skill")
    _write_skill(user_skills, "user-only", description="user skill")
    _write_skill(project_skills, "project-only", description="project skill")

    monkeypatch.setenv("HOME", str(home_dir))
    prompt = module.build_system_prompt(workspace, builtin_skills_dir=builtin_skills)

    assert "builtin-only" in prompt
    assert "user-only" in prompt
    assert "project-only" in prompt
    assert str(home_dir / ".tg_agent" / "skills" / ".builtin" / "builtin-only").lower() in prompt.lower()


def test_build_system_prompt_prefers_project_skill_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_module("agent_core.bootstrap.agent_factory")

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    builtin_skills = tmp_path / "builtin_skills"
    home_dir = tmp_path / "home"
    user_skills = home_dir / ".tg_agent" / "skills"
    project_skills = workspace / "skills"

    _write_skill(builtin_skills, "builtin-shared", name="shared", description="builtin override")
    _write_skill(user_skills, "user-shared", name="shared", description="user override")
    _write_skill(project_skills, "project-shared", name="shared", description="project override")

    monkeypatch.setenv("HOME", str(home_dir))
    prompt = module.build_system_prompt(workspace, builtin_skills_dir=builtin_skills)

    assert "project override" in prompt
    assert "builtin override" not in prompt


def test_claude_code_build_system_prompt_includes_project_context(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_module("claude_code")
    skill_module = _load_module("cli.at_commands")
    registry_module = _load_module("agent_core.agent.registry")

    monkeypatch.setattr(module, "build_project_context", lambda: "PROJECT_CONTEXT_MARKER")

    skill_registry = skill_module.AtCommandRegistry(skill_dirs=[tmp_path / "skills"])
    agent_registry = registry_module.AgentRegistry(tmp_path / "agents")

    prompt = module._build_system_prompt(
        tmp_path,
        skill_registry=skill_registry,
        agent_registry=agent_registry,
    )

    assert "PROJECT_CONTEXT_MARKER" in prompt


def test_sync_workspace_agents_md_deduplicates_and_replaces_content(tmp_path: Path) -> None:
    module = _load_module("agent_core.bootstrap.session_bootstrap")
    sdk_module = _load_module("agent_core")
    state = module.WorkspaceInstructionState()
    agent = sdk_module.Agent(llm=DummyLLM(), tools=[], system_prompt="system prompt")

    agents_md_path = tmp_path / "TGAGENTS.md"
    agents_md_path.write_text("first rule", encoding="utf-8")

    state = module.sync_workspace_agents_md(agent, tmp_path, state)
    state = module.sync_workspace_agents_md(agent, tmp_path, state)

    first_system_messages = [
        message
        for message in agent.messages
        if message.role == "system" and getattr(message, "content", "") == "first rule"
    ]
    system_messages = [message for message in agent.messages if message.role == "system"]

    assert len(first_system_messages) == 1
    assert len(system_messages) == 2
    assert state.injected_content == "first rule"

    agents_md_path.write_text("second rule", encoding="utf-8")
    state = module.sync_workspace_agents_md(agent, tmp_path, state)

    system_contents = [
        getattr(message, "content", "") for message in agent.messages if message.role == "system"
    ]

    assert "first rule" not in system_contents
    assert "second rule" in system_contents
    assert state.injected_content == "second rule"
