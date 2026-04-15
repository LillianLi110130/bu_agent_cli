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


def _write_project_context_file(workspace: Path, filename: str, content: str) -> Path:
    project_file = workspace / filename
    project_file.parent.mkdir(parents=True, exist_ok=True)
    project_file.write_text(content, encoding="utf-8")
    return project_file


def _write_prompt_template(prompts_root: Path) -> Path:
    template_path = prompts_root / "system.md"
    template_path.parent.mkdir(parents=True, exist_ok=True)
    template_path.write_text(
        "\n".join(
            [
                "System",
                "${SYSTEM_INFO}",
                "${PROJECT_CONTEXT}",
                "${SKILLS}",
                "${WORKING_DIR}",
                "${SUBAGENTS}",
            ]
        ),
        encoding="utf-8",
    )
    return template_path


def test_build_system_prompt_uses_packaged_skills_outside_workspace(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_module("agent_core.bootstrap.agent_factory")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    prompt = module.build_system_prompt(tmp_path)

    assert "brainstorming" in prompt
    assert str(tmp_path) in prompt


def test_build_system_prompt_accepts_runtime_registries(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_module("agent_core.bootstrap.agent_factory")

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    prompt_dir = tmp_path / "prompts"
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(module, "_PROMPTS_DIR", prompt_dir)
    _write_prompt_template(prompt_dir)

    class FakeSkill:
        def __init__(self, name: str, description: str) -> None:
            self.name = name
            self.description = description
            self.path = f"/skills/{name}"

    class FakeSkillRegistry:
        def get_all(self):
            return [FakeSkill("runtime-skill", "runtime description")]

    class FakeAgentConfig:
        description = "runtime agent description"

    class FakeAgentRegistry:
        def list_callable_agents(self):
            return ["runtime-agent"]

        def get_config(self, name):
            assert name == "runtime-agent"
            return FakeAgentConfig()

    prompt = module.build_system_prompt(
        workspace,
        skill_registry=FakeSkillRegistry(),
        agent_registry=FakeAgentRegistry(),
    )

    assert "runtime-skill" in prompt
    assert "runtime description" in prompt
    assert "runtime-agent: runtime agent description" in prompt
    assert str(workspace) in prompt


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


def test_build_system_prompt_prefers_tg_agent_home_project_context_files(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_module("agent_core.bootstrap.agent_factory")

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    home_dir = tmp_path / "home"
    prompt_dir = tmp_path / "prompts"
    monkeypatch.setenv("HOME", str(home_dir))
    monkeypatch.setattr(module, "_PROMPTS_DIR", prompt_dir)
    _write_prompt_template(prompt_dir)

    soul_path = _write_project_context_file(home_dir / ".tg_agent", "SOUL.md", "# home soul")
    identity_path = _write_project_context_file(
        home_dir / ".tg_agent", "IDENTITY.md", "# home identity"
    )
    user_path = _write_project_context_file(home_dir / ".tg_agent", "USER.md", "# home user")
    _write_project_context_file(prompt_dir, "SOUL.md", "# fallback soul")
    _write_project_context_file(prompt_dir, "IDENTITY.md", "# fallback identity")
    _write_project_context_file(prompt_dir, "USER.md", "# fallback user")

    prompt = module.build_system_prompt(workspace)

    assert "# Project Context" in prompt
    assert "The following project context files have been loaded:" in prompt
    assert (
        "If SOUL.md is present, embody its persona and tone. Avoid stiff, generic replies; "
        "follow its guidance unless higher-priority instructions override it."
    ) in prompt
    assert f"### {soul_path}" in prompt
    assert "# home soul" in prompt
    assert f"### {identity_path}" in prompt
    assert "# home identity" in prompt
    assert f"### {user_path}" in prompt
    assert "# home user" in prompt
    assert "# fallback soul" not in prompt


def test_build_system_prompt_falls_back_to_prompt_context_files(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_module("agent_core.bootstrap.agent_factory")

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    home_dir = tmp_path / "home"
    prompt_dir = tmp_path / "prompts"
    monkeypatch.setenv("HOME", str(home_dir))
    monkeypatch.setattr(module, "_PROMPTS_DIR", prompt_dir)
    _write_prompt_template(prompt_dir)

    soul_path = _write_project_context_file(prompt_dir, "SOUL.md", "# prompt soul")
    identity_path = _write_project_context_file(prompt_dir, "IDENTITY.md", "# prompt identity")
    user_path = _write_project_context_file(prompt_dir, "USER.md", "# prompt user")

    prompt = module.build_system_prompt(workspace)

    assert f"### {soul_path}" in prompt
    assert "# prompt soul" in prompt
    assert f"### {identity_path}" in prompt
    assert "# prompt identity" in prompt
    assert f"### {user_path}" in prompt
    assert "# prompt user" in prompt


def test_build_system_prompt_keeps_empty_project_context_section_when_files_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_module("agent_core.bootstrap.agent_factory")

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(module, "_PROMPTS_DIR", tmp_path / "prompts")
    _write_prompt_template(tmp_path / "prompts")

    prompt = module.build_system_prompt(workspace)

    assert "# Project Context" in prompt
    assert "The following project context files have been loaded:" in prompt
    assert "If SOUL.md is present, embody its persona and tone." not in prompt
    assert "### " not in prompt


def test_create_subagent_factory_excludes_project_context(monkeypatch) -> None:
    module = _load_module("agent_core.bootstrap.agent_factory")
    config_module = _load_module("agent_core.agent.config")

    monkeypatch.setattr(
        "config.model_config.get_model_config",
        lambda _model: ("dummy-model", "https://example.invalid", "test-key"),
    )

    config = config_module.AgentConfig(
        name="reviewer",
        description="test subagent",
        system_prompt="You are a subagent.",
        tools=[],
        model="dummy-model",
        mode="subagent",
    )

    agent = module._create_subagent_factory(config=config, parent_ctx=object(), all_tools=[])

    assert "You are a subagent." in agent.system_prompt
    assert "## System Information" in agent.system_prompt
    assert "# Project Context" not in agent.system_prompt
    assert "The following project context files have been loaded:" not in agent.system_prompt


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
