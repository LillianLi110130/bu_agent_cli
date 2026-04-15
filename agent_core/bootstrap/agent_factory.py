"""Shared agent bootstrap helpers for CLI and gateway runtimes."""

from __future__ import annotations

import os
import platform
from pathlib import Path
from string import Template
from typing import Any

from agent_core import Agent
from agent_core.agent.config import AgentConfig
from agent_core.llm import ChatOpenAI
from agent_core.runtime_paths import application_root, tg_agent_home
from agent_core.skill.discovery import default_skill_dirs, discover_skill_files
from tools import ALL_TOOLS, SandboxContext, get_sandbox_context

_APP_ROOT = application_root()
_PACKAGE_ROOT = _APP_ROOT / "agent_core"
_PROMPTS_DIR = _PACKAGE_ROOT / "prompts"
_SKILLS_DIR = _APP_ROOT / "skills"
_PROJECT_CONTEXT_FILENAMES = ("SOUL.md", "IDENTITY.md", "USER.md")


def _format_skills(skills: list[Any]) -> str:
    """Format skills list into a readable string for the prompt."""
    if not skills:
        return "No skills available."

    return "\n".join(
        (f"- {skill.name}\n" f"  - Path: {skill.path}\n" f"  - Desc: {skill.description}")
        for skill in sorted(skills, key=lambda item: item.name)
    )


def _load_prompt_template(template_name: str = "system.md") -> str:
    """Load a prompt template from the packaged prompts directory."""
    template_path = _PROMPTS_DIR / template_name
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {template_path}")
    return template_path.read_text(encoding="utf-8")


def _resolve_project_context_paths(prompts_dir: Path | None = None) -> list[Path]:
    """Resolve project context files with ~/.tg_agent priority and prompts fallback."""
    resolved_prompts_dir = prompts_dir or _PROMPTS_DIR
    user_home_dir = tg_agent_home()
    resolved_paths: list[Path] = []

    for filename in _PROJECT_CONTEXT_FILENAMES:
        home_path = user_home_dir / filename
        prompt_path = resolved_prompts_dir / filename
        if home_path.exists():
            resolved_paths.append(home_path)
        elif prompt_path.exists():
            resolved_paths.append(prompt_path)

    return resolved_paths


def build_project_context(prompts_dir: Path | None = None) -> str:
    """Render project context files into a dedicated prompt section."""
    existing_paths = _resolve_project_context_paths(prompts_dir=prompts_dir)

    lines = [
        "## Project Context",
        "The following project context files have been loaded:",
    ]
    if any(path.name == "SOUL.md" for path in existing_paths):
        lines.append(
            "If SOUL.md is present, embody its persona and tone. Avoid stiff, generic "
            "replies; follow its guidance unless higher-priority instructions override it."
        )

    for path in existing_paths:
        lines.extend(
            [
                "",
                f"### {path}",
                path.read_text(encoding="utf-8").rstrip(),
            ]
        )

    return "\n".join(lines)


def _get_system_info() -> str:
    """Collect and format basic operating system information."""
    system = platform.system()
    release = platform.release()

    if system == "Windows":
        version = release.split(".")[0] if "." in release else release
        return f"Windows {version}"

    if system == "Linux":
        try:
            import distro

            distro_name = distro.name()
            distro_version = distro.version()
            if distro_version:
                return f"{distro_name} {distro_version}"
            return distro_name
        except ImportError:
            try:
                with open("/etc/os-release", encoding="utf-8") as os_release_file:
                    content = os_release_file.read()
                for line in content.split("\n"):
                    if line.startswith("PRETTY_NAME="):
                        pretty_name = line.split("=", 1)[1].strip('"')
                        return pretty_name
            except (IOError, OSError):
                pass
            return f"Linux {release}"

    if system == "Darwin":
        version = platform.mac_ver()[0]
        if version:
            return f"macOS {version}"
        return "macOS"

    return f"{system} {release}"


def _build_agents_text(agent_registry: Any) -> str:
    """Format callable agent metadata into the prompt section."""
    callable_agents = agent_registry.list_callable_agents()

    if not callable_agents:
        return "No subagents available."

    agent_lines: list[str] = []
    for agent_name in callable_agents:
        config = agent_registry.get_config(agent_name)
        if config:
            agent_lines.append(f"- {agent_name}: {config.description}")

    return "\n".join(agent_lines) or "No subagents available."


def build_system_prompt(
    working_dir: Path,
    builtin_skills_dir: Path | None = None,
    skill_registry: Any | None = None,
    agent_registry: Any | None = None,
) -> str:
    """Build the system prompt using built-in and custom skills."""
    from agent_core.agent.registry import get_agent_registry

    if skill_registry is None:
        resolved_builtin_skills_dir = builtin_skills_dir or _SKILLS_DIR
        skills = discover_skill_files(
            default_skill_dirs(
                workspace_root=working_dir,
                builtin_skills_dir=resolved_builtin_skills_dir,
            )
        )
    else:
        skills = skill_registry.get_all()

    resolved_agent_registry = agent_registry or get_agent_registry()

    template = Template(_load_prompt_template("system.md"))
    return template.substitute(
        SKILLS=_format_skills(skills),
        WORKING_DIR=str(working_dir),
        SUBAGENTS=_build_agents_text(resolved_agent_registry),
        SYSTEM_INFO=_get_system_info(),
        PROJECT_CONTEXT=build_project_context(),
    )


def build_subagent_system_prompt(system_prompt: str) -> str:
    """Append shared runtime context for subagent system prompts."""
    return (
        f"{system_prompt}\n\n## System Information\n\n"
        f"The current environment: {_get_system_info()}"
    )


def create_llm(model: str | None = None) -> ChatOpenAI:
    """Create an LLM instance from a preset or environment variables."""
    resolved_model = model or (os.getenv("LLM_MODEL") or "").strip() or "GLM-4.7"
    base_url = (os.getenv("LLM_BASE_URL") or "").strip() or "https://open.bigmodel.cn/api/coding/paas/v4"
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip() or "OPENAI_API_KEY"

    return ChatOpenAI(
        model=resolved_model,
        api_key=api_key,
        base_url=base_url,
    )


def create_agent(
    model: str | None,
    root_dir: Path | str | None = None,
    mode: str = "primary",
    agent_config: AgentConfig | None = None,
) -> tuple[Agent, SandboxContext]:
    """Create a configured Agent and SandboxContext."""
    from agent_core.agent.registry import get_agent_registry
    from agent_core.agent.subagent_manager import SubagentManager

    ctx = SandboxContext.create(root_dir)
    llm = create_llm(model)
    system_prompt = build_system_prompt(ctx.working_dir)
    registry = get_agent_registry()

    subagent_manager = SubagentManager(
        agent_factory=_create_subagent_factory,
        registry=registry,
        all_tools=ALL_TOOLS,
        workspace=ctx.working_dir,
        context=ctx,
    )
    ctx.subagent_manager = subagent_manager

    agent = Agent(
        llm=llm,
        tools=ALL_TOOLS,
        system_prompt=system_prompt,
        dependency_overrides={get_sandbox_context: lambda: ctx},
        mode=mode,
        agent_config=agent_config,
    )

    subagent_manager.set_main_agent(agent)
    return agent, ctx


def _create_subagent_factory(config: AgentConfig, parent_ctx: Any, all_tools: list) -> Agent:
    """Factory function to create subagent instances."""
    from config.model_config import get_model_config

    model, base_url, api_key = get_model_config(config.model)

    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=config.temperature,
    )

    return Agent(
        llm=llm,
        tools=all_tools,
        system_prompt=build_subagent_system_prompt(config.system_prompt),
        mode="subagent",
        agent_config=config,
        dependency_overrides={get_sandbox_context: lambda: parent_ctx},
    )
