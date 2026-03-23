"""
Claude Code CLI - An interactive coding assistant with file operations.

Includes bash, file operations (read/write/edit), search (glob/grep),
todo management, and task completion - all with dependency injection
for secure filesystem access.

Usage:
    py -3.10 claude_code.py
    py -3.10 claude_code.py --model gpt-4o
    py -3.10 claude_code.py --root-dir ./other-project

Environment Variables:
    LLM_MODEL: Model to use (default: GLM-4.7)
    LLM_BASE_URL: LLM API base URL (default: https://open.bigmodel.cn/api/coding/paas/v4)
    OPENAI_API_KEY: API key for OpenAI-compatible APIs
"""

import argparse
import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bu_agent_sdk import Agent
from bu_agent_sdk.agent import (
    AgentHook,
    AuditHook,
    HumanApprovalHook,
    build_default_approval_policy,
)
from bu_agent_sdk.llm import ChatOpenAI
from bu_agent_sdk.agent.config import AgentConfig
from bu_agent_sdk.agent.registry import AgentRegistry
from bu_agent_sdk.plugin import PluginManager
from cli.app import ClaudeCodeCLI
from cli.at_commands import AtCommand, AtCommandRegistry
from cli.slash_commands import SlashCommandRegistry
from tools import ALL_TOOLS, SandboxContext, get_sandbox_context


# =============================================================================
# Prompt & Skills Loading
# =============================================================================

# Directory paths
_SCRIPT_DIR = Path(__file__).parent.resolve()
_PROMPTS_DIR = _SCRIPT_DIR / "bu_agent_sdk" / "prompts"
_SKILLS_DIR = _SCRIPT_DIR / "bu_agent_sdk" / "skills"
_AGENTS_DIR = _PROMPTS_DIR / "agents"
_PLUGINS_DIR = _SCRIPT_DIR / "plugins"


@dataclass(slots=True)
class RuntimeRegistries:
    slash_registry: SlashCommandRegistry
    skill_registry: AtCommandRegistry
    agent_registry: AgentRegistry
    plugin_manager: PluginManager


def _format_skills(skills: list[AtCommand]) -> str:
    """Format skills list into a readable string for the prompt."""
    if not skills:
        return "No skills available."

    skills_formatted = "\n".join(
        (f"- {skill.name}\n" f"  - Path: {skill.path}\n" f"  - Desc: {skill.description}")
        for skill in sorted(skills, key=lambda item: item.name)
    )
    return skills_formatted


def _load_prompt_template(template_name: str = "system.md") -> str:
    """Load a prompt template from the prompts directory."""
    template_path = _PROMPTS_DIR / template_name
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {template_path}")
    return template_path.read_text(encoding="utf-8")


def _get_system_info() -> str:
    """收集并格式化系统信息（仅操作系统名称和版本）"""
    import platform

    system = platform.system()
    release = platform.release()

    # 根据不同平台格式化系统信息
    if system == "Windows":
        # Windows: Windows 10, Windows 11
        # 从 release 中提取版本号 (如 10, 11)
        version = release.split(".")[0] if "." in release else release
        return f"Windows {version}"

    elif system == "Linux":
        # Linux: 尝试获取发行版信息 (Ubuntu 20.04, CentOS 7, etc.)
        try:
            import distro

            # 使用 distro 模块获取更友好的发行版名称
            distro_name = distro.name()
            distro_version = distro.version()
            if distro_version:
                return f"{distro_name} {distro_version}"
            return distro_name
        except ImportError:
            # 如果 distro 不可用，回退到基本方法
            # Linux 通常在 /etc/os-release 中有发行版信息
            try:
                with open("/etc/os-release") as f:
                    content = f.read()
                    for line in content.split("\n"):
                        if line.startswith("PRETTY_NAME="):
                            pretty_name = line.split("=", 1)[1].strip('"')
                            return pretty_name
            except (IOError, OSError):
                pass
            # 最终回退：Linux 内核版本
            return f"Linux {release}"

    elif system == "Darwin":
        # macOS: macOS 14.0
        version = platform.mac_ver()[0]
        if version:
            return f"macOS {version}"
        return "macOS"

    else:
        # 其他未知系统
        return f"{system} {release}"


def create_runtime_registries(
    *,
    workspace_root: Path,
    plugin_dir: Path | None = None,
    plugin_dirs: list[tuple[str, Path]] | None = None,
    skills_dir: Path | None = None,
    agents_dir: Path | None = None,
) -> RuntimeRegistries:
    """Create shared registries for built-ins and workspace plugins."""
    slash_registry = SlashCommandRegistry()
    skill_registry = AtCommandRegistry(skills_dir or _SKILLS_DIR)
    agent_registry = AgentRegistry(agents_dir or _AGENTS_DIR)
    resolved_plugin_dirs = plugin_dirs or [
        ("builtin", plugin_dir or _PLUGINS_DIR),
        ("workspace", workspace_root / ".tg_agent" / "plugins"),
    ]
    plugin_manager = PluginManager(
        plugin_dir=None,
        plugin_dirs=resolved_plugin_dirs,
        slash_registry=slash_registry,
        skill_registry=skill_registry,
        agent_registry=agent_registry,
    )
    plugin_manager.load_all()
    return RuntimeRegistries(
        slash_registry=slash_registry,
        skill_registry=skill_registry,
        agent_registry=agent_registry,
        plugin_manager=plugin_manager,
    )


def _build_system_prompt(
    working_dir: Path,
    skill_registry: AtCommandRegistry,
    agent_registry: AgentRegistry,
) -> str:
    """Build the system prompt by loading template and injecting skills."""
    from string import Template

    # Load skills and Format skills
    skills = skill_registry.get_all()
    skills_text = _format_skills(skills)

    # Load subagents
    callable_agents = agent_registry.list_callable_agents()

    # Format subagents
    agents_text = ""
    if callable_agents:
        agents_lines = []
        for agent_name in callable_agents:
            config = agent_registry.get_config(agent_name)
            if config:
                agents_lines.append(f"- {agent_name}: {config.description}")
        agents_text = "\n".join(agents_lines)
    else:
        agents_text = "No subagents available."

    # Get system information
    system_info_text = _get_system_info()

    template_str = _load_prompt_template("system.md")

    template = Template(template_str)
    prompt = template.substitute(
        SKILLS=skills_text,
        WORKING_DIR=str(working_dir),
        SUBAGENTS=agents_text,
        SYSTEM_INFO=system_info_text,
    )

    return prompt


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Claude Code CLI - Interactive coding assistant")
    parser.add_argument(
        "--model",
        "-m",
        help="LLM model to use (default: from LLM_MODEL env var or GLM-4.7)",
    )
    parser.add_argument(
        "--root-dir",
        "-r",
        help="Root directory for sandbox (default: current working directory)",
    )
    return parser.parse_args()


def create_llm(model: str | None = None) -> ChatOpenAI:
    """Create LLM instance based on environment or model parameter."""
    model = model or os.getenv("LLM_MODEL", "GLM-4.7")
    base_url = os.getenv("LLM_BASE_URL", "https://open.bigmodel.cn/api/coding/paas/v4")
    api_key = os.getenv("OPENAI_API_KEY", "OPENAI_API_KEY")

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
    )


def build_agent_hooks(*, mode: str) -> list[AgentHook]:
    """Build runtime hooks for agent instances.

    Keep the default set non-invasive for CLI usage. The built-in
    FinishGuardHook is already attached inside Agent, so this helper
    only adds optional extra hooks.
    """
    hooks: list[AgentHook] = [
        HumanApprovalHook(policy=build_default_approval_policy(mode)),
        AuditHook(),
    ]

    return hooks


def create_agent(
    model: str | None,
    root_dir: Path | str | None = None,
    mode: str = "primary",
    agent_config: AgentConfig | None = None,
    runtime_registries: RuntimeRegistries | None = None,
) -> tuple[Agent, SandboxContext, RuntimeRegistries]:
    """Create configured Agent and SandboxContext.

    Returns:
        Tuple of (Agent, SandboxContext, RuntimeRegistries)
    """
    from bu_agent_sdk.agent.subagent_manager import SubagentManager

    ctx = SandboxContext.create(root_dir)
    llm = create_llm(model)
    runtime = runtime_registries or create_runtime_registries(
        workspace_root=ctx.working_dir,
    )

    system_prompt = _build_system_prompt(
        ctx.working_dir,
        skill_registry=runtime.skill_registry,
        agent_registry=runtime.agent_registry,
    )

    subagent_manager = SubagentManager(
        agent_factory=_create_subagent_factory,
        registry=runtime.agent_registry,
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
        hooks=build_agent_hooks(mode=mode),
    )

    if subagent_manager:
        subagent_manager.set_main_agent(agent)

    return agent, ctx, runtime


def _create_subagent_factory(config: AgentConfig, parent_ctx: Any, all_tools: list) -> Agent:
    """Factory function to create subagent instances."""
    from config.model_config import get_model_config

    model, base_url, api_key = get_model_config(config.model)

    llm = ChatOpenAI(
        model=model, api_key=api_key, base_url=base_url, temperature=config.temperature
    )

    # 为子代理添加系统信息到系统提示词
    system_info_text = _get_system_info()
    system_prompt = f"{config.system_prompt}\n\n## System Information\n\nThe current environment: {system_info_text}"

    agent = Agent(
        llm=llm,
        tools=all_tools,
        system_prompt=system_prompt,
        mode="subagent",
        agent_config=config,
        dependency_overrides={get_sandbox_context: lambda: parent_ctx},
        hooks=build_agent_hooks(mode="subagent"),
    )
    return agent


async def main():
    """Main entry point."""
    args = parse_args()

    agent, ctx, runtime = create_agent(
        model=args.model,
        root_dir=args.root_dir,
    )
    cli = ClaudeCodeCLI(
        agent=agent,
        context=ctx,
        slash_registry=runtime.slash_registry,
        at_registry=runtime.skill_registry,
        agent_registry=runtime.agent_registry,
        plugin_manager=runtime.plugin_manager,
        system_prompt_builder=lambda: _build_system_prompt(
            ctx.working_dir,
            skill_registry=runtime.skill_registry,
            agent_registry=runtime.agent_registry,
        ),
    )

    try:
        await cli.run()
    except KeyboardInterrupt:
        print("\n[yellow]Goodbye![/yellow]")


def cli_main():
    """Console script entry point."""
    asyncio.run(main())


if __name__ == "__main__":
    cli_main()
