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
from pathlib import Path
from typing import Any

from bu_agent_sdk import Agent
from bu_agent_sdk.llm import ChatOpenAI
from bu_agent_sdk.agent.config import AgentConfig

from bu_agent_sdk.skill.loader import load_skills
from bu_agent_sdk.skill.types import Skill
from cli.app import ClaudeCodeCLI
from tools import ALL_TOOLS, SandboxContext, get_sandbox_context


# =============================================================================
# Prompt & Skills Loading
# =============================================================================

# Directory paths
_SCRIPT_DIR = Path(__file__).parent.resolve()
_PROMPTS_DIR = _SCRIPT_DIR / "bu_agent_sdk" / "prompts"
_SKILLS_DIR = _SCRIPT_DIR / "bu_agent_sdk" / "skills"


def _format_skills(skills: list[Skill]) -> str:
    """Format skills list into a readable string for the prompt."""
    if not skills:
        return "No skills available."
    
    skills_formatted = "\n".join(
        (
            f"- {skill.name}\n"
            f"  - Path: {skill.path}\n"
            f"  - Desc: {skill.description}"
        )
        for skill in skills
    )
    return skills_formatted


def _load_prompt_template(template_name: str = "system.md") -> str:
    """Load a prompt template from the prompts directory."""
    template_path = _PROMPTS_DIR / template_name
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {template_path}")
    return template_path.read_text(encoding="utf-8")


def _build_system_prompt(working_dir: Path) -> str:
    """Build the system prompt by loading template and injecting skills."""
    from string import Template

    # Load skills and Format skills
    skills = load_skills(working_dir / "bu_agent_sdk" / "skills")
    skills_text = _format_skills(skills)

    # Load subagents
    from bu_agent_sdk.agent.registry import get_agent_registry
    registry = get_agent_registry()
    callable_agents = registry.list_callable_agents()

    # Format subagents
    agents_text = ""
    if callable_agents:
        agents_lines = []
        for agent_name in callable_agents:
            config = registry.get_config(agent_name)
            if config:
                agents_lines.append(f"- {agent_name}: {config.description}")
        agents_text = "\n".join(agents_lines)
    else:
        agents_text = "No subagents available."

    template_str = _load_prompt_template("system.md")

    template = Template(template_str)
    prompt = template.substitute(
        SKILLS=skills_text,
        WORKING_DIR=str(working_dir),
        SUBAGENTS=agents_text,
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


def create_agent(
    model: str | None, root_dir: Path | str | None = None,
    mode: str = "primary", agent_config: AgentConfig | None = None
) -> tuple[Agent, SandboxContext]:
    """Create configured Agent and SandboxContext.

    Returns:
        Tuple of (Agent, SandboxContext)
    """
    from bu_agent_sdk.agent.subagent_manager import SubagentManager

    ctx = SandboxContext.create(root_dir)
    llm = create_llm(model)

    system_prompt = _build_system_prompt(ctx.working_dir)

    from bu_agent_sdk.agent.registry import get_agent_registry
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

    if subagent_manager:
        subagent_manager.set_main_agent(agent)

    return agent, ctx


def _create_subagent_factory(config: AgentConfig, parent_ctx: Any, all_tools: list) -> Agent:
    """Factory function to create subagent instances."""
    from config.model_config import get_model_config
    model, base_url, api_key = get_model_config(config.model)

    llm = ChatOpenAI(model=model, api_key=api_key, base_url=base_url, temperature=config.temperature)

    agent = Agent(
        llm=llm,
        tools=all_tools,
        system_prompt=config.system_prompt,
        mode="subagent",
        agent_config=config,
        dependency_overrides={get_sandbox_context: lambda: parent_ctx},
    )
    return agent


async def main():
    """Main entry point."""
    args = parse_args()

    agent, ctx = create_agent(model=args.model, root_dir=args.root_dir)
    cli = ClaudeCodeCLI(agent=agent, context=ctx)

    try:
        await cli.run()
    except KeyboardInterrupt:
        print("\n[yellow]Goodbye![/yellow]")


def cli_main():
    """Console script entry point."""
    asyncio.run(main())


if __name__ == "__main__":
    cli_main()
