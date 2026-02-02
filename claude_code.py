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

from bu_agent_sdk import Agent
from bu_agent_sdk.llm import ChatOpenAI

from cli.app import ClaudeCodeCLI
from tools import ALL_TOOLS, SandboxContext, get_sandbox_context


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
    api_key = os.getenv("OPENAI_API_KEY", "your_key")

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
    )


def create_agent(
    model: str | None, root_dir: Path | str | None = None
) -> tuple[Agent, SandboxContext]:
    """Create configured Agent and SandboxContext.

    Returns:
        Tuple of (Agent, SandboxContext)
    """
    ctx = SandboxContext.create(root_dir)
    llm = create_llm(model)

    agent = Agent(
        llm=llm,
        tools=ALL_TOOLS,
        system_prompt=(
            "You are a coding assistant. You can read, write, and edit files, "
            "run shell commands, search for files and content, and manage todos. "
            f"Working directory: {ctx.working_dir}"
        ),
        dependency_overrides={get_sandbox_context: lambda: ctx},
    )
    return agent, ctx


async def main():
    """Main entry point."""
    args = parse_args()

    agent, ctx = create_agent(model=args.model, root_dir=args.root_dir)
    cli = ClaudeCodeCLI(agent=agent, context=ctx)

    try:
        await cli.run()
    except KeyboardInterrupt:
        print("\n[yellow]Goodbye![/yellow]")


if __name__ == "__main__":
    asyncio.run(main())
