"""Claude Code CLI entrypoint."""

import argparse
import asyncio

from rich.console import Console

from bu_agent_sdk.bootstrap.agent_factory import create_agent
from cli.app import ClaudeCodeCLI

console = Console()


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


async def main():
    """Main entry point."""
    args = parse_args()
    agent, ctx = create_agent(model=args.model, root_dir=args.root_dir)
    cli = ClaudeCodeCLI(agent=agent, context=ctx)

    try:
        await cli.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]Goodbye![/yellow]")


def cli_main():
    """Console script entry point."""
    asyncio.run(main())


if __name__ == "__main__":
    cli_main()
