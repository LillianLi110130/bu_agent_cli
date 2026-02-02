"""Bash command execution tool."""

import subprocess
from bu_agent_sdk.tools import Depends, tool
from typing import Annotated

from tools.sandbox import SandboxContext, get_sandbox_context


@tool("Execute a shell command and return output")
async def bash(
    command: str,
    ctx: Annotated[SandboxContext, Depends(get_sandbox_context)],
    timeout: int = 30,
) -> str:
    """Run a bash command in the sandbox working directory."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(ctx.working_dir),
        )
        output = result.stdout + result.stderr
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout}s"
    except Exception as e:
        return f"Error: {e}"
