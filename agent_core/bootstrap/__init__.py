"""Shared bootstrap helpers for CLI and gateway entrypoints."""

from agent_core.bootstrap.agent_factory import build_system_prompt, create_agent, create_llm
from agent_core.bootstrap.session_bootstrap import (
    WorkspaceInstructionState,
    sync_workspace_agents_md,
)

__all__ = [
    "build_system_prompt",
    "create_agent",
    "create_llm",
    "WorkspaceInstructionState",
    "sync_workspace_agents_md",
]
