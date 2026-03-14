"""Session bootstrap helpers shared by CLI and gateway runtimes."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from bu_agent_sdk.llm.messages import SystemMessage, UserMessage

if TYPE_CHECKING:
    from bu_agent_sdk import Agent


@dataclass
class WorkspaceInstructionState:
    """Tracks the currently injected workspace instruction content."""

    content_hash: str | None = None
    injected_content: str | None = None


def sync_workspace_agents_md(
    agent: "Agent",
    workspace_dir: Path,
    state: WorkspaceInstructionState | None = None,
) -> WorkspaceInstructionState:
    """Synchronize workspace AGENTS.md into pinned developer context."""
    resolved_state = state or WorkspaceInstructionState()
    agents_path = workspace_dir / "AGENTS.md"

    if not agents_path.exists():
        _clear_injected_content(agent, resolved_state)
        return resolved_state

    content = agents_path.read_text(encoding="utf-8").strip()
    if not content:
        _clear_injected_content(agent, resolved_state)
        return resolved_state

    if not agent._context and agent.system_prompt:
        agent._context.add_message(SystemMessage(content=agent.system_prompt))

    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    if (
        resolved_state.content_hash == content_hash
        and resolved_state.injected_content
        and _has_injected_content(agent, resolved_state.injected_content)
    ):
        return resolved_state

    if resolved_state.injected_content:
        _remove_injected_content(agent, resolved_state.injected_content)

    agent._context.inject_message(UserMessage(content=content), pinned=True)
    resolved_state.content_hash = content_hash
    resolved_state.injected_content = content
    return resolved_state


def _has_injected_content(agent: "Agent", content: str) -> bool:
    return any(
        message.role == "developer" and getattr(message, "content", "") == content
        for message in agent._context.get_messages()
    )


def _clear_injected_content(agent: "Agent", state: WorkspaceInstructionState) -> None:
    if state.injected_content:
        _remove_injected_content(agent, state.injected_content)
    state.content_hash = None
    state.injected_content = None


def _remove_injected_content(agent: "Agent", content: str) -> None:
    messages = agent._context.get_messages()
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.role == "developer" and getattr(message, "content", "") == content:
            agent._context.remove_message_at(index)
