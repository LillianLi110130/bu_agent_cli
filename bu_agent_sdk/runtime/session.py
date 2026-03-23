"""Session-scoped runtime container for a gateway-managed agent."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from bu_agent_sdk.bootstrap.session_bootstrap import WorkspaceInstructionState

if TYPE_CHECKING:
    from bu_agent_sdk import Agent
    from tools.sandbox import SandboxContext


@dataclass
class AgentRuntime:
    """Holds the mutable state for one chat session."""

    agent: "Agent"
    context: "SandboxContext"
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_used_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    workspace_instruction_state: WorkspaceInstructionState = field(
        default_factory=WorkspaceInstructionState,
        repr=False,
    )

    def touch(self) -> None:
        """Refresh the runtime's last-used timestamp."""
        self.last_used_at = datetime.now(UTC)
