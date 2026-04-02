"""
Compaction subservice for managing conversation context.

Automatically summarizes and compresses conversation history when token usage
approaches model's context window
"""

from agent_core.agent.compaction.models import (
    CompactionConfig,
    CompactionResult,
    CompactionWorkingState,
    TokenUsage,
)
from agent_core.agent.compaction.service import CompactionService

__all__ = [
    "CompactionConfig",
    "CompactionResult",
    "CompactionWorkingState",
    "CompactionService",
    "TokenUsage",
]
