"""
Models for the compaction subservice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_core.llm.views import ChatInvokeUsage

# Default ratios of context window to use for budget stages.
DEFAULT_THRESHOLD_RATIO = 0.80
DEFAULT_WARN_THRESHOLD = 0.65
DEFAULT_HARD_THRESHOLD = 0.92

DEFAULT_SUMMARY_PROMPT = """You are compacting an unfinished coding-agent conversation into a
working set that will replace older history.

Return exactly three blocks in this order and do not include markdown fences:

<summary>
A concise continuation summary for another coding agent. Focus on what the user wants,
what has already been done, key technical findings, and the next concrete steps.
</summary>

<working_state>
{
  "user_goal": "string",
  "user_constraints": ["string"],
  "confirmed_conclusions": ["string"],
  "files_reviewed": ["string"],
  "files_modified": ["string"],
  "failed_attempts": ["string"],
  "remaining_actions": ["string"],
  "artifact_refs": ["string"],
  "recent_history_notes": ["string"]
}
</working_state>

<checkpoint_ref>
compaction-inline-short-id
</checkpoint_ref>

Rules:
- The task is usually a coding task. Preserve concrete file paths, decisions, unresolved bugs,
  and pending implementation steps.
- Keep the summary compact but sufficient to continue work without repeating already-completed
  investigation.
- In working_state, output strict JSON only: double-quoted keys and strings, no markdown fences,
  no comments, no trailing commas, no unescaped newlines inside strings.
- In working_state, use empty strings or empty arrays when information is unavailable.
- Do not invent files, refs, or decisions.
- remaining_actions must contain only actions that are still unfinished, still valid, and should be
  done next. Do not keep actions that are completed, obsolete, or contradicted by
  confirmed_conclusions.
- recent_history_notes should only capture a few very recent active threads that matter if the
  preserved tail is lost.
- recent_history_notes must not preserve casual chatter, completed intermediate steps, or obsolete
  state.
- If a recent_tail_reference block is present, use it only to decide whether requests in the
  compacted segment were already answered or completed. Do not duplicate that tail in the summary.
"""


@dataclass
class CompactionConfig:
    """Configuration for the compaction service.

    The compaction service monitors token usage and automatically summarizes
    conversation history when approaching the model's context window limit.

    Attributes:
            enabled: Whether compaction is enabled. Defaults to True.
            threshold_ratio: Backward-compatible alias for the compaction threshold.
            warn_threshold: Ratio of context window at which sliding-window cleanup may start.
            compact_threshold: Ratio of context window at which compaction triggers.
            hard_threshold: Ratio of context window at which the context is in a danger zone.
            preserve_recent_messages: Maximum count of recent active messages kept outside the
                    compacted working set.
            preserve_recent_token_ratio: Maximum share of the compact threshold reserved for that
                    recent active tail.
            model: Optional model to use for generating summaries. If None, uses the agent's model.
            summary_prompt: Custom prompt for summary generation.
    """

    enabled: bool = True
    threshold_ratio: float = DEFAULT_THRESHOLD_RATIO
    warn_threshold: float = DEFAULT_WARN_THRESHOLD
    compact_threshold: float | None = None
    hard_threshold: float = DEFAULT_HARD_THRESHOLD
    preserve_recent_messages: int = 6
    preserve_recent_token_ratio: float = 0.25
    model: str | None = None
    summary_prompt: str = DEFAULT_SUMMARY_PROMPT


@dataclass
class CompactionWorkingState:
    """Structured working set extracted from a compaction response."""

    user_goal: str = ""
    user_constraints: list[str] = field(default_factory=list)
    confirmed_conclusions: list[str] = field(default_factory=list)
    files_reviewed: list[str] = field(default_factory=list)
    files_modified: list[str] = field(default_factory=list)
    failed_attempts: list[str] = field(default_factory=list)
    remaining_actions: list[str] = field(default_factory=list)
    artifact_refs: list[str] = field(default_factory=list)
    recent_history_notes: list[str] = field(default_factory=list)


@dataclass
class CompactionResult:
    """Result of a compaction operation.

    Attributes:
            compacted: Whether compaction was performed.
            original_tokens: Token count before compaction.
            new_tokens: Token count after compaction (estimated from summary output tokens).
            summary: The generated summary text (if compaction was performed).
            working_state: Structured working set extracted from the summary.
            checkpoint_ref: Logical checkpoint ref for the compacted interval.
    """

    compacted: bool
    original_tokens: int = 0
    new_tokens: int = 0
    summary: str | None = None
    working_state: CompactionWorkingState | None = None
    checkpoint_ref: str | None = None
    checkpoint_path: str | None = None


@dataclass
class TokenUsage:
    """Token usage tracking for compaction decisions.

    Attributes:
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            cache_creation_tokens: Number of tokens used to create cache (Anthropic).
            cache_read_tokens: Number of cached tokens read.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Calculate total tokens for compaction threshold check.

        This matches the Anthropic SDK's calculation:
        input_tokens + cache_creation_input_tokens + cache_read_input_tokens + output_tokens
        """
        return (
            self.input_tokens
            + self.cache_creation_tokens
            + self.cache_read_tokens
            + self.output_tokens
        )

    @classmethod
    def from_usage(cls, usage: ChatInvokeUsage | None) -> TokenUsage:
        """Create TokenUsage from ChatInvokeUsage."""
        if usage is None:
            return cls()

        return cls(
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            cache_creation_tokens=usage.prompt_cache_creation_tokens or 0,
            cache_read_tokens=usage.prompt_cached_tokens or 0,
        )
