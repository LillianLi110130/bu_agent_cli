"""Unified context budget estimation for CLI runtime flows."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import TYPE_CHECKING, Sequence

from agent_core.llm.messages import BaseMessage
from config.model_config import get_model_limits

if TYPE_CHECKING:
    from agent_core.agent.compaction.models import CompactionConfig
    from agent_core.llm.views import ChatInvokeUsage
    from agent_core.tokens import TokenCost

log = logging.getLogger(__name__)

DEFAULT_CONTEXT_WINDOW = 128_000
DEFAULT_WARN_THRESHOLD = 0.65
DEFAULT_HARD_THRESHOLD = 0.92


@dataclass(slots=True)
class BudgetAssessment:
    """Snapshot of the current context budget for one model."""

    model: str
    context_limit: int
    warn_threshold: int
    compact_threshold: int
    hard_threshold: int
    baseline_prompt_tokens: int
    incremental_tokens: int
    estimated_tokens: int
    message_count: int
    warn_threshold_ratio: float
    compact_threshold_ratio: float
    hard_threshold_ratio: float
    threshold_utilization: float
    context_utilization: float
    trigger: str | None = None

    @property
    def needs_warning(self) -> bool:
        return self.estimated_tokens >= self.warn_threshold

    @property
    def needs_compaction(self) -> bool:
        return self.estimated_tokens >= self.compact_threshold

    @property
    def exceeds_hard_limit(self) -> bool:
        return self.estimated_tokens >= self.hard_threshold


@dataclass(slots=True)
class ContextBudgetEngine:
    """Estimate current prompt budget from real usage plus local increments."""

    config: "CompactionConfig"
    token_cost: "TokenCost | None" = None
    baseline_prompt_tokens: int = 0
    baseline_message_count: int = 0
    baseline_model: str | None = None
    last_trigger: str | None = None
    last_assessment: BudgetAssessment | None = None
    _context_limit_cache: dict[str, int] = field(default_factory=dict, repr=False)

    @property
    def warn_threshold_ratio(self) -> float:
        return float(getattr(self.config, "warn_threshold", DEFAULT_WARN_THRESHOLD))

    @property
    def compact_threshold_ratio(self) -> float:
        configured = getattr(self.config, "compact_threshold", None)
        if configured is not None:
            return float(configured)
        return float(getattr(self.config, "threshold_ratio", 0.80))

    @property
    def hard_threshold_ratio(self) -> float:
        return float(getattr(self.config, "hard_threshold", DEFAULT_HARD_THRESHOLD))

    async def get_model_context_limit(self, model: str) -> int:
        """Get the context window for a model with lightweight caching."""
        cached = self._context_limit_cache.get(model)
        if cached is not None:
            return cached

        context_limit = DEFAULT_CONTEXT_WINDOW
        preset_max_input_tokens, _ = get_model_limits(model)
        if preset_max_input_tokens is not None:
            context_limit = preset_max_input_tokens

        if context_limit == DEFAULT_CONTEXT_WINDOW and self.token_cost is not None:
            try:
                pricing = await self.token_cost.get_model_pricing(model)
            except Exception as exc:
                log.debug("Failed to fetch model pricing for %s: %s", model, exc)
                pricing = None

            if pricing is not None:
                if pricing.max_input_tokens:
                    context_limit = pricing.max_input_tokens
                elif pricing.max_tokens:
                    context_limit = pricing.max_tokens

        self._context_limit_cache[model] = context_limit
        return context_limit

    def reset(self) -> None:
        """Drop prompt baselines after history rewrites."""
        self.baseline_prompt_tokens = 0
        self.baseline_message_count = 0
        self.baseline_model = None
        self.last_trigger = None
        self.last_assessment = None

    def note_trigger(self, trigger: str | None) -> None:
        """Store the latest trigger reason for observability/debugging."""
        self.last_trigger = trigger
        if self.last_assessment is not None:
            self.last_assessment.trigger = trigger

    def record_usage(
        self,
        *,
        model: str,
        messages: Sequence[BaseMessage],
        usage: "ChatInvokeUsage | None",
    ) -> None:
        """Use the last real prompt usage as the next estimation baseline."""
        if usage is None:
            return

        self.baseline_prompt_tokens = max(0, int(usage.prompt_tokens))
        self.baseline_message_count = len(messages)
        self.baseline_model = model

    def estimate_tokens_for_messages(self, messages: Sequence[BaseMessage]) -> int:
        """Estimate token usage for a message list via serialized payload size."""
        if not messages:
            return 0

        total = 0
        for message in messages:
            try:
                serialized = message.model_dump_json()
            except Exception:
                serialized = str(message)
            total += max(1, len(serialized) // 4) + 4
        return total

    def estimate_current_tokens(
        self,
        *,
        model: str,
        messages: Sequence[BaseMessage],
    ) -> tuple[int, int]:
        """Estimate current prompt tokens from the latest real baseline."""
        baseline_is_usable = (
            self.baseline_message_count > 0
            and self.baseline_message_count <= len(messages)
        )
        if not baseline_is_usable:
            total_estimate = self.estimate_tokens_for_messages(messages)
            return total_estimate, total_estimate

        incremental_messages = messages[self.baseline_message_count :]
        incremental_tokens = self.estimate_tokens_for_messages(incremental_messages)
        estimated_total = self.baseline_prompt_tokens + incremental_tokens
        return estimated_total, incremental_tokens

    async def assess(
        self,
        *,
        model: str,
        messages: Sequence[BaseMessage],
        usage: "ChatInvokeUsage | None" = None,
        trigger: str | None = None,
    ) -> BudgetAssessment:
        """Build a full budget assessment for the current context."""
        if usage is not None:
            self.record_usage(model=model, messages=messages, usage=usage)

        context_limit = await self.get_model_context_limit(model)
        warn_threshold = int(context_limit * self.warn_threshold_ratio)
        compact_threshold = int(context_limit * self.compact_threshold_ratio)
        hard_threshold = int(context_limit * self.hard_threshold_ratio)
        estimated_tokens, incremental_tokens = self.estimate_current_tokens(
            model=model,
            messages=messages,
        )

        assessment = BudgetAssessment(
            model=model,
            context_limit=context_limit,
            warn_threshold=warn_threshold,
            compact_threshold=compact_threshold,
            hard_threshold=hard_threshold,
            baseline_prompt_tokens=self.baseline_prompt_tokens
            if self.baseline_message_count > 0
            else 0,
            incremental_tokens=incremental_tokens,
            estimated_tokens=estimated_tokens,
            message_count=len(messages),
            warn_threshold_ratio=self.warn_threshold_ratio,
            compact_threshold_ratio=self.compact_threshold_ratio,
            hard_threshold_ratio=self.hard_threshold_ratio,
            threshold_utilization=estimated_tokens / max(1, compact_threshold),
            context_utilization=estimated_tokens / max(1, context_limit),
            trigger=trigger or self.last_trigger,
        )
        self.last_assessment = assessment
        if trigger is not None:
            self.last_trigger = trigger
        return assessment
