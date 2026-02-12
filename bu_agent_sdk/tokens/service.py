"""Token usage service with optional model limits from custom pricing."""

from datetime import datetime

from bu_agent_sdk.llm.views import ChatInvokeUsage
from bu_agent_sdk.tokens.custom_pricing import CUSTOM_MODEL_PRICING
from bu_agent_sdk.tokens.views import (
    ModelPricing,
    ModelUsageStats,
    ModelUsageTokens,
    TokenUsageEntry,
    UsageSummary,
)

class TokenCost:
    """Service for tracking token usage."""

    def __init__(self):
        self.usage_history: list[TokenUsageEntry] = []
        self._initialized = True

    async def get_model_pricing(self, model_name: str) -> ModelPricing | None:
        """Get pricing information for a specific model"""
        # Check custom pricing first
        if model_name in CUSTOM_MODEL_PRICING:
            data = CUSTOM_MODEL_PRICING[model_name]
            return ModelPricing(
                model=model_name,
                input_cost_per_token=data.get("input_cost_per_token"),
                output_cost_per_token=data.get("output_cost_per_token"),
                max_tokens=data.get("max_tokens"),
                max_input_tokens=data.get("max_input_tokens"),
                max_output_tokens=data.get("max_output_tokens"),
                cache_read_input_token_cost=data.get("cache_read_input_token_cost"),
                cache_creation_input_token_cost=data.get(
                    "cache_creation_input_token_cost"
                ),
            )

        return None

    def add_usage(self, model: str, usage: ChatInvokeUsage) -> TokenUsageEntry:
        """Add token usage entry to history (without calculating cost)"""
        entry = TokenUsageEntry(
            model=model,
            timestamp=datetime.now(),
            usage=usage,
        )

        self.usage_history.append(entry)

        return entry

    def get_usage_tokens_for_model(self, model: str) -> ModelUsageTokens:
        """Get usage tokens for a specific model"""
        filtered_usage = [u for u in self.usage_history if u.model == model]

        return ModelUsageTokens(
            model=model,
            prompt_tokens=sum(u.usage.prompt_tokens for u in filtered_usage),
            prompt_cached_tokens=sum(
                u.usage.prompt_cached_tokens or 0 for u in filtered_usage
            ),
            completion_tokens=sum(u.usage.completion_tokens for u in filtered_usage),
            total_tokens=sum(
                u.usage.prompt_tokens + u.usage.completion_tokens
                for u in filtered_usage
            ),
        )

    async def get_usage_summary(
        self, model: str | None = None, since: datetime | None = None
    ) -> UsageSummary:
        """Get summary of token usage."""
        filtered_usage = self.usage_history

        if model:
            filtered_usage = [u for u in filtered_usage if u.model == model]

        if since:
            filtered_usage = [u for u in filtered_usage if u.timestamp >= since]

        if not filtered_usage:
            return UsageSummary(
                total_prompt_tokens=0,
                total_prompt_cached_tokens=0,
                total_completion_tokens=0,
                total_tokens=0,
                entry_count=0,
            )

        # Calculate totals
        total_prompt = sum(u.usage.prompt_tokens for u in filtered_usage)
        total_completion = sum(u.usage.completion_tokens for u in filtered_usage)
        total_tokens = total_prompt + total_completion
        total_prompt_cached = sum(
            u.usage.prompt_cached_tokens or 0 for u in filtered_usage
        )

        # Calculate per-model stats
        model_stats: dict[str, ModelUsageStats] = {}

        for entry in filtered_usage:
            if entry.model not in model_stats:
                model_stats[entry.model] = ModelUsageStats(model=entry.model)

            stats = model_stats[entry.model]
            stats.prompt_tokens += entry.usage.prompt_tokens
            stats.completion_tokens += entry.usage.completion_tokens
            stats.total_tokens += (
                entry.usage.prompt_tokens + entry.usage.completion_tokens
            )
            stats.invocations += 1

            # Cost calculation is disabled (no pricing data used)

        # Calculate averages
        for stats in model_stats.values():
            if stats.invocations > 0:
                stats.average_tokens_per_invocation = (
                    stats.total_tokens / stats.invocations
                )

        return UsageSummary(
            total_prompt_tokens=total_prompt,
            total_prompt_cached_tokens=total_prompt_cached,
            total_completion_tokens=total_completion,
            total_tokens=total_tokens,
            entry_count=len(filtered_usage),
            by_model=model_stats,
        )

    def _format_tokens(self, tokens: int) -> str:
        """Format token count with k suffix for thousands"""
        if tokens >= 1000000000:
            return f"{tokens / 1000000000:.1f}B"
        if tokens >= 1000000:
            return f"{tokens / 1000000:.1f}M"
        if tokens >= 1000:
            return f"{tokens / 1000:.1f}k"
        return str(tokens)

    async def get_cost_by_model(self) -> dict[str, ModelUsageStats]:
        """Get cost breakdown by model"""
        summary = await self.get_usage_summary()
        return summary.by_model

    def clear_history(self) -> None:
        """Clear usage history"""
        self.usage_history = []
