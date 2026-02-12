from datetime import datetime

from pydantic import BaseModel, Field

from bu_agent_sdk.llm.views import ChatInvokeUsage


class TokenUsageEntry(BaseModel):
    """Single token usage entry"""

    model: str
    timestamp: datetime
    usage: ChatInvokeUsage


class ModelPricing(BaseModel):
    """Pricing information for a model"""

    model: str
    input_cost_per_token: float | None
    output_cost_per_token: float | None

    cache_read_input_token_cost: float | None
    cache_creation_input_token_cost: float | None

    max_tokens: int | None
    max_input_tokens: int | None
    max_output_tokens: int | None


class ModelUsageStats(BaseModel):
    """Usage statistics for a single model"""

    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    invocations: int = 0
    average_tokens_per_invocation: float = 0.0


class ModelUsageTokens(BaseModel):
    """Usage tokens for a single model"""

    model: str
    prompt_tokens: int
    prompt_cached_tokens: int
    completion_tokens: int
    total_tokens: int


class UsageSummary(BaseModel):
    """Summary of token usage"""

    total_prompt_tokens: int
    total_prompt_cached_tokens: int
    total_completion_tokens: int
    total_tokens: int
    entry_count: int

    by_model: dict[str, ModelUsageStats] = Field(default_factory=dict)
