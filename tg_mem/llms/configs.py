from typing import Optional

from pydantic import BaseModel, Field, field_validator


class LlmConfig(BaseModel):
    provider: str = Field(description="Provider of the LLM (openai/openai_like)", default="openai")
    config: Optional[dict] = Field(description="Configuration for the specific LLM", default={})

    @field_validator("provider")
    def validate_provider(cls, provider: str):
        if provider in ("openai", "openai_like", "openai-like"):
            return provider
        raise ValueError(f"Unsupported LLM provider: {provider}")

    @field_validator("config")
    def validate_config(cls, v, values):
        provider = values.data.get("provider")
        if provider in ("openai", "openai_like", "openai-like"):
            return v
        raise ValueError(f"Unsupported LLM provider: {provider}")
