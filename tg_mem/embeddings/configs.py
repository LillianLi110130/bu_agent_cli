from typing import Optional

from pydantic import BaseModel, Field, field_validator


class EmbedderConfig(BaseModel):
    provider: str = Field(
        description="Provider of the embedding model (openai/huggingface/fastembed)",
        default="openai",
    )
    config: Optional[dict] = Field(description="Configuration for the specific embedding model", default={})

    @field_validator("provider")
    def validate_provider(cls, provider: str):
        if provider in ["openai", "huggingface", "fastembed"]:
            return provider
        raise ValueError(f"Unsupported embedding provider: {provider}")

    @field_validator("config")
    def validate_config(cls, v, values):
        provider = values.data.get("provider")
        if provider in ["openai", "huggingface", "fastembed"]:
            return v
        raise ValueError(f"Unsupported embedding provider: {provider}")
