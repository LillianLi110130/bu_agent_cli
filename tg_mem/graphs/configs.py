from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from tg_mem.llms.configs import LlmConfig


class Neo4jConfig(BaseModel):
    url: Optional[str] = Field(None, description="Host address for the graph database")
    username: Optional[str] = Field(None, description="Username for the graph database")
    password: Optional[str] = Field(None, description="Password for the graph database")
    database: Optional[str] = Field(None, description="Database for the graph database")
    base_label: Optional[bool] = Field(None, description="Whether to use base node label __Entity__ for all entities")

    @model_validator(mode="before")
    def check_host_port_or_path(cls, values):
        if values is None:
            return values
        url, username, password = (
            values.get("url"),
            values.get("username"),
            values.get("password"),
        )
        if not url or not username or not password:
            raise ValueError("Please provide 'url', 'username' and 'password'.")
        return values


class GraphStoreConfig(BaseModel):
    provider: str = Field(
        description="Provider of the graph store (neo4j)",
        default="neo4j",
    )
    config: Optional[Neo4jConfig] = Field(
        description="Configuration for neo4j graph store",
        default=None,
    )
    llm: Optional[LlmConfig] = Field(description="LLM configuration for querying the graph store", default=None)
    custom_prompt: Optional[str] = Field(
        description="Custom prompt to fetch entities from the given text", default=None
    )
    threshold: float = Field(
        description="Threshold for embedding similarity when matching nodes during graph ingestion.",
        default=0.7,
        ge=0.0,
        le=1.0,
    )

    @field_validator("provider")
    def validate_provider(cls, provider: str):
        if provider != "neo4j":
            raise ValueError(f"Unsupported graph store provider: {provider}")
        return provider

    @field_validator("config")
    def validate_config(cls, v, values):
        provider = values.data.get("provider")
        if provider != "neo4j":
            raise ValueError(f"Unsupported graph store provider: {provider}")

        if v is None:
            return None

        if isinstance(v, Neo4jConfig):
            return v

        if isinstance(v, dict):
            return Neo4jConfig(**v)

        # Fallback for pydantic models with model_dump support
        if hasattr(v, "model_dump"):
            return Neo4jConfig(**v.model_dump())

        raise ValueError("Invalid graph store config")
