import os
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, model_validator

from tg_mem.embeddings.configs import EmbedderConfig
from tg_mem.graphs.configs import GraphStoreConfig
from tg_mem.llms.configs import LlmConfig
from tg_mem.configs.mysql import MysqlConfig
from tg_mem.vector_stores.configs import VectorStoreConfig
from tg_mem.configs.rerankers.config import RerankerConfig

# Set up the directory path
home_dir = os.path.expanduser("~")
mem0_dir = os.environ.get("MEM0_DIR") or os.path.join(home_dir, ".mem0")


class MemoryItem(BaseModel):
    id: str = Field(..., description="The unique identifier for the text data")
    memory: str = Field(
        ..., description="The memory deduced from the text data"
    )  # TODO After prompt changes from platform, update this
    hash: Optional[str] = Field(None, description="The hash of the memory")
    # The metadata value can be anything and not just string. Fix it
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata for the text data")
    score: Optional[float] = Field(None, description="The score associated with the text data")
    created_at: Optional[str] = Field(None, description="The timestamp when the memory was created")
    updated_at: Optional[str] = Field(None, description="The timestamp when the memory was updated")


class MemoryConfig(BaseModel):
    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_mysql_fields(cls, values):
        if not isinstance(values, dict):
            return values

        if "history_db_path" in values:
            raise ValueError("'history_db_path' has been removed. Please use 'mysql.db_uri'.")

        removed_flat_keys = (
            "mysql_host",
            "mysql_port",
            "mysql_user",
            "mysql_password",
            "mysql_database",
            "mysql_charset",
        )
        provided_flat_keys = [key for key in removed_flat_keys if key in values]
        if provided_flat_keys:
            raise ValueError(
                "Legacy MySQL flat fields have been removed "
                f"({', '.join(provided_flat_keys)}). Please use 'mysql.db_uri'."
            )

        mysql_values = values.get("mysql")
        if isinstance(mysql_values, BaseModel):
            mysql_values = mysql_values.model_dump(exclude_unset=True)
        if mysql_values is None:
            mysql_values = {}

        if isinstance(mysql_values, dict):
            removed_nested_keys = ("host", "port", "user", "password", "database", "charset")
            provided_nested_keys = [key for key in removed_nested_keys if key in mysql_values]
            if provided_nested_keys:
                raise ValueError(
                    "MySQL nested connection fields have been removed "
                    f"({', '.join(provided_nested_keys)}). Please use 'mysql.db_uri'."
                )

            history_db_uri = values.get("history_db_uri")
            if history_db_uri and not mysql_values.get("db_uri"):
                mysql_values["db_uri"] = history_db_uri

            if mysql_values:
                values["mysql"] = mysql_values

        return values

    vector_store: VectorStoreConfig = Field(
        description="Configuration for the vector store",
        default_factory=VectorStoreConfig,
    )
    llm: LlmConfig = Field(
        description="Configuration for the language model",
        default_factory=LlmConfig,
    )
    embedder: EmbedderConfig = Field(
        description="Configuration for the embedding model",
        default_factory=EmbedderConfig,
    )
    history_db_uri: Optional[str] = Field(
        description="Deprecated alias for mysql.db_uri",
        default=None,
    )
    mysql: MysqlConfig = Field(
        description="Configuration for MySQL history storage",
        default_factory=MysqlConfig,
    )

    graph_store: GraphStoreConfig = Field(
        description="Configuration for the graph",
        default_factory=GraphStoreConfig,
    )
    reranker: Optional[RerankerConfig] = Field(
        description="Configuration for the reranker",
        default=None,
    )
    version: str = Field(
        description="The version of the API",
        default="v1.1",
    )
    custom_fact_extraction_prompt: Optional[str] = Field(
        description="Custom prompt for the fact extraction",
        default=None,
    )
    custom_update_memory_prompt: Optional[str] = Field(
        description="Custom prompt for the update memory",
        default=None,
    )
    prompt_service_url: Optional[str] = Field(
        description="HTTP endpoint used to fetch user-specific extraction/update prompts",
        default=None,
    )
    prompt_service_timeout: float = Field(
        description="Prompt service request timeout in seconds",
        default=3.0,
    )


class AzureConfig(BaseModel):
    """
    Configuration settings for Azure.

    Args:
        api_key (str): The API key used for authenticating with the Azure service.
        azure_deployment (str): The name of the Azure deployment.
        azure_endpoint (str): The endpoint URL for the Azure service.
        api_version (str): The version of the Azure API being used.
        default_headers (Dict[str, str]): Headers to include in requests to the Azure API.
    """

    api_key: str = Field(
        description="The API key used for authenticating with the Azure service.",
        default=None,
    )
    azure_deployment: str = Field(description="The name of the Azure deployment.", default=None)
    azure_endpoint: str = Field(description="The endpoint URL for the Azure service.", default=None)
    api_version: str = Field(description="The version of the Azure API being used.", default=None)
    default_headers: Optional[Dict[str, str]] = Field(
        description="Headers to include in requests to the Azure API.", default=None
    )
