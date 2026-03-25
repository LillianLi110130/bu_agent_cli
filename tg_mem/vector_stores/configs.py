from typing import Dict, Optional

from pydantic import BaseModel, Field, model_validator


class VectorStoreConfig(BaseModel):
    provider: str = Field(
        description="Provider of the vector store (elasticsearch/none)",
        default="none",
    )
    config: Optional[Dict] = Field(description="Configuration for the specific vector store", default=None)

    _provider_configs: Dict[str, Optional[str]] = {
        "elasticsearch": "ElasticsearchConfig",
        "none": None,
    }

    @model_validator(mode="after")
    def validate_and_create_config(self) -> "VectorStoreConfig":
        provider = self.provider
        config = self.config

        if provider not in self._provider_configs:
            raise ValueError(f"Unsupported vector store provider: {provider}")

        # MySQL-only mode: explicitly disable vector store.
        if provider == "none":
            self.config = None
            return self

        module = __import__(
            f"tg_mem.configs.vector_stores.{provider}",
            fromlist=[self._provider_configs[provider]],
        )
        config_class = getattr(module, self._provider_configs[provider])

        if config is None:
            config = {}

        if not isinstance(config, dict):
            if not isinstance(config, config_class):
                raise ValueError(f"Invalid config type for provider {provider}")
            return self

        self.config = config_class(**config)
        return self
