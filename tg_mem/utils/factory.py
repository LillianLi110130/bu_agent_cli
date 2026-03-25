import importlib
from typing import Dict, Optional, Union

from tg_mem.configs.embeddings.base import BaseEmbedderConfig
from tg_mem.configs.llms.base import BaseLlmConfig
from tg_mem.configs.llms.openai import OpenAIConfig
from tg_mem.configs.rerankers.base import BaseRerankerConfig
from tg_mem.configs.rerankers.cohere import CohereRerankerConfig
from tg_mem.configs.rerankers.huggingface import HuggingFaceRerankerConfig
from tg_mem.configs.rerankers.llm import LLMRerankerConfig
from tg_mem.configs.rerankers.sentence_transformer import SentenceTransformerRerankerConfig
from tg_mem.configs.rerankers.zero_entropy import ZeroEntropyRerankerConfig


def load_class(class_type):
    module_path, class_name = class_type.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class LlmFactory:
    """
    Factory for creating LLM instances.
    """

    provider_to_class = {
        "openai": ("tg_mem.llms.openai.OpenAILLM", OpenAIConfig),
        "openai_like": ("tg_mem.llms.openai.OpenAILLM", OpenAIConfig),
        "openai-like": ("tg_mem.llms.openai.OpenAILLM", OpenAIConfig),
    }

    @classmethod
    def create(cls, provider_name: str, config: Optional[Union[BaseLlmConfig, Dict]] = None, **kwargs):
        if provider_name not in cls.provider_to_class:
            raise ValueError(f"Unsupported Llm provider: {provider_name}")

        class_type, config_class = cls.provider_to_class[provider_name]
        llm_class = load_class(class_type)

        if config is None:
            config = config_class(**kwargs)
        elif isinstance(config, dict):
            config.update(kwargs)
            config = config_class(**config)
        elif isinstance(config, BaseLlmConfig):
            if config_class != BaseLlmConfig:
                config_dict = {
                    "model": config.model,
                    "temperature": config.temperature,
                    "api_key": config.api_key,
                    "max_tokens": config.max_tokens,
                    "top_p": config.top_p,
                    "top_k": config.top_k,
                    "enable_vision": config.enable_vision,
                    "vision_details": config.vision_details,
                    "http_client_proxies": config.http_client,
                }
                config_dict.update(kwargs)
                config = config_class(**config_dict)
        return llm_class(config)

    @classmethod
    def register_provider(cls, name: str, class_path: str, config_class=None):
        if config_class is None:
            config_class = BaseLlmConfig
        cls.provider_to_class[name] = (class_path, config_class)

    @classmethod
    def get_supported_providers(cls) -> list:
        return list(cls.provider_to_class.keys())


class EmbedderFactory:
    provider_to_class = {
        "openai": "tg_mem.embeddings.openai.OpenAIEmbedding",
        "huggingface": "tg_mem.embeddings.huggingface.HuggingFaceEmbedding",
        "fastembed": "tg_mem.embeddings.fastembed.FastEmbedEmbedding",
    }

    @classmethod
    def create(cls, provider_name, config, _vector_config: Optional[dict]):
        class_type = cls.provider_to_class.get(provider_name)
        if class_type:
            embedder_instance = load_class(class_type)
            base_config = BaseEmbedderConfig(**config)
            return embedder_instance(base_config)
        else:
            raise ValueError(f"Unsupported Embedder provider: {provider_name}")


class VectorStoreFactory:
    provider_to_class = {
        "elasticsearch": "tg_mem.vector_stores.elasticsearch.ElasticsearchDB",
        "none": None,
    }

    @classmethod
    def create(cls, provider_name, config):
        # Explicit MySQL-only mode.
        if provider_name == "none":
            return None

        class_type = cls.provider_to_class.get(provider_name)
        if class_type:
            if not isinstance(config, dict):
                config = config.model_dump()
            vector_store_instance = load_class(class_type)
            return vector_store_instance(**config)
        else:
            raise ValueError(f"Unsupported VectorStore provider: {provider_name}")

    @classmethod
    def reset(cls, instance):
        instance.reset()
        return instance


class GraphStoreFactory:
    """
    Factory for creating MemoryGraph instances.
    """

    provider_to_class = {
        "neo4j": "tg_mem.memory.graph_memory.MemoryGraph",
    }

    @classmethod
    def create(cls, provider_name, config):
        class_type = cls.provider_to_class.get(provider_name)
        if not class_type:
            raise ValueError(f"Unsupported graph store provider: {provider_name}")

        try:
            GraphClass = load_class(class_type)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Could not import MemoryGraph for provider '{provider_name}': {e}")
        return GraphClass(config)


class RerankerFactory:
    """
    Factory for creating reranker instances with appropriate configurations.
    Supports provider-specific configs following the same pattern as other factories.
    """

    provider_to_class = {
        "cohere": ("tg_mem.reranker.cohere_reranker.CohereReranker", CohereRerankerConfig),
        "sentence_transformer": ("tg_mem.reranker.sentence_transformer_reranker.SentenceTransformerReranker", SentenceTransformerRerankerConfig),
        "zero_entropy": ("tg_mem.reranker.zero_entropy_reranker.ZeroEntropyReranker", ZeroEntropyRerankerConfig),
        "llm_reranker": ("tg_mem.reranker.llm_reranker.LLMReranker", LLMRerankerConfig),
        "huggingface": ("tg_mem.reranker.huggingface_reranker.HuggingFaceReranker", HuggingFaceRerankerConfig),
    }

    @classmethod
    def create(cls, provider_name: str, config: Optional[Union[BaseRerankerConfig, Dict]] = None, **kwargs):
        if provider_name not in cls.provider_to_class:
            raise ValueError(f"Unsupported reranker provider: {provider_name}")

        class_path, config_class = cls.provider_to_class[provider_name]

        if config is None:
            config = config_class(**kwargs)
        elif isinstance(config, dict):
            config = config_class(**config, **kwargs)
        elif not isinstance(config, BaseRerankerConfig):
            raise ValueError(f"Config must be a {config_class.__name__} instance or dict")

        try:
            reranker_class = load_class(class_path)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Could not import reranker for provider '{provider_name}': {e}")

        return reranker_class(config)
