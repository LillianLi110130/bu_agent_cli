from __future__ import annotations

from typing import cast

from agent_core.llm.base import BaseChatModel
from agent_core.llm.gateway import ChatGateway
from agent_core.llm.openai.chat import ChatOpenAI
from config.model_config import ModelPreset, resolve_model_config


def create_chat_model(
    model_name: str | None = None,
    *,
    presets: dict[str, ModelPreset] | None = None,
    fallback_llm: BaseChatModel | None = None,
) -> BaseChatModel:
    """Create a provider-aware chat model from presets or environment variables."""
    fallback_base_url = getattr(fallback_llm, "base_url", None)
    fallback_api_key = getattr(fallback_llm, "api_key", None)
    resolved = resolve_model_config(
        model_name,
        presets=presets,
        fallback_base_url=str(fallback_base_url) if fallback_base_url else None,
        fallback_api_key=str(fallback_api_key) if fallback_api_key else None,
    )

    if resolved.provider == "gateway":
        timeout = getattr(fallback_llm, "timeout", 300.0)
        default_headers = getattr(fallback_llm, "default_headers", None)
        return ChatGateway(
            model=resolved.model,
            api_key=resolved.api_key,
            base_url=resolved.base_url,
            timeout=timeout,
            default_headers=default_headers,
        )

    if isinstance(fallback_llm, ChatOpenAI):
        return ChatOpenAI(
            model=resolved.model,
            api_key=resolved.api_key or fallback_llm.api_key,
            base_url=resolved.base_url or fallback_llm.base_url,
            temperature=fallback_llm.temperature,
            frequency_penalty=fallback_llm.frequency_penalty,
            reasoning_effort=fallback_llm.reasoning_effort,
            seed=fallback_llm.seed,
            service_tier=fallback_llm.service_tier,
            top_p=fallback_llm.top_p,
            parallel_tool_calls=fallback_llm.parallel_tool_calls,
            prompt_cache_key=fallback_llm.prompt_cache_key,
            prompt_cache_retention=fallback_llm.prompt_cache_retention,
            max_retries=fallback_llm.max_retries,
            timeout=fallback_llm.timeout,
            default_headers=fallback_llm.default_headers,
            default_query=fallback_llm.default_query,
        )

    return cast(
        BaseChatModel,
        ChatOpenAI(
            model=resolved.model,
            api_key=resolved.api_key,
            base_url=resolved.base_url,
        ),
    )
