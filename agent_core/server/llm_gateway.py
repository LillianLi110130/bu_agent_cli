from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from agent_core.llm.openai.chat import ChatOpenAI
from agent_core.server.models import (
    LLMQueryRequest,
    LLMDoneEvent,
    LLMUsageEvent,
    TextEvent,
    ThinkingEvent,
    ToolCallEvent,
)
from agent_core.server.route_config import GatewayRoute, load_gateway_routes


def _parse_tool_arguments(arguments: str) -> dict[str, Any]:
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return {"_raw": arguments}
    if isinstance(parsed, dict):
        return parsed
    return {"_raw": arguments}


@dataclass
class LLMGatewayService:
    """Provider-facing LLM adapter that normalizes server-side stream events."""

    default_base_url: str | None = None
    default_api_key: str | None = None
    default_model: str | None = None
    routes: dict[str, GatewayRoute] | None = None

    def __post_init__(self) -> None:
        if self.routes is None:
            self.routes = load_gateway_routes()

    def _resolve_route(self, requested_model: str) -> GatewayRoute | None:
        if self.routes is None:
            return None
        return self.routes.get(requested_model)

    def _resolve_upstream_model(self, requested_model: str, route: GatewayRoute | None) -> str:
        if route is not None:
            return route.upstream_model
        return requested_model or self.default_model or os.getenv("LLM_MODEL", "GLM-4.7")

    def _resolve_upstream_base_url(self, route: GatewayRoute | None) -> str | None:
        if route is not None and route.base_url:
            return route.base_url
        return self.default_base_url or os.getenv("LLM_BASE_URL")

    def _resolve_upstream_api_key(self, route: GatewayRoute | None) -> str | None:
        if route is not None:
            return (
                os.getenv(route.api_key_env)
                or self.default_api_key
                or os.getenv("OPENAI_API_KEY")
            )
        return self.default_api_key or os.getenv("OPENAI_API_KEY")

    def _build_llm(self, requested_model: str) -> ChatOpenAI:
        route = self._resolve_route(requested_model)
        if self.routes and route is None:
            known_aliases = ", ".join(sorted(self.routes.keys()))
            raise ValueError(
                f"Unknown gateway model alias '{requested_model}'. "
                f"Configured aliases: {known_aliases or '<none>'}"
            )
        provider = route.provider if route is not None else "openai"
        if provider != "openai":
            raise ValueError(f"Unsupported gateway provider route: {provider}")
        return ChatOpenAI(
            model=self._resolve_upstream_model(requested_model, route),
            api_key=self._resolve_upstream_api_key(route),
            base_url=self._resolve_upstream_base_url(route),
            max_input_tokens=route.max_input_tokens if route is not None else None,
            max_completion_tokens=route.max_output_tokens if route is not None else None,
        )

    async def query_stream(self, request: LLMQueryRequest) -> AsyncIterator[object]:
        llm = self._build_llm(request.model)
        stop_reason: str | None = None
        try:
            async for chunk in llm.astream(
                messages=request.messages,
                tools=request.tools,
                tool_choice=request.tool_choice,
                metadata=request.metadata,
            ):
                if chunk.delta:
                    yield TextEvent(content=chunk.delta)
                if chunk.thinking:
                    yield ThinkingEvent(content=chunk.thinking)
                if chunk.tool_calls:
                    for tool_call in chunk.tool_calls:
                        yield ToolCallEvent(
                            tool=tool_call.function.name,
                            args=_parse_tool_arguments(tool_call.function.arguments),
                            tool_call_id=tool_call.id,
                            display_name=tool_call.function.name,
                        )
                if chunk.usage is not None:
                    yield LLMUsageEvent(usage=chunk.usage)
                if chunk.stop_reason is not None:
                    stop_reason = chunk.stop_reason
            yield LLMDoneEvent(stop_reason=stop_reason)
        finally:
            await llm.close()
