from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from agent_core.llm.base import BaseChatModel, ToolChoice, ToolDefinition
from agent_core.llm.exceptions import ModelProviderError
from agent_core.llm.messages import BaseMessage, Function, ToolCall
from agent_core.llm.views import ChatInvokeCompletion, ChatInvokeCompletionChunk, ChatInvokeUsage
from cli.worker.auth import load_persisted_auth_result, persist_updated_authorization


@dataclass
class ChatGateway(BaseChatModel):
    """Gateway-backed chat model that keeps local runtime semantics intact."""

    model: str
    api_key: str | None = None
    base_url: str | None = None
    timeout: float | httpx.Timeout | None = 300.0
    route_path: str = "/llm/query-stream"
    default_headers: Mapping[str, str] | None = None
    http_client: httpx.AsyncClient | None = None
    base_dir: str | Path | None = None
    stream_line_log_file: str | Path | None = "llm.log"
    _client: httpx.AsyncClient | None = field(default=None, init=False, repr=False)
    _owns_client: bool = field(default=False, init=False, repr=False)
    _authorization: str | None = field(default=None, init=False, repr=False)

    @property
    def provider(self) -> str:
        return "gateway"

    @property
    def name(self) -> str:
        return self.model

    def _get_client(self) -> httpx.AsyncClient:
        if self.http_client is not None:
            return self.http_client
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
            self._owns_client = True
        return self._client

    async def close(self) -> None:
        client = self._client
        self._client = None
        if client is None or not self._owns_client:
            self._owns_client = False
            return
        self._owns_client = False
        await client.aclose()

    @staticmethod
    def _normalize_authorization_value(value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            return None
        lowered = normalized.lower()
        if lowered.startswith("bearer ") or lowered.startswith("basic "):
            return normalized
        return f"Bearer {normalized}"

    def _resolve_authorization(self) -> str | None:
        if self._authorization:
            return self._authorization

        default_authorization = self._normalize_authorization_value(
            dict(self.default_headers or {}).get("Authorization")
        )
        if default_authorization:
            self._authorization = default_authorization
            return default_authorization

        try:
            persisted = load_persisted_auth_result(self.base_dir)
        except Exception:
            persisted = None
        if persisted is not None:
            self._authorization = persisted.authorization
            return persisted.authorization

        fallback_authorization = self._normalize_authorization_value(self.api_key)
        self._authorization = fallback_authorization
        return fallback_authorization

    def _build_headers(self, authorization: str | None = None) -> dict[str, str]:
        headers = dict(self.default_headers or {})
        resolved_authorization = authorization or self._resolve_authorization()
        if resolved_authorization:
            headers["Authorization"] = resolved_authorization
        headers.setdefault("Accept", "text/event-stream")
        return headers

    @staticmethod
    async def _read_stream_error_message(response: httpx.Response) -> str:
        body = await response.aread()
        message = body.decode("utf-8", errors="replace").strip()
        if message:
            return message
        return f"Gateway request failed with HTTP {response.status_code}"

    def _refresh_authorization_from_response(
        self,
        response: httpx.Response,
        *,
        request_authorization: str | None,
    ) -> bool:
        response_authorization = self._normalize_authorization_value(
            response.headers.get("Authorization")
        )
        if not response_authorization:
            return False
        if response_authorization == self._authorization:
            return False
        if self._authorization is not None and self._authorization != request_authorization:
            return False

        self._authorization = response_authorization
        persist_updated_authorization(
            base_dir=self.base_dir,
            authorization=response_authorization,
        )
        return True

    def _build_payload(
        self,
        messages: list[BaseMessage],
        tools: list[ToolDefinition] | None,
        tool_choice: ToolChoice | None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [message.model_dump(mode="json") for message in messages],
            "tools": [tool.model_dump(mode="json") for tool in tools] if tools else None,
            "tool_choice": tool_choice,
            "metadata": kwargs.get("metadata"),
        }
        return payload

    @staticmethod
    def _parse_tool_call(event_data: dict[str, Any]) -> ToolCall:
        args = event_data.get("args")
        if isinstance(args, dict):
            arguments = json.dumps(args, ensure_ascii=False)
        elif isinstance(args, str):
            arguments = args
        else:
            arguments = "{}"
        return ToolCall(
            id=str(event_data.get("tool_call_id", "")),
            function=Function(
                name=str(event_data.get("tool", "")),
                arguments=arguments,
            ),
            type="function",
        )

    async def astream(
        self,
        messages: list[BaseMessage],
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatInvokeCompletionChunk]:
        if not self.base_url:
            raise ModelProviderError(
                message="Gateway base_url is required for ChatGateway",
                model=self.name,
            )

        client = self._get_client()
        payload = self._build_payload(messages, tools, tool_choice, **kwargs)
        url = f"{self.base_url.rstrip('/')}{self.route_path}"

        try:
            request_authorization = self._resolve_authorization()
            for attempt in range(2):
                async with client.stream(
                    "POST",
                    url,
                    json=payload,
                    headers=self._build_headers(
                        request_authorization if attempt == 0 else None
                    ),
                    timeout=None,
                ) as response:
                    authorization_changed = self._refresh_authorization_from_response(
                        response,
                        request_authorization=(
                            request_authorization if attempt == 0 else self._authorization
                        ),
                    )
                    if response.is_error:
                        if attempt == 0 and authorization_changed:
                            request_authorization = self._authorization
                            continue
                        raise ModelProviderError(
                            message=await self._read_stream_error_message(response),
                            status_code=response.status_code,
                            model=self.name,
                        )

                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        if line.strip() == ": done":
                            break
                        if not line.startswith("data: "):
                            continue

                        event_data = json.loads(line[6:])
                        event_type = event_data.get("type")

                        if event_type == "text":
                            yield ChatInvokeCompletionChunk(
                                delta=str(event_data.get("content", ""))
                            )
                        elif event_type == "thinking":
                            yield ChatInvokeCompletionChunk(
                                thinking=str(event_data.get("content", ""))
                            )
                        elif event_type == "tool_call":
                            yield ChatInvokeCompletionChunk(
                                tool_calls=[self._parse_tool_call(event_data)]
                            )
                        elif event_type == "usage":
                            usage_payload = event_data.get("usage") or {}
                            yield ChatInvokeCompletionChunk(
                                usage=ChatInvokeUsage(**usage_payload)
                            )
                        elif event_type == "done":
                            yield ChatInvokeCompletionChunk(
                                stop_reason=event_data.get("stop_reason")
                            )
                        elif event_type == "error":
                            raise ModelProviderError(
                                message=str(event_data.get("error", "Gateway request failed")),
                                model=self.name,
                            )
                    return
        except httpx.HTTPStatusError as exc:
            raise ModelProviderError(
                message=exc.response.text or str(exc),
                status_code=exc.response.status_code,
                model=self.name,
            ) from exc
        except httpx.HTTPError as exc:
            raise ModelProviderError(message=str(exc), model=self.name) from exc
        except json.JSONDecodeError as exc:
            raise ModelProviderError(
                message=f"Invalid gateway stream payload: {exc}",
                model=self.name,
            ) from exc

    async def ainvoke_streaming(
        self,
        messages: list[BaseMessage],
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
        **kwargs: Any,
    ) -> ChatInvokeCompletion:
        content_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        usage: ChatInvokeUsage | None = None
        stop_reason: str | None = None
        thinking_parts: list[str] = []

        async for chunk in self.astream(messages, tools, tool_choice, **kwargs):
            if chunk.delta:
                content_parts.append(chunk.delta)
            if chunk.tool_calls:
                tool_calls.extend(chunk.tool_calls)
            if chunk.usage is not None:
                usage = chunk.usage
            if chunk.stop_reason is not None:
                stop_reason = chunk.stop_reason
            if chunk.thinking:
                thinking_parts.append(chunk.thinking)

        return ChatInvokeCompletion(
            content="".join(content_parts) if content_parts else None,
            tool_calls=tool_calls,
            thinking="".join(thinking_parts) if thinking_parts else None,
            usage=usage,
            stop_reason=stop_reason,
        )

    async def ainvoke(
        self,
        messages: list[BaseMessage],
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
        **kwargs: Any,
    ) -> ChatInvokeCompletion:
        return await self.ainvoke_streaming(messages, tools, tool_choice, **kwargs)
