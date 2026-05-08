import json
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Literal

import httpx
from openai import APIConnectionError, APIStatusError, AsyncOpenAI, RateLimitError
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from openai.types.shared.chat_model import ChatModel
from openai.types.shared.function_definition import FunctionDefinition
from openai.types.shared_params.reasoning_effort import ReasoningEffort

from agent_core.llm.base import BaseChatModel, ToolChoice, ToolDefinition
from agent_core.llm.exceptions import ModelProviderError, ModelRateLimitError
from agent_core.llm.messages import AssistantMessage, BaseMessage, Function, ToolCall, ToolMessage
from agent_core.llm.openai.serializer import OpenAIMessageSerializer
from agent_core.llm.views import ChatInvokeCompletion, ChatInvokeCompletionChunk, ChatInvokeUsage
from config.model_config import get_model_limits

logger = logging.getLogger("agent_core.llm.openai")

_TOOL_CALL_ARGUMENTS_PREVIEW_CHARS = 500
_CURL_DEBUG_STRING_PREVIEW_CHARS = 2000


@dataclass
class ChatOpenAI(BaseChatModel):
    """
    A wrapper around AsyncOpenAI that implements the BaseChatModel protocol.

    This class provides tool calling support for OpenAI models.

    Example:
        ```python
        from agent_core.llm import ChatOpenAI
        from agent_core.llm.base import ToolDefinition
        from agent_core.llm.messages import UserMessage

        llm = ChatOpenAI(model='gpt-4o', api_key='...')

        # Define tools
        tools = [ToolDefinition(name='get_weather', description='Get weather for a location', parameters={'type': 'object', 'properties': {...}})]

        # Invoke with tools
        response = await llm.ainvoke(messages=[UserMessage(content="What's the weather?")], tools=tools)

        if response.has_tool_calls:
            for tc in response.tool_calls:
                print(f'Call {tc.function.name} with {tc.function.arguments}')
        ```
    """

    # Model configuration
    model: ChatModel | str

    # Model params
    temperature: float | None = 0.2
    frequency_penalty: float | None = (
        0.3  # this avoids infinite generation of \t for models like 4.1-mini
    )
    reasoning_effort: ReasoningEffort = "low"
    seed: int | None = None
    service_tier: Literal["auto", "default", "flex", "priority", "scale"] | None = None
    top_p: float | None = None
    parallel_tool_calls: bool = True  # Allow multiple tool calls in a single response
    prompt_cache_key: str | None = "agent_core-agent"
    prompt_cache_retention: Literal["in_memory", "24h"] | None = None
    extended_cache_models: tuple[str, ...] = field(
        default_factory=lambda: (
            "gpt-5.2",
            "gpt-5.1-codex-max",
            "gpt-5.1",
            "gpt-5.1-codex",
            "gpt-5.1-codex-mini",
            "gpt-5.1-chat-latest",
            "gpt-5",
            "gpt-5-codex",
            "gpt-4.1",
        )
    )

    # Client initialization parameters
    api_key: str | None = None
    organization: str | None = None
    project: str | None = None
    base_url: str | httpx.URL | None = None
    websocket_base_url: str | httpx.URL | None = None
    timeout: float | httpx.Timeout | None = None
    max_retries: int = 5  # Increase default retries for automation reliability
    default_headers: Mapping[str, str] | None = None
    default_query: Mapping[str, object] | None = None
    http_client: httpx.AsyncClient | None = None
    _strict_response_validation: bool = False
    max_input_tokens: int | None = None
    max_completion_tokens: int | None = None
    reasoning_models: list[ChatModel | str] | None = field(
        default_factory=lambda: [
            "o4-mini",
            "o3",
            "o3-mini",
            "o1",
            "o1-pro",
            "o3-pro",
            "gpt-5",
            "gpt-5-mini",
            "gpt-5-nano",
        ]
    )
    _client: AsyncOpenAI | None = field(default=None, init=False, repr=False)
    _owns_client: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        resolved_max_input, resolved_max_output = get_model_limits(str(self.model))

        if self.max_input_tokens is None:
            self.max_input_tokens = resolved_max_input

        if self.max_completion_tokens is None:
            self.max_completion_tokens = resolved_max_output or 4096

    # Static
    @property
    def provider(self) -> str:
        return "openai"

    def _get_client_params(self) -> dict[str, Any]:
        """Prepare client parameters dictionary."""
        # Define base client params
        base_params = {
            "api_key": self.api_key,
            "organization": self.organization,
            "project": self.project,
            "base_url": self.base_url,
            "websocket_base_url": self.websocket_base_url,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "default_headers": self.default_headers,
            "default_query": self.default_query,
            "_strict_response_validation": self._strict_response_validation,
        }

        # Create client_params dict with non-None values
        client_params = {k: v for k, v in base_params.items() if v is not None}

        # Add http_client if provided
        if self.http_client is not None:
            client_params["http_client"] = self.http_client

        return client_params

    def get_client(self) -> AsyncOpenAI:
        """
        Returns an AsyncOpenAI client.

        Returns:
                AsyncOpenAI: An instance of the AsyncOpenAI client.
        """
        if self._client is not None and not self._client.is_closed():
            return self._client

        client_params = self._get_client_params()
        self._client = AsyncOpenAI(**client_params)
        self._owns_client = self.http_client is None
        return self._client

    async def close(self) -> None:
        """Close the owned AsyncOpenAI client to avoid leaking transports on shutdown."""
        client = self._client
        self._client = None
        if client is None:
            return
        if not self._owns_client:
            self._owns_client = False
            return

        self._owns_client = False
        if not client.is_closed():
            await client.close()

    @property
    def name(self) -> str:
        return str(self.model)

    def _build_usage(self, raw_usage: Any) -> ChatInvokeUsage | None:
        if raw_usage is None:
            return None

        completion_tokens = raw_usage.completion_tokens
        completion_token_details = getattr(raw_usage, "completion_tokens_details", None)
        if completion_token_details is not None:
            reasoning_tokens = getattr(completion_token_details, "reasoning_tokens", None)
            if reasoning_tokens is not None:
                completion_tokens += reasoning_tokens

        prompt_tokens_details = getattr(raw_usage, "prompt_tokens_details", None)
        prompt_cached_tokens = (
            getattr(prompt_tokens_details, "cached_tokens", None)
            if prompt_tokens_details is not None
            else None
        )

        return ChatInvokeUsage(
            prompt_tokens=raw_usage.prompt_tokens,
            prompt_cached_tokens=prompt_cached_tokens,
            prompt_cache_creation_tokens=None,
            prompt_image_tokens=None,
            completion_tokens=completion_tokens,
            total_tokens=raw_usage.total_tokens,
        )

    def _get_usage(self, response: ChatCompletion) -> ChatInvokeUsage | None:
        return self._build_usage(response.usage)

    @staticmethod
    def _debug_enabled() -> bool:
        return bool(os.getenv("BU_AGENT_SDK_LLM_DEBUG") or os.getenv("bu_agent_sdk_LLM_DEBUG"))

    @staticmethod
    def _full_curl_debug_enabled() -> bool:
        return bool(os.getenv("BU_AGENT_SDK_LLM_DEBUG_FULL_CURL"))

    @staticmethod
    def _raw_response_debug_enabled() -> bool:
        return bool(os.getenv("BU_AGENT_SDK_LLM_DEBUG_RAW_RESPONSE"))

    @staticmethod
    def _preview_tool_arguments(
        arguments: str,
        max_chars: int = _TOOL_CALL_ARGUMENTS_PREVIEW_CHARS,
    ) -> tuple[str, str]:
        if len(arguments) <= max_chars * 2:
            return arguments, arguments
        return arguments[:max_chars], arguments[-max_chars:]

    @staticmethod
    def _summarize_tool_arguments(arguments: str) -> dict[str, Any]:
        head, tail = ChatOpenAI._preview_tool_arguments(arguments)
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError as exc:
            json_ok = False
            json_error = str(exc)
            parsed_keys = None
        else:
            json_ok = isinstance(parsed, dict)
            json_error = None if json_ok else f"decoded {type(parsed).__name__}, not object"
            parsed_keys = sorted(parsed.keys()) if isinstance(parsed, dict) else None

        return {
            "args_len": len(arguments),
            "json_ok": json_ok,
            "json_error": json_error,
            "has_content_key_text": '"content"' in arguments,
            "parsed_keys": parsed_keys,
            "args_head": head,
            "args_tail": tail,
        }

    @classmethod
    def _log_tool_call_debug(
        cls,
        label: str,
        *,
        tool_call_id: str | None,
        tool_name: str | None,
        arguments: str,
        message_index: int | None = None,
    ) -> None:
        if not cls._debug_enabled():
            return

        summary = cls._summarize_tool_arguments(arguments)
        index_text = f" message_index={message_index}" if message_index is not None else ""
        logger.info(
            "[LLM_DEBUG] %s tool_call%s id=%r name=%r args_len=%s json_ok=%s "
            "json_error=%r has_content_key_text=%s parsed_keys=%r args_head=%r args_tail=%r",
            label,
            index_text,
            tool_call_id,
            tool_name,
            summary["args_len"],
            summary["json_ok"],
            summary["json_error"],
            summary["has_content_key_text"],
            summary["parsed_keys"],
            summary["args_head"],
            summary["args_tail"],
        )

    @classmethod
    def _log_outbound_tool_call_debug(cls, openai_messages: list[Any]) -> None:
        if not cls._debug_enabled():
            return

        for message_index, message in enumerate(openai_messages):
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            tool_calls = message.get("tool_calls")
            if not isinstance(tool_calls, list):
                continue
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                function = tool_call.get("function")
                if not isinstance(function, dict):
                    continue
                arguments = function.get("arguments")
                if not isinstance(arguments, str):
                    arguments = "" if arguments is None else str(arguments)
                cls._log_tool_call_debug(
                    "outbound",
                    message_index=message_index,
                    tool_call_id=tool_call.get("id"),
                    tool_name=function.get("name"),
                    arguments=arguments,
                )

    def _completion_url_for_debug(self) -> str:
        base_url = str(self.base_url or "https://api.openai.com/v1").rstrip("/")
        if base_url.endswith("/chat/completions"):
            return base_url
        return f"{base_url}/chat/completions"

    def _headers_for_curl_debug(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer ***REDACTED***"
        if self.default_headers:
            for key, value in self.default_headers.items():
                lowered = key.lower()
                if lowered in {"authorization", "api-key", "x-api-key"}:
                    headers[key] = "***REDACTED***"
                else:
                    headers[key] = str(value)
        return headers

    @classmethod
    def _sanitize_curl_debug_value(cls, value: Any, *, full: bool) -> Any:
        if isinstance(value, str):
            if full or len(value) <= _CURL_DEBUG_STRING_PREVIEW_CHARS:
                return value
            omitted = len(value) - _CURL_DEBUG_STRING_PREVIEW_CHARS
            return value[:_CURL_DEBUG_STRING_PREVIEW_CHARS] + f"...<truncated {omitted} chars>"
        if isinstance(value, list):
            return [cls._sanitize_curl_debug_value(item, full=full) for item in value]
        if isinstance(value, tuple):
            return [cls._sanitize_curl_debug_value(item, full=full) for item in value]
        if isinstance(value, dict):
            return {
                str(key): cls._sanitize_curl_debug_value(item, full=full)
                for key, item in value.items()
            }
        if hasattr(value, "model_dump"):
            return cls._sanitize_curl_debug_value(value.model_dump(), full=full)
        return value

    @staticmethod
    def _shell_single_quote(value: str) -> str:
        return "'" + value.replace("'", "'\"'\"'") + "'"

    def _log_curl_debug(
        self,
        *,
        openai_messages: list[Any],
        model_params: dict[str, Any],
        stream: bool,
    ) -> None:
        if not self._debug_enabled():
            return

        body = {
            "model": self.model,
            "messages": openai_messages,
            **model_params,
        }
        if stream:
            body["stream"] = True

        full = self._full_curl_debug_enabled()
        debug_body = self._sanitize_curl_debug_value(body, full=full)
        body_json = json.dumps(debug_body, ensure_ascii=False, default=str)
        headers = self._headers_for_curl_debug()
        header_parts = [
            f"  -H {self._shell_single_quote(f'{key}: {value}')} \\"
            for key, value in headers.items()
        ]
        curl_lines = [
            f"curl -X POST {self._shell_single_quote(self._completion_url_for_debug())} \\",
            *header_parts,
            f"  --data-raw {self._shell_single_quote(body_json)}",
        ]
        logger.info(
            "[LLM_DEBUG] outbound_curl stream=%s full_body=%s command=%s",
            stream,
            full,
            "\n".join(curl_lines),
        )

    @classmethod
    def _coerce_debug_payload(cls, payload: Any) -> Any:
        if hasattr(payload, "model_dump"):
            try:
                return payload.model_dump(mode="json")
            except TypeError:
                return payload.model_dump()
        return payload

    @classmethod
    def _log_raw_response_debug(
        cls,
        label: str,
        payload: Any,
        *,
        stream_chunk_index: int | None = None,
    ) -> None:
        if not cls._raw_response_debug_enabled():
            return

        coerced_payload = cls._coerce_debug_payload(payload)
        full = cls._full_curl_debug_enabled()
        debug_payload = cls._sanitize_curl_debug_value(coerced_payload, full=full)
        payload_json = json.dumps(debug_payload, ensure_ascii=False, default=str)
        chunk_text = (
            f" chunk_index={stream_chunk_index}" if stream_chunk_index is not None else ""
        )
        logger.info(
            "[LLM_DEBUG] %s%s full_body=%s payload=%s",
            label,
            chunk_text,
            full,
            payload_json,
        )

    def _serialize_tools(self, tools: list[ToolDefinition]) -> list[ChatCompletionToolParam]:
        """Convert ToolDefinitions to OpenAI's tool format."""
        result = []
        for tool in tools:
            params = tool.parameters
            # For strict mode, OpenAI requires ALL properties in 'required'
            # Transform optional params to required + nullable
            if tool.strict and params.get("properties"):
                params = self._make_strict_schema(params)

            result.append(
                ChatCompletionToolParam(
                    type="function",
                    function=FunctionDefinition(
                        name=tool.name,
                        description=tool.description,
                        parameters=params,
                        strict=tool.strict,
                    ),
                )
            )
        return result

    def _make_strict_schema(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Transform schema for OpenAI strict mode: all properties must be required."""
        schema = schema.copy()
        props = schema.get("properties", {})
        required = set(schema.get("required", []))

        new_props = {}
        for name, prop in props.items():
            prop = self._make_strict_property(prop, name in required)
            new_props[name] = prop

        schema["properties"] = new_props
        schema["required"] = list(props.keys())  # All properties required
        schema["additionalProperties"] = False
        return schema

    def _make_strict_property(self, prop: dict[str, Any], is_required: bool) -> dict[str, Any]:
        """Transform a single property for strict mode, recursively handling nested objects."""
        prop = prop.copy()

        # Handle nested objects
        if prop.get("type") == "object" and "properties" in prop:
            prop = self._make_strict_schema(prop)

        # Handle arrays with object items
        if prop.get("type") == "array" and "items" in prop:
            items = prop["items"]
            if isinstance(items, dict) and items.get("type") == "object" and "properties" in items:
                prop["items"] = self._make_strict_schema(items)

        # Make optional params nullable
        if not is_required:
            if "type" in prop:
                prop["type"] = [prop["type"], "null"]
            elif "anyOf" not in prop:
                prop["anyOf"] = [prop, {"type": "null"}]

        return prop

    def _resolve_prompt_cache_retention(self) -> str | None:
        """Select prompt cache retention based on model support."""
        if self.prompt_cache_retention is not None:
            return self.prompt_cache_retention

        model_name = str(self.model).lower()
        if any(key in model_name for key in self.extended_cache_models):
            return "24h"
        return None

    def _get_tool_choice(
        self, tool_choice: ToolChoice | None, tools: list[ToolDefinition] | None
    ) -> Any:
        """Convert our tool_choice to OpenAI's format."""
        if tool_choice is None or tools is None:
            return None

        if tool_choice == "auto":
            return "auto"
        elif tool_choice == "required":
            return "required"
        elif tool_choice == "none":
            return "none"
        else:
            # Specific tool name - force that tool
            return {"type": "function", "function": {"name": tool_choice}}

    def _extract_tool_calls(self, response: ChatCompletion) -> list[ToolCall]:
        """Extract tool calls from OpenAI response."""
        tool_calls: list[ToolCall] = []
        message = response.choices[0].message

        # 调试：打印原始 tool_calls
        debug_enabled = os.getenv("BU_AGENT_SDK_LLM_DEBUG")
        if debug_enabled and message.tool_calls:
            logger.info(f"[DEBUG] 原始 message.tool_calls:")
            for tc in message.tool_calls:
                logger.info(f"[DEBUG]   id={tc.id}, function={tc.function}")

        if message.tool_calls:
            for tc in message.tool_calls:
                # 检查 arguments 是否为 None
                args = tc.function.arguments if tc.function.arguments else "{}"
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        function=Function(
                            name=tc.function.name,
                            arguments=args,
                        ),
                        type="function",
                    )
                )

        return tool_calls

    def _sanitize_messages_for_openai(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        """Drop malformed tool history so OpenAI-compatible backends accept the request.

        OpenAI requires each `tool` message to be preceded by an assistant message
        with matching `tool_calls`, and each assistant tool-call block must be
        followed by the matching tool result messages before the conversation moves on.
        """
        sanitized: list[BaseMessage] = []
        index = 0

        while index < len(messages):
            message = messages[index]

            if isinstance(message, ToolMessage):
                logger.warning(
                    "Dropping orphan tool message for tool_call_id=%s before OpenAI request",
                    message.tool_call_id,
                )
                index += 1
                continue

            if not isinstance(message, AssistantMessage) or not message.tool_calls:
                sanitized.append(message)
                index += 1
                continue

            expected_tool_ids = {tool_call.id for tool_call in message.tool_calls}
            collected_tool_messages: list[ToolMessage] = []
            next_index = index + 1

            while next_index < len(messages) and isinstance(messages[next_index], ToolMessage):
                tool_message = messages[next_index]
                if tool_message.tool_call_id in expected_tool_ids:
                    collected_tool_messages.append(tool_message)
                else:
                    logger.warning(
                        "Dropping mismatched tool message for tool_call_id=%s before OpenAI request",
                        tool_message.tool_call_id,
                    )
                next_index += 1

            received_tool_ids = {
                tool_message.tool_call_id for tool_message in collected_tool_messages
            }
            if received_tool_ids == expected_tool_ids:
                sanitized.append(message)
                sanitized.extend(collected_tool_messages)
                index = next_index
                continue

            logger.warning(
                "Stripping incomplete assistant tool_calls before OpenAI request; "
                "expected=%s received=%s",
                sorted(expected_tool_ids),
                sorted(received_tool_ids),
            )
            if message.content is not None or message.refusal is not None:
                sanitized.append(
                    AssistantMessage(
                        content=message.content,
                        name=message.name,
                        refusal=message.refusal,
                        tool_calls=None,
                        cache=message.cache,
                    )
                )
            index = next_index

        return sanitized

    @staticmethod
    def _get_stream_tool_call_aliases(
        tool_call_delta: Any,
        *,
        position: int,
    ) -> list[tuple[str, str | int]]:
        """Return best-effort aliases for one streaming tool call delta."""
        del position
        aliases: list[tuple[str, str | int]] = []

        tool_call_id = getattr(tool_call_delta, "id", None)
        if isinstance(tool_call_id, str) and tool_call_id:
            aliases.append(("id", tool_call_id))

        tool_call_index = getattr(tool_call_delta, "index", None)
        if isinstance(tool_call_index, int):
            aliases.append(("index", tool_call_index))

        return aliases

    @staticmethod
    def _tool_arguments_form_complete_json_object(arguments: str) -> bool:
        """Return True when the buffered tool arguments already parse as one JSON object."""
        try:
            parsed = json.loads(arguments)
        except Exception:
            return False
        return isinstance(parsed, dict)

    @classmethod
    def _infer_stream_tool_call_key(
        cls,
        *,
        tool_calls_buffer: dict[str, dict[str, str]],
        tool_call_update_order: list[str],
        incoming_name: str | None,
        incoming_arguments: str | None,
    ) -> str | None:
        """Best-effort fallback when a delta chunk omits both id and index.

        OpenAI-compatible backends sometimes continue one tool call without repeating
        its id/index. We avoid position-based matching because sparse chunk arrays can
        shift positions and concatenate two different JSON payloads. Instead, prefer
        buffered calls that are still incomplete, most recently updated first.
        """
        del incoming_arguments

        candidate_keys: list[str] = []
        for key in reversed(tool_call_update_order):
            entry = tool_calls_buffer.get(key)
            if entry is None:
                continue
            if incoming_name and not entry.get("name"):
                candidate_keys.append(key)
                continue
            if not cls._tool_arguments_form_complete_json_object(entry.get("arguments", "")):
                candidate_keys.append(key)

        if candidate_keys:
            return candidate_keys[0]
        if len(tool_calls_buffer) == 1:
            return next(iter(tool_calls_buffer))
        return None

    async def ainvoke(
        self,
        messages: list[BaseMessage],
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
        **kwargs: Any,
    ) -> ChatInvokeCompletion:
        """
        Invoke the model with the given messages and optional tools.

        Args:
            messages: List of chat messages
            tools: Optional list of tools the model can call
            tool_choice: Control how the model uses tools

        Returns:
            ChatInvokeCompletion with content and/or tool_calls
        """
        sanitized_messages = self._sanitize_messages_for_openai(messages)
        openai_messages = OpenAIMessageSerializer.serialize_messages(sanitized_messages)
        self._log_outbound_tool_call_debug(openai_messages)

        try:
            model_params: dict[str, Any] = {}

            if self.temperature is not None:
                model_params["temperature"] = self.temperature

            if self.frequency_penalty is not None:
                model_params["frequency_penalty"] = self.frequency_penalty

            if self.max_completion_tokens is not None:
                model_params["max_completion_tokens"] = self.max_completion_tokens

            if self.top_p is not None:
                model_params["top_p"] = self.top_p

            if self.seed is not None:
                model_params["seed"] = self.seed

            if self.service_tier is not None:
                model_params["service_tier"] = self.service_tier

            extra_body: dict[str, Any] = {}
            if self.prompt_cache_key is not None:
                extra_body["prompt_cache_key"] = self.prompt_cache_key
            cache_retention = self._resolve_prompt_cache_retention()
            if cache_retention is not None:
                extra_body["prompt_cache_retention"] = cache_retention
            if extra_body:
                model_params["extra_body"] = extra_body

            # Handle reasoning models (o1, o3, etc.)
            if self.reasoning_models and any(
                str(m).lower() in str(self.model).lower() for m in self.reasoning_models
            ):
                model_params["reasoning_effort"] = self.reasoning_effort
                model_params.pop("temperature", None)
                model_params.pop("frequency_penalty", None)

            # Add tools if provided
            if tools:
                model_params["tools"] = self._serialize_tools(tools)
                model_params["parallel_tool_calls"] = self.parallel_tool_calls

                openai_tool_choice = self._get_tool_choice(tool_choice, tools)
                if openai_tool_choice is not None:
                    model_params["tool_choice"] = openai_tool_choice

            self._log_curl_debug(
                openai_messages=openai_messages,
                model_params=model_params,
                stream=False,
            )

            # Make the API call
            response = await self.get_client().chat.completions.create(
                model=self.model,
                messages=openai_messages,
                **model_params,
            )
            self._log_raw_response_debug("inbound_raw_response", response)

            # Extract usage
            usage = self._get_usage(response)

            # Log token usage if bu_agent_sdk_LLM_DEBUG is set
            if usage and os.getenv("bu_agent_sdk_LLM_DEBUG"):
                cached = usage.prompt_cached_tokens or 0
                input_tokens = usage.prompt_tokens - cached
                logger.info(
                    f"📊 {self.model}: {input_tokens:,} in + {cached:,} cached + {usage.completion_tokens:,} out"
                )

            # Extract content
            content = response.choices[0].message.content

            # Extract tool calls
            tool_calls = self._extract_tool_calls(response)
            for tool_call in tool_calls:
                self._log_tool_call_debug(
                    "inbound",
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.function.name,
                    arguments=tool_call.function.arguments,
                )

            return ChatInvokeCompletion(
                content=content,
                tool_calls=tool_calls,
                usage=usage,
                stop_reason=response.choices[0].finish_reason if response.choices else None,
            )

        except RateLimitError as e:
            raise ModelRateLimitError(message=e.message, model=self.name) from e

        except APIConnectionError as e:
            raise ModelProviderError(message=str(e), model=self.name) from e

        except APIStatusError as e:
            raise ModelProviderError(
                message=e.message, status_code=e.status_code, model=self.name
            ) from e

        except Exception as e:
            raise ModelProviderError(message=str(e), model=self.name) from e

    async def astream(
        self,
        messages: list[BaseMessage],
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatInvokeCompletionChunk]:
        """流式调用模型，逐token返回内容

        工具调用参数累积：
        - 在流式过程中累积完整的工具调用参数
        - 只在流结束时返回完整的工具调用信息
        - 这样可以避免 Agent 层退化到非流式模式

        Args:
            messages: 消息列表
            tools: 可选的工具列表
            tool_choice: 工具选择策略
            **kwargs: 额外参数

        Yields:
            ChatInvokeCompletionChunk: 包含增量内容的chunk
            - delta: 增量文本
            - tool_calls: 工具调用（只在最后一个chunk返回完整信息）
            - usage: Token统计（只在最后一个chunk）
            - stop_reason: 停止原因（只在最后一个chunk）
        """
        from agent_core.llm.views import ChatInvokeCompletionChunk

        sanitized_messages = self._sanitize_messages_for_openai(messages)
        openai_messages = OpenAIMessageSerializer.serialize_messages(sanitized_messages)
        self._log_outbound_tool_call_debug(openai_messages)

        try:
            # 准备模型参数（与 ainvoke 相同）
            model_params: dict[str, Any] = {}

            if self.temperature is not None:
                model_params["temperature"] = self.temperature
            if self.frequency_penalty is not None:
                model_params["frequency_penalty"] = self.frequency_penalty
            if self.max_completion_tokens is not None:
                model_params["max_completion_tokens"] = self.max_completion_tokens
            if self.top_p is not None:
                model_params["top_p"] = self.top_p
            if self.seed is not None:
                model_params["seed"] = self.seed
            if self.service_tier is not None:
                model_params["service_tier"] = self.service_tier

            extra_body: dict[str, Any] = {}
            if self.prompt_cache_key is not None:
                extra_body["prompt_cache_key"] = self.prompt_cache_key
            cache_retention = self._resolve_prompt_cache_retention()
            if cache_retention is not None:
                extra_body["prompt_cache_retention"] = cache_retention
            if extra_body:
                model_params["extra_body"] = extra_body

            # 处理推理模型
            if self.reasoning_models and any(
                str(m).lower() in str(self.model).lower() for m in self.reasoning_models
            ):
                model_params["reasoning_effort"] = self.reasoning_effort
                model_params.pop("temperature", None)
                model_params.pop("frequency_penalty", None)

            # 添加工具
            if tools:
                model_params["tools"] = self._serialize_tools(tools)
                model_params["parallel_tool_calls"] = self.parallel_tool_calls
                openai_tool_choice = self._get_tool_choice(tool_choice, tools)
                if openai_tool_choice is not None:
                    model_params["tool_choice"] = openai_tool_choice

            # 流式调用
            model_params["stream_options"] = {"include_usage": True}
            self._log_curl_debug(
                openai_messages=openai_messages,
                model_params=model_params,
                stream=True,
            )
            stream = await self.get_client().chat.completions.create(
                model=self.model,
                messages=openai_messages,
                stream=True,
                **model_params,
            )

            # 工具调用累积缓冲区：index -> {id, name, arguments}
            # OpenAI 流式 API 使用 index 来标识同一个工具调用的不同 chunk
            tool_calls_buffer: dict[str, dict[str, str]] = {}
            tool_call_aliases: dict[tuple[str, str | int], str] = {}
            anonymous_tool_call_counter = 0
            tool_call_update_order: list[str] = []
            last_usage: ChatInvokeUsage | None = None
            usage_emitted = False

            # 遍历流式响应
            stream_chunk_index = 0
            async for chunk in stream:
                self._log_raw_response_debug(
                    "inbound_raw_stream_chunk",
                    chunk,
                    stream_chunk_index=stream_chunk_index,
                )
                stream_chunk_index += 1
                usage = self._build_usage(getattr(chunk, "usage", None))
                if usage is not None:
                    last_usage = usage

                if not chunk.choices:
                    continue

                choice = chunk.choices[0]
                delta = choice.delta

                # 处理增量文本
                content_delta = delta.content or ""
                thinking_delta = getattr(delta, "reasoning", None) or None

                # 累积工具调用参数
                if delta.tool_calls:
                    for position, tc in enumerate(delta.tool_calls):
                        # OpenAI 使用 index 识别同一个工具调用
                        aliases = self._get_stream_tool_call_aliases(tc, position=position)
                        canonical_key = next(
                            (tool_call_aliases[alias] for alias in aliases if alias in tool_call_aliases),
                            None,
                        )
                        tool_call_id = getattr(tc, "id", None)
                        tool_call_index = getattr(tc, "index", None)
                        incoming_name = tc.function.name if tc.function else None
                        incoming_arguments = tc.function.arguments if tc.function else None

                        if canonical_key is None:
                            if isinstance(tool_call_id, str) and tool_call_id:
                                canonical_key = f"id:{tool_call_id}"
                            elif isinstance(tool_call_index, int):
                                canonical_key = f"index:{tool_call_index}"
                            else:
                                canonical_key = self._infer_stream_tool_call_key(
                                    tool_calls_buffer=tool_calls_buffer,
                                    tool_call_update_order=tool_call_update_order,
                                    incoming_name=incoming_name,
                                    incoming_arguments=incoming_arguments,
                                )
                                if canonical_key is None:
                                    canonical_key = f"anonymous:{anonymous_tool_call_counter}"
                                    anonymous_tool_call_counter += 1

                        if canonical_key not in tool_calls_buffer:
                            # 第一次遇到这个工具调用，初始化
                            fallback_id = (
                                tool_call_id
                                if isinstance(tool_call_id, str) and tool_call_id
                                else (
                                    f"call_{tool_call_index}"
                                    if isinstance(tool_call_index, int)
                                    else canonical_key.replace(":", "_")
                                )
                            )
                            tool_calls_buffer[canonical_key] = {
                                "id": fallback_id,
                                "name": "",
                                "arguments": "",
                            }

                        # 累积参数增量（arguments 是分段返回的）
                        entry = tool_calls_buffer[canonical_key]
                        for alias in aliases:
                            tool_call_aliases[alias] = canonical_key

                        if isinstance(tool_call_id, str) and tool_call_id:
                            entry["id"] = tool_call_id
                            tool_call_aliases[("id", tool_call_id)] = canonical_key
                        if isinstance(tool_call_index, int):
                            tool_call_aliases[("index", tool_call_index)] = canonical_key

                        if tc.function:
                            if tc.function.name:
                                entry["name"] = tc.function.name
                            if tc.function.arguments:
                                entry["arguments"] += tc.function.arguments

                        if canonical_key in tool_call_update_order:
                            tool_call_update_order.remove(canonical_key)
                        tool_call_update_order.append(canonical_key)

                # 在中间 chunk 中，只返回 delta 和空的 tool_calls
                # 完整的 tool_calls 只在最后一个 chunk 返回
                yield ChatInvokeCompletionChunk(
                    delta=content_delta,
                    tool_calls=[],  # 中间 chunk 不返回工具调用
                    thinking=thinking_delta,
                    usage=usage,
                    stop_reason=choice.finish_reason,
                )
                if usage is not None:
                    usage_emitted = True

            # 流结束后，构建完整的工具调用列表
            complete_tool_calls: list[ToolCall] = []
            for tc_data in tool_calls_buffer.values():
                tc_name = tc_data.get("name", "")
                # 只返回有名称的完整工具调用
                if tc_name:
                    complete_tool_calls.append(
                        ToolCall(
                            id=tc_data["id"],
                            function=Function(
                                name=tc_name,
                                arguments=tc_data.get("arguments", "{}") or "{}",
                            ),
                            type="function",
                        )
                    )

            for tool_call in complete_tool_calls:
                self._log_tool_call_debug(
                    "inbound_stream_complete",
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.function.name,
                    arguments=tool_call.function.arguments,
                )

            # 最后再 yield 一次，这次包含完整的工具调用信息
            # 如果流结束时没有工具调用，这个 chunk 会被 agent 层忽略
            final_usage = last_usage if not usage_emitted else None
            if complete_tool_calls:
                yield ChatInvokeCompletionChunk(
                    delta="",
                    tool_calls=complete_tool_calls,
                    thinking=None,
                    usage=final_usage,
                    stop_reason=None,
                )
            elif final_usage is not None:
                yield ChatInvokeCompletionChunk(
                    delta="",
                    tool_calls=[],
                    thinking=None,
                    usage=final_usage,
                    stop_reason=None,
                )

        except RateLimitError as e:
            raise ModelRateLimitError(message=e.message, model=self.name) from e
        except APIConnectionError as e:
            raise ModelProviderError(message=str(e), model=self.name) from e
        except APIStatusError as e:
            raise ModelProviderError(
                message=e.message, status_code=e.status_code, model=self.name
            ) from e
        except Exception as e:
            raise ModelProviderError(message=str(e), model=self.name) from e

    async def ainvoke_streaming(
        self,
        messages: list[BaseMessage],
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
        **kwargs: Any,
    ) -> ChatInvokeCompletion:
        """
        使用流式调用但同步返回结果（解决OAM90s超时问题）

        内部实现：
        1. 调用 astream() 获取流式迭代器
        2. 收集所有 chunk 的内容
        3. 组装成完整的 ChatInvokeCompletion 返回

        Args:
            messages: 消息列表
            tools: 可选的工具列表
            tool_choice: 工具选择策略
            **kwargs: 额外参数

        Returns:
            ChatInvokeCompletion: 与 ainvoke() 返回类型相同
        """
        from agent_core.llm.messages import ToolCall

        # 收集所有内容
        content_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        thinking_parts: list[str] = []
        usage: ChatInvokeUsage | None = None
        stop_reason: str | None = None

        async for chunk in self.astream(messages, tools, tool_choice, **kwargs):
            # 累积文本内容
            if chunk.delta:
                content_parts.append(chunk.delta)
            if chunk.thinking:
                thinking_parts.append(chunk.thinking)

            # 累积工具调用（只在最后的chunk中）
            if chunk.tool_calls:
                tool_calls.extend(chunk.tool_calls)

            # 收集 usage 和 stop_reason（只在最后的chunk中）
            if chunk.usage:
                usage = chunk.usage
            if chunk.stop_reason:
                stop_reason = chunk.stop_reason

        return ChatInvokeCompletion(
            content="".join(content_parts) if content_parts else None,
            tool_calls=tool_calls,
            thinking="".join(thinking_parts) if thinking_parts else None,
            usage=usage,
            stop_reason=stop_reason,
        )
