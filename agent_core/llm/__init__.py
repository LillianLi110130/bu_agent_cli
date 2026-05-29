"""
LLM abstraction layer with type-safe tool calling support.

This module provides a unified interface for chat models across different providers
with first-class support for tool calling.
"""

from typing import TYPE_CHECKING, Any

# Core types - always imported
from agent_core.llm.base import BaseChatModel, ToolChoice, ToolDefinition
from agent_core.llm.messages import (
    AssistantMessage,
    BaseMessage,
    DeveloperMessage,
    Function,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from agent_core.llm.messages import (
    ContentPartImageParam as ContentImage,
)
from agent_core.llm.messages import (
    ContentPartRedactedThinkingParam as ContentRedactedThinking,
)
from agent_core.llm.messages import (
    ContentPartRefusalParam as ContentRefusal,
)
from agent_core.llm.messages import (
    ContentPartTextParam as ContentText,
)
from agent_core.llm.messages import (
    ContentPartThinkingParam as ContentThinking,
)
from agent_core.llm.views import ChatInvokeCompletion, ChatInvokeUsage

# Type stubs for lazy imports
if TYPE_CHECKING:
    from agent_core.llm.gateway.chat import ChatGateway
    from agent_core.llm.openai.chat import ChatOpenAI


__all__ = [
    # Message types
    "BaseMessage",
    "UserMessage",
    "SystemMessage",
    "AssistantMessage",
    "ToolMessage",
    "DeveloperMessage",
    # Tool calling types
    "ToolCall",
    "Function",
    "ToolDefinition",
    "ToolChoice",
    # Response types
    "ChatInvokeCompletion",
    "ChatInvokeUsage",
    # Content parts with better names
    "ContentText",
    "ContentRefusal",
    "ContentImage",
    "ContentThinking",
    "ContentRedactedThinking",
    # Chat models
    "BaseChatModel",
    "ChatGateway",
    "ChatOpenAI",
]

_CHAT_MODEL_EXPORTS = {"ChatGateway", "ChatOpenAI"}


def __getattr__(name: str) -> Any:
    if name not in _CHAT_MODEL_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from agent_core.llm.gateway.chat import ChatGateway
    from agent_core.llm.openai.chat import ChatOpenAI

    exports = {
        "ChatGateway": ChatGateway,
        "ChatOpenAI": ChatOpenAI,
    }
    globals().update(exports)
    return exports[name]


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
