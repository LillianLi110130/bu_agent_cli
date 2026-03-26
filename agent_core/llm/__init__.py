"""
LLM abstraction layer with type-safe tool calling support.

This module provides a unified interface for chat models across different providers
with first-class support for tool calling.
"""

from typing import TYPE_CHECKING

# Auto-load .env file for API keys
from dotenv import load_dotenv

load_dotenv()

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

# Chat models - direct import
from agent_core.llm.openai.chat import ChatOpenAI

# Type stubs for lazy imports
if TYPE_CHECKING:
    pass


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
    "ChatOpenAI",
]
