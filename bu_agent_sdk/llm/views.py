from pydantic import BaseModel

from bu_agent_sdk.llm.messages import ToolCall


class ChatInvokeCompletionChunk(BaseModel):
    """
    流式响应的单个数据块

    每个chunk包含增量内容，用于实时显示LLM输出
    """

    delta: str = ""
    """增量文本内容"""

    tool_calls: list[ToolCall] = []
    """工具调用（流式时只在第一个chunk返回）"""

    thinking: str | None = None
    """思考内容（推理模型）"""

    usage: ChatInvokeUsage | None = None
    """Token使用统计（只在最后一个chunk返回）"""

    stop_reason: str | None = None
    """停止原因（只在最后一个chunk返回）"""

    @property
    def has_tool_calls(self) -> bool:
        """检查是否包含工具调用"""
        return len(self.tool_calls) > 0


class ChatInvokeUsage(BaseModel):
    """
    Usage information for a chat model invocation.
    """

    prompt_tokens: int
    """The number of tokens in the prompt (this includes the cached tokens as well. When calculating the cost, subtract the cached tokens from the prompt tokens)"""

    prompt_cached_tokens: int | None
    """The number of cached tokens."""

    prompt_cache_creation_tokens: int | None
    """Anthropic only: The number of tokens used to create the cache."""

    prompt_image_tokens: int | None
    """Google only: The number of tokens in the image (prompt tokens is the text tokens + image tokens in that case)"""

    completion_tokens: int
    """The number of tokens in the completion."""

    total_tokens: int
    """The total number of tokens in the response."""


class ChatInvokeCompletion(BaseModel):
    """
    Response from a chat model invocation.

    For tool calling workflows:
    - If the model wants to call tools, `tool_calls` will be populated and `content` may be empty
    - If the model responds with text, `content` will be populated and `tool_calls` will be empty
    - Both can be present (model can respond with text AND request tool calls)
    """

    content: str | None = None
    """The text content of the response, if any."""

    tool_calls: list[ToolCall] = []
    """Tool calls requested by the model.

	When non-empty, the client should:
	1. Execute each tool call
	2. Send ToolMessage(s) back with the results
	3. Continue the conversation
	"""

    # Thinking stuff (for reasoning models)
    thinking: str | None = None
    """Extended thinking content (Anthropic, Google)."""

    redacted_thinking: str | None = None
    """Redacted thinking content (Anthropic)."""

    usage: ChatInvokeUsage | None = None
    """Token usage information for this response."""

    stop_reason: str | None = None
    """The reason the model stopped generating.

	Common values:
	- 'end_turn' / 'stop': Normal completion
	- 'tool_use' / 'tool_calls': Model wants to call tools
	- 'max_tokens': Hit token limit
	- 'stop_sequence': Hit a stop sequence
	"""

    @property
    def has_tool_calls(self) -> bool:
        """Check if the response contains tool calls."""
        return len(self.tool_calls) > 0

    @property
    def text(self) -> str:
        """Get the text content, or empty string if None."""
        return self.content or ""
