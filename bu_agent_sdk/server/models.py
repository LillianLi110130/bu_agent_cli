"""
Request and response models for the HTTP API.

These Pydantic models define the API contract for the server.
"""

from datetime import UTC, datetime
from typing import Any, Literal
from pydantic import BaseModel, Field
from bu_agent_sdk.tokens.views import UsageSummary, ModelUsageStats


class QueryRequest(BaseModel):
    """Request model for agent query endpoint."""

    message: str = Field(
        ...,
        description="The user message to send to the agent",
        examples=["What is the weather today?"],
    )
    session_id: str | None = Field(
        default=None,
        description="Optional session ID for maintaining conversation state. "
        "If not provided, a new session will be created.",
    )
    user_id: str | None = Field(
        default=None,
        description="Optional user ID used to bind and restore persistent session memory.",
    )
    stream: bool = Field(
        default=False,
        description="Whether to stream the response as Server-Sent Events",
    )
    skill: str | None = Field(
        default=None,
        description="Optional skill name to invoke. "
        "If provided, the skill content will be prepended to the message.",
    )


class ToolCallInfo(BaseModel):
    """Information about a tool call."""

    tool: str = Field(..., description="Name of the tool being called")
    args: dict[str, Any] = Field(default_factory=dict, description="Arguments passed to the tool")
    tool_call_id: str = Field(..., description="Unique ID of this tool call")
    display_name: str = Field(default="", description="Human-readable description")


class ToolResultInfo(BaseModel):
    """Result of a tool execution."""

    tool: str = Field(..., description="Name of the tool that was called")
    result: str = Field(..., description="Result returned by the tool")
    tool_call_id: str = Field(..., description="ID of the corresponding tool call")
    is_error: bool = Field(default=False, description="Whether the execution resulted in an error")


class StepInfo(BaseModel):
    """Information about a logical step."""

    step_id: str = Field(..., description="Unique ID for this step")
    title: str = Field(..., description="Human-readable title for this step")
    step_number: int = Field(default=0, description="Sequential step number")
    status: Literal["started", "completed", "error"] = Field(..., description="Step status")
    duration_ms: float = Field(default=0.0, description="Duration in milliseconds")


class UsageInfo(BaseModel):
    """Token usage information."""

    total_tokens: int = Field(..., description="Total tokens used")
    total_prompt_tokens: int = Field(..., description="Total prompt tokens")
    total_completion_tokens: int = Field(..., description="Total completion tokens")
    by_model: dict[str, Any] = Field(
        default_factory=dict, description="Usage statistics broken down by model"
    )

    @classmethod
    def from_usage_summary(cls, summary: UsageSummary) -> "UsageInfo":
        """Create UsageInfo from UsageSummary."""
        return cls(
            total_tokens=summary.total_tokens,
            total_prompt_tokens=summary.total_prompt_tokens,
            total_completion_tokens=summary.total_completion_tokens,
            by_model={
                model: ModelUsageStats(
                    model=stats.model,
                    prompt_tokens=stats.prompt_tokens,
                    completion_tokens=stats.completion_tokens,
                    total_tokens=stats.total_tokens,
                    invocations=stats.invocations,
                    average_tokens_per_invocation=stats.average_tokens_per_invocation,
                ).model_dump()
                for model, stats in summary.by_model.items()
            },
        )


class QueryResponse(BaseModel):
    """Response model for agent query endpoint."""

    session_id: str = Field(..., description="Session ID for this conversation")
    response: str = Field(..., description="The agent's final response")
    usage: UsageInfo = Field(..., description="Token usage information")
    skill_used: str | None = Field(default=None, description="The skill that was applied, if any")


class StreamEventType(BaseModel):
    """Base class for stream event types."""

    type: str = Field(..., description="Event type name")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC), description="Event timestamp")


class TextEvent(StreamEventType):
    """Emitted when the assistant produces text content."""

    type: Literal["text"] = "text"
    content: str = Field(..., description="The text content from the assistant")


class TextDeltaEvent(StreamEventType):
    """流式文本增量事件 - 实时输出LLM生成的每个token

    客户端应该将 delta 内容追加到当前文本缓冲区中。
    """

    type: Literal["text_delta"] = "text_delta"
    delta: str = Field(..., description="增量文本内容（可能为空字符串）")


class ThinkingEvent(StreamEventType):
    """Emitted when the model produces thinking/reasoning content."""

    type: Literal["thinking"] = "thinking"
    content: str = Field(..., description="The thinking content")


class ThinkingStartEvent(StreamEventType):
    """Emitted when a thinking block starts (optimistic detection of think tag)."""

    type: Literal["thinking_start"] = "thinking_start"
    think_id: str = Field(..., description="Unique ID for this thinking block")


class ThinkingDeltaEvent(StreamEventType):
    """流式思考增量事件 - 实时输出思考内容的每个字符."""

    type: Literal["thinking_delta"] = "thinking_delta"
    delta: str = Field(..., description="增量思考内容（可能为空字符串）")
    think_id: str = Field(..., description="The ID of the thinking block this delta belongs to")


class ThinkingEndEvent(StreamEventType):
    """Emitted when a thinking block ends (think tag detected)."""

    type: Literal["thinking_end"] = "thinking_end"
    think_id: str = Field(..., description="The ID of the completed thinking block")


class ToolCallEvent(StreamEventType):
    """Emitted when the assistant calls a tool."""

    type: Literal["tool_call"] = "tool_call"
    tool: str = Field(..., description="Name of the tool being called")
    args: dict[str, Any] = Field(default_factory=dict, description="Arguments passed to the tool")
    tool_call_id: str = Field(..., description="Unique ID of this tool call")
    display_name: str = Field(default="", description="Human-readable description")


class ToolResultEvent(StreamEventType):
    """Emitted when a tool returns a result."""

    type: Literal["tool_result"] = "tool_result"
    tool: str = Field(..., description="Name of the tool that was called")
    result: str = Field(..., description="Result returned by the tool")
    tool_call_id: str = Field(..., description="ID of the corresponding tool call")
    is_error: bool = Field(default=False, description="Whether the execution resulted in an error")
    screenshot_base64: str | None = Field(
        default=None, description="Base64-encoded screenshot if available"
    )


class StepStartEvent(StreamEventType):
    """Emitted when the agent starts a logical step."""

    type: Literal["step_start"] = "step_start"
    step_id: str = Field(..., description="Unique ID for this step")
    title: str = Field(..., description="Human-readable title for this step")
    step_number: int = Field(default=0, description="Sequential step number")


class StepCompleteEvent(StreamEventType):
    """Emitted when a step completes."""

    type: Literal["step_complete"] = "step_complete"
    step_id: str = Field(..., description="ID of the completed step")
    status: Literal["completed", "error"] = Field(..., description="Final status of the step")
    duration_ms: float = Field(default=0.0, description="Duration in milliseconds")


class FinalResponseEvent(StreamEventType):
    """Emitted when the agent produces its final response."""

    type: Literal["final"] = "final"
    content: str = Field(..., description="The final response content")


class HiddenMessageEvent(StreamEventType):
    """Emitted when the agent injects a hidden user message."""

    type: Literal["hidden"] = "hidden"
    content: str = Field(..., description="The content of the hidden message")


# Union type for all stream events (for type checking)
StreamEvent = (
    TextEvent
    | TextDeltaEvent
    | ThinkingEvent
    | ThinkingStartEvent
    | ThinkingDeltaEvent
    | ThinkingEndEvent
    | ToolCallEvent
    | ToolResultEvent
    | StepStartEvent
    | StepCompleteEvent
    | FinalResponseEvent
    | HiddenMessageEvent
)


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error message")
    detail: str | None = Field(default=None, description="Detailed error information")
    error_code: str | None = Field(default=None, description="Application-specific error code")


class SessionCreateRequest(BaseModel):
    """Request model for creating a new session."""

    system_prompt: str | None = Field(
        default=None, description="Optional system prompt for this session"
    )
    user_id: str | None = Field(
        default=None,
        description="Optional user ID to bind to the created session.",
    )


class SessionCreateResponse(BaseModel):
    """Response model for session creation."""

    session_id: str = Field(..., description="The newly created session ID")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SessionInfoResponse(BaseModel):
    """Response model for session info."""

    session_id: str = Field(..., description="Session ID")
    user_id: str | None = Field(default=None, description="Bound user ID for this session")
    created_at: datetime = Field(..., description="Session creation timestamp")
    message_count: int = Field(..., description="Number of messages in this session")
    usage: UsageInfo = Field(..., description="Token usage information")


class ClearHistoryRequest(BaseModel):
    """Request model for clearing session history."""

    session_id: str = Field(..., description="Session ID to clear history for")


class ClearHistoryResponse(BaseModel):
    """Response model for clearing history."""

    session_id: str = Field(..., description="Session ID that was cleared")
    cleared: bool = Field(..., description="Whether the history was cleared")


class HealthResponse(BaseModel):
    """Health check response."""

    status: Literal["healthy", "unhealthy"] = Field(..., description="Server health status")
    version: str = Field(default="1.0.0", description="API version")
    active_sessions: int = Field(default=0, description="Number of active sessions")


# ================================
# Skills API Models
# ================================


class SkillInfo(BaseModel):
    """Information about a skill."""

    name: str = Field(..., description="Unique skill identifier")
    display_name: str = Field(..., description="Human-readable display name")
    description: str = Field(..., description="Skill description")
    content: str = Field(..., description="Skill content in markdown format")
    category: str = Field(default="General", description="Skill category")
    source: Literal["config", "database", "api"] = Field(..., description="Where the skill was loaded from")
    enabled: bool = Field(default=True, description="Whether the skill is enabled")
    version: str = Field(default="1.0", description="Skill version")
    tags: list[str] = Field(default_factory=list, description="Optional tags for filtering")


class SkillListResponse(BaseModel):
    """Response model for listing skills."""

    skills: list[SkillInfo] = Field(..., description="List of all skills")
    total: int = Field(..., description="Total number of skills")


class SkillByCategoryResponse(BaseModel):
    """Response model for listing skills by category."""

    categories: dict[str, list[SkillInfo]] = Field(
        ..., description="Skills grouped by category"
    )


class SkillInvokeRequest(BaseModel):
    """Request model for invoking a skill."""

    skill_name: str = Field(..., description="Name of the skill to invoke")
    user_input: str = Field(..., description="User input to process with the skill")


class SkillInvokeResponse(BaseModel):
    """Response model for skill invocation result."""

    skill_name: str = Field(..., description="Name of the invoked skill")
    enhanced_prompt: str = Field(..., description="The enhanced prompt with skill content")


class SkillCreateRequest(BaseModel):
    """Request model for creating a new skill (database mode)."""

    name: str = Field(..., description="Unique skill identifier")
    display_name: str = Field(..., description="Human-readable display name")
    description: str = Field(..., description="Skill description")
    content: str = Field(..., description="Skill content in markdown format")
    category: str = Field(default="General", description="Skill category")
    enabled: bool = Field(default=True, description="Whether the skill is enabled")
    version: str = Field(default="1.0", description="Skill version")
    tags: list[str] = Field(default_factory=list, description="Optional tags")


class SkillUpdateRequest(BaseModel):
    """Request model for updating a skill."""

    display_name: str | None = Field(None, description="Human-readable display name")
    description: str | None = Field(None, description="Skill description")
    content: str | None = Field(None, description="Skill content in markdown format")
    category: str | None = Field(None, description="Skill category")
    enabled: bool | None = Field(None, description="Whether the skill is enabled")
    version: str | None = Field(None, description="Skill version")
    tags: list[str] | None = Field(None, description="Optional tags")


class SkillDeleteResponse(BaseModel):
    """Response model for skill deletion."""

    deleted: bool = Field(..., description="Whether the skill was deleted")
    skill_name: str = Field(..., description="Name of the deleted skill")
