"""
Simple agentic loop implementation with native tool calling.

Usage:
    from bu_agent_sdk.llm import ChatOpenAI
    from bu_agent_sdk.tools import tool
    from bu_agent_sdk import Agent

    @tool("Search the web")
    async def search(query: str) -> str:
        return f"Results for {query}"

    agent = Agent(
        llm=ChatOpenAI(model="gpt-4o"),
        tools=[search],
    )

    response = await agent.query("Find information about Python")
    follow_up = await agent.query("Tell me more about that")

    # Compaction is enabled by default with dynamic thresholds based on model limits
    from bu_agent_sdk.agent.compaction import CompactionConfig

    agent = Agent(
        llm=ChatOpenAI(model="gpt-4o"),
        tools=[search],
        # Custom threshold ratio (default is 0.80 = 80% of model's context window)
        compaction=CompactionConfig(threshold_ratio=0.70),
        # Or disable compaction entirely:
        # compaction=CompactionConfig(enabled=False),
    )

    # Access usage statistics:
    summary = await agent.usage
    print(f"Total tokens: {summary.total_tokens}")
"""


class TaskComplete(Exception):
    """Exception raised when a task is completed via the done tool.

    This provides explicit task completion signaling instead of relying on
    the absence of tool calls. The agent loop catches this exception and
    returns the completion message.

    Attributes:
        message: A description of why the task is complete and what was accomplished.
    """

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


import asyncio
import json
import logging
import random
import time
from collections.abc import AsyncIterator
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path

from bu_agent_sdk.agent.compaction import CompactionConfig
from bu_agent_sdk.agent.context import ContextManager
from bu_agent_sdk.agent.config import AgentConfig

logger = logging.getLogger("bu_agent_sdk.agent")
from bu_agent_sdk.agent.events import (
    AgentEvent,
    FinalResponseEvent,
    HiddenUserMessageEvent,
    StepCompleteEvent,
    StepStartEvent,
    TextEvent,
    ThinkingEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from bu_agent_sdk.llm.base import BaseChatModel, ToolChoice, ToolDefinition
from bu_agent_sdk.llm.exceptions import ModelProviderError, ModelRateLimitError
from bu_agent_sdk.llm.messages import (
    AssistantMessage,
    BaseMessage,
    ContentPartImageParam,
    ContentPartTextParam,
    DeveloperMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from bu_agent_sdk.llm.views import ChatInvokeCompletion
from bu_agent_sdk.observability import Laminar, observe
from bu_agent_sdk.tokens import TokenCost, UsageSummary
from bu_agent_sdk.tools.decorator import Tool


@dataclass
class ModelSwitchPreflightResult:
    """Result of preflight checks before switching to another model."""

    ok: bool
    target_model: str
    estimated_tokens: int
    threshold: int
    context_limit: int
    threshold_utilization: float
    compacted: bool = False
    reason: str | None = None


@dataclass
class Agent:
    """
    Simple agentic loop that manages tool calling and message history.

    The agent will:
    1. Send the task to the LLM with available tools
    2. If the LLM returns tool calls, execute them and add results to history
    3. Repeat until the LLM returns a text response without tool calls
    4. Return the final response

    When compaction is enabled, the agent will automatically compress the
    conversation history when token usage exceeds the configured threshold.

    Attributes:
        llm: The language model to use for the agent.
        tools: List of Tool instances (created with @tool decorator).
        system_prompt: Optional system prompt to guide the agent.
        max_iterations: Maximum number of LLM calls before stopping.
        tool_choice: How the LLM should choose tools ('auto', 'required', 'none').
        compaction: Optional configuration for automatic context compaction.
        dependency_overrides: Optional dict to override tool dependencies.
        mode: Agent mode ('primary', 'subagent', 'all').
        agent_config: Agent configuration (used for subagent and all modes).
    """

    llm: BaseChatModel
    tools: list[Tool]
    system_prompt: str | None = None
    max_iterations: int = 200  # 200 steps max for now
    tool_choice: ToolChoice = "auto"
    compaction: CompactionConfig | None = None
    dependency_overrides: dict | None = None
    ephemeral_storage_path: Path | None = None
    """Path to store destroyed ephemeral message content. If None, content is discarded."""
    require_done_tool: bool = False
    """If True, the agent will only finish when the 'done' tool is called, not when LLM returns no tool calls."""
    llm_max_retries: int = 5
    """Maximum retries for LLM errors at the agent level (matches browser-use default)."""
    llm_retry_base_delay: float = 1.0
    """Base delay in seconds for exponential backoff on LLM retries."""
    llm_retry_max_delay: float = 60.0
    """Maximum delay in seconds between LLM retry attempts."""
    llm_retryable_status_codes: set[int] = field(
        default_factory=lambda: {429, 500, 502, 503, 504}
    )
    """HTTP status codes that trigger retries (matches browser-use)."""
    mode: str = "primary"
    """Agent mode: 'primary' (can call subagents), 'subagent' (can be called), 'all' (both)."""
    agent_config: AgentConfig | None = None
    """Agent configuration with tool permissions and other metadata."""

    # Internal state
    _context: ContextManager = field(default_factory=ContextManager, repr=False)
    _tool_map: dict[str, Tool] = field(default_factory=dict, repr=False)
    _token_cost: TokenCost = field(default=None, repr=False)  # type: ignore

    def __post_init__(self):
        # Validate that all tools are Tool instances
        for t in self.tools:
            assert isinstance(
                t, Tool
            ), f"Expected Tool instance, got {type(t).__name__}. Did you forget to use the @tool decorator?"

        # Build tool lookup map
        self._tool_map = {t.name: t for t in self.tools}

        # Filter tools based on mode and agent_config
        if self.mode in ("subagent", "all") and self.agent_config and self.agent_config.tools:
            self._filter_tools_by_config()

        # Initialize token cost service
        self._token_cost = TokenCost()

        # Initialize compaction service in context (enabled by default)
        # Use provided config or create default (which has enabled=True)
        compaction_config = (
            self.compaction if self.compaction is not None else CompactionConfig()
        )
        self._context.configure_compaction(
            config=compaction_config,
            llm=self.llm,
            token_cost=self._token_cost,
        )

    @property
    def tool_definitions(self) -> list[ToolDefinition]:
        """Get tool definitions for all registered tools."""
        return [t.definition for t in self.tools]

    @property
    def messages(self) -> list[BaseMessage]:
        """Get the current message history (read-only copy)."""
        return self._context.get_messages()

    @property
    def token_cost(self) -> TokenCost:
        """Get the token cost service for direct access to usage tracking."""
        return self._token_cost

    async def get_usage(self) -> UsageSummary:
        """Get usage summary for the agent.

        Returns:
            UsageSummary with token counts.
        """
        return await self._token_cost.get_usage_summary()

    def clear_history(self):
        """Clear the message history and token usage."""
        self._context.clear_messages()
        self._token_cost.clear_history()

    def _filter_tools_by_config(self):
        """根据agent配置过滤工具"""
        if not self.agent_config or not self.agent_config.tools:
            return

        tools_config = self.agent_config.tools
        filtered_tools = []

        # 如果mode是subagent，应该禁用todo工具以及禁用任务工具（防止递归）
        forced_disabled = {"subagent", "todo_read", "todo_write"} \
            if self.mode == "subagent" else set()

        for tool in self.tools:
            if tool.name in forced_disabled:
                continue
            tool_allowed = tools_config.get(tool.name, True)
            if tool_allowed:
                filtered_tools.append(tool)

        self.tools = filtered_tools
        self._tool_map = {t.name: t for t in self.tools}

    def load_history(self, messages: list[BaseMessage]) -> None:
        """Load message history to continue a previous conversation.

        Use this to resume a conversation from previously saved state,
        e.g., when loading from a database on a new machine.

        Note: The system prompt will NOT be re-added on the next query()
        call since _context will be non-empty.

        Args:
                messages: List of BaseMessage instances to load.

        Example:
                # Load and parse messages from your DB
                messages = [parse_message(row) for row in db.query(...)]

                agent = BU(llm=llm, tools=tools, ...)
                agent.load_history(messages)

                # Continue with follow-up
                response = await agent.query("Continue the task...")
        """
        self._context.replace_messages(messages)
        self._token_cost.clear_history()

    # 对标记为ephemeral 的 ToolMessage，按 tool 维度，只保留最近N条
    def _destroy_ephemeral_messages(self) -> None:
        """Destroy old ephemeral message content, keeping the last N per tool.

        Tools can specify how many outputs to keep via _ephemeral attribute:
        - _ephemeral = 3 means keep the last 3 outputs of this tool
        - _ephemeral = True is treated as _ephemeral = 1 (keep last 1)

        Older outputs beyond the limit have their content:
        1. Optionally saved to disk if ephemeral_storage_path is set
        2. Replaced with '<removed to save context>'

        This should be called after each LLM invocation.
        """
        self._context.prune_ephemeral(
            tool_map=self._tool_map,
            storage_path=self.ephemeral_storage_path,
        )

    async def _execute_tool_call(self, tool_call: ToolCall) -> ToolMessage:
        """Execute a single tool call and return the result as a ToolMessage."""
        tool_name = tool_call.function.name
        tool = self._tool_map.get(tool_name)

        if tool is None:
            return ToolMessage(
                tool_call_id=tool_call.id,
                tool_name=tool_name,
                content=f"Error: Unknown tool '{tool_name}'",
                is_error=True,
            )

        # Create Laminar span for tool execution
        if Laminar is not None:
            span_context = Laminar.start_as_current_span(
                name=tool_name,
                input={
                    "tool": tool_name,
                    "arguments": tool_call.function.arguments,
                },
                span_type="TOOL",
            )
        else:
            span_context = nullcontext()

        # Handle TaskComplete outside the span context to avoid it being logged as an error
        task_complete_exception = None

        with span_context:
            try:
                # Parse arguments
                args = json.loads(tool_call.function.arguments)

                # Execute the tool (with dependency overrides if configured)
                result = await tool.execute(
                    _overrides=self.dependency_overrides, **args
                )

                # Check if the tool is marked as ephemeral (can be bool or int for keep count)
                is_ephemeral = bool(tool.ephemeral)  # Convert int to bool (2 -> True)

                tool_message = ToolMessage(
                    tool_call_id=tool_call.id,
                    tool_name=tool_name,
                    content=result,
                    is_error=False,
                    ephemeral=is_ephemeral,
                )

                # Set span output
                if Laminar is not None:
                    Laminar.set_span_output(
                        {
                            "result": (
                                result[:500]
                                if isinstance(result, str)
                                else str(result)[:500]
                            )
                        }
                    )

                return tool_message

            except json.JSONDecodeError as e:
                error_msg = f"Error parsing arguments: {e}"
                if Laminar is not None:
                    Laminar.set_span_output({"error": error_msg})
                return ToolMessage(
                    tool_call_id=tool_call.id,
                    tool_name=tool_name,
                    content=error_msg,
                    is_error=True,
                )
            except TaskComplete as e:
                # Capture TaskComplete to re-raise after span closes cleanly
                if Laminar is not None:
                    Laminar.set_span_output({"task_complete": True, "message": str(e)})
                task_complete_exception = e
            except Exception as e:
                error_msg = f"Error executing tool: {e}"
                if Laminar is not None:
                    Laminar.set_span_output({"error": error_msg})
                return ToolMessage(
                    tool_call_id=tool_call.id,
                    tool_name=tool_name,
                    content=error_msg,
                    is_error=True,
                )

        # Re-raise TaskComplete after span has closed cleanly
        if task_complete_exception is not None:
            raise task_complete_exception

        # This should be unreachable - all code paths either return or raise
        raise RuntimeError("Unexpected code path in _execute_tool_call")

    def _extract_screenshot(self, tool_message: ToolMessage) -> str | None:
        """Extract screenshot base64 from a tool message if present.

        Browser tools may return ContentPartImageParam with screenshots.
        This method extracts the base64 data from such messages.

        Args:
                tool_message: The tool message to extract screenshot from.

        Returns:
                Base64-encoded screenshot string, or None if no screenshot.
        """
        content = tool_message.content

        # If content is a string, no screenshot
        if isinstance(content, str):
            return None

        # If content is a list of content parts, look for images
        if isinstance(content, list):
            for part in content:
                # Check if it's an image content part
                if hasattr(part, "type") and part.type == "image_url":
                    image_url = getattr(part, "image_url", None)
                    if image_url:
                        url = getattr(image_url, "url", "") or image_url.get("url", "")
                        if url.startswith("data:image/png;base64,"):
                            return url.split(",", 1)[1]
                        elif url.startswith("data:image/jpeg;base64,"):
                            return url.split(",", 1)[1]
                # Handle dict format
                elif isinstance(part, dict) and part.get("type") == "image_url":
                    image_url = part.get("image_url", {})
                    url = image_url.get("url", "")
                    if url.startswith("data:image/png;base64,"):
                        return url.split(",", 1)[1]
                    elif url.startswith("data:image/jpeg;base64,"):
                        return url.split(",", 1)[1]

        return None

    async def _invoke_llm(self) -> ChatInvokeCompletion:
        """Invoke the LLM with current messages and tools.

        Includes retry logic with exponential backoff for LLM errors
        """
        last_error: Exception | None = None

        for attempt in range(self.llm_max_retries):
            try:
                response = await self.llm.ainvoke(
                    messages=self._context.get_messages(),
                    tools=self.tool_definitions if self.tools else None,
                    tool_choice=self.tool_choice if self.tools else None,
                )

                # Track token usage
                if response.usage:
                    self._token_cost.add_usage(self.llm.model, response.usage)

                return response

            except ModelRateLimitError as e:
                # Rate limit errors are always retryable
                last_error = e
                if attempt < self.llm_max_retries - 1:
                    delay = min(
                        self.llm_retry_base_delay * (2**attempt),
                        self.llm_retry_max_delay,
                    )
                    jitter = random.uniform(
                        0, delay * 0.1
                    )  # 10% jitter (matches browser-use)
                    total_delay = delay + jitter
                    logger.warning(
                        f"⚠️ Got rate limit error, retrying in {total_delay:.1f}s... "
                        f"(attempt {attempt + 1}/{self.llm_max_retries})"
                    )
                    await asyncio.sleep(total_delay)
                    continue
                raise

            except ModelProviderError as e:
                last_error = e
                # Check if status code is retryable
                is_retryable = (
                    hasattr(e, "status_code")
                    and e.status_code in self.llm_retryable_status_codes
                )
                if is_retryable and attempt < self.llm_max_retries - 1:
                    delay = min(
                        self.llm_retry_base_delay * (2**attempt),
                        self.llm_retry_max_delay,
                    )
                    jitter = random.uniform(
                        0, delay * 0.1
                    )  # 10% jitter (matches browser-use)
                    total_delay = delay + jitter
                    logger.warning(
                        f"⚠️ Got {e.status_code} error, retrying in {total_delay:.1f}s... "
                        f"(attempt {attempt + 1}/{self.llm_max_retries})"
                    )
                    await asyncio.sleep(total_delay)
                    continue
                # Non-retryable or exhausted retries
                raise

            except Exception as e:
                # Handle timeout and connection errors (retryable)
                last_error = e
                error_message = str(e).lower()
                is_timeout = "timeout" in error_message or "cancelled" in error_message
                is_connection_error = (
                    "connection" in error_message or "connect" in error_message
                )

                if (
                    is_timeout or is_connection_error
                ) and attempt < self.llm_max_retries - 1:
                    delay = min(
                        self.llm_retry_base_delay * (2**attempt),
                        self.llm_retry_max_delay,
                    )
                    jitter = random.uniform(0, delay * 0.1)
                    total_delay = delay + jitter
                    error_type = "timeout" if is_timeout else "connection error"
                    logger.warning(
                        f"⚠️ Got {error_type}, retrying in {total_delay:.1f}s... "
                        f"(attempt {attempt + 1}/{self.llm_max_retries})"
                    )
                    await asyncio.sleep(total_delay)
                    continue
                # Non-retryable error
                raise

        if last_error is not None:
            raise last_error
        raise RuntimeError("Retry loop completed without return or exception")

    async def _generate_max_iterations_summary(self) -> str:
        """Generate a summary of what was accomplished when max iterations is reached.

        Uses the LLM to summarize the conversation history and actions taken.
        """
        # Build a summary prompt
        summary_prompt = """The task has reached the maximum number of steps allowed.
Please provide a concise summary of:
1. What was accomplished so far
2. What actions were taken
3. What remains incomplete (if anything)
4. Any partial results or findings

Keep the summary brief but informative."""

        # Add the summary request as a user message temporarily
        self._context.add_message(UserMessage(content=summary_prompt))

        try:
            # Invoke LLM without tools to get a summary response
            response = await self.llm.ainvoke(
                messages=self._context.get_messages(),
                tools=None,
                tool_choice=None,
            )
            summary = response.content or "Unable to generate summary."
        except Exception as e:
            logger.warning(f"Failed to generate max iterations summary: {e}")
            summary = f"Task stopped after {self.max_iterations} iterations. Unable to generate summary due to error."
        finally:
            # Remove the temporary summary prompt
            self._context.remove_message_at()

        return f"[Max iterations reached]\n\n{summary}"

    async def _get_incomplete_todos_prompt(self) -> str | None:
        """Hook for subclasses to check for incomplete todos before finishing.

        This method is called when the LLM is about to stop (no more tool calls in CLI mode,
        or done tool called in autonomous mode).

        The prompt should ask the LLM to:
        1. Continue working on incomplete tasks
        2. Mark completed tasks as done
        3. Revise the todo list if tasks are no longer relevant
        """
        return None

    async def _maintain_context(self, response: ChatInvokeCompletion) -> None:
        """Apply sliding window and then compaction using the best token signal."""
        did_slide = False
        if self._context.sliding_window_messages is not None:
            did_slide = await self._context.apply_sliding_window_by_messages(
                keep_count=self._context.sliding_window_messages,
            )
            # Alternative strategy (round-based):
            # did_slide = await self._context.apply_sliding_window_by_rounds(
            #     keep_rounds=self._context.sliding_window_messages,
            # )

        if did_slide:
            await self._context.check_and_compact_estimated(self.llm)
        else:
            await self._context.check_and_compact(self.llm, response.usage)

    def _split_persistent_instruction_prefix(
        self,
    ) -> tuple[list[BaseMessage], list[BaseMessage]]:
        """Split history into persistent prefix and compactable conversation.

        Leading system/developer messages are long-lived behavior constraints.
        Preserve them verbatim, and only compact the remaining conversation tail.
        """
        prefix: list[BaseMessage] = []
        split_index = 0

        for msg in self._context.get_messages():
            if isinstance(msg, (SystemMessage, DeveloperMessage)):
                prefix.append(msg)
                split_index += 1
                continue
            break

        return prefix, list(self._context.get_messages()[split_index:])

    def _is_context_overflow_error(self, error: Exception) -> bool:
        """Detect provider errors caused by context/token length limits."""
        message = str(error).lower()
        status_code = getattr(error, "status_code", None)

        overflow_markers = (
            "context length",
            "maximum context",
            "prompt too long",
            "too many tokens",
            "maximum tokens",
            "context window",
            "token limit",
            "input is too long",
        )

        # 413 is a strong signal for payload/context overflow.
        if status_code == 413:
            return True

        # Some providers return 400/422 for context overflow; verify by message.
        if status_code in {400, 422} and any(marker in message for marker in overflow_markers):
            return True

        return any(marker in message for marker in overflow_markers)

    async def _compact_messages_now(self) -> bool:
        """Force compaction immediately and replace current message history."""
        if self._context._compaction_service is None:
            return False
        if not self._context._compaction_service.config.enabled:
            return False
        if not self._context.get_messages():
            return False

        prefix_messages, compactable_messages = (
            self._split_persistent_instruction_prefix()
        )
        if not compactable_messages:
            return False

        try:
            result = await self._context._compaction_service.compact(
                compactable_messages, self.llm
            )
        except Exception as e:
            logger.warning(f"Failed to compact messages for recovery: {e}")
            return False

        self._context.replace_messages([
            *prefix_messages,
            *self._context._compaction_service.create_compacted_messages(result.summary or ""),
        ])
        return True

    async def preflight_model_switch(
        self,
        target_model: str,
        utilization_limit: float = 0.95,
    ) -> ModelSwitchPreflightResult:
        """Check whether current context is safe for a target model.

        If usage is too high and compaction is enabled, compact once and reassess.
        """
        if self._context._compaction_service is None:
            return ModelSwitchPreflightResult(
                ok=True,
                target_model=target_model,
                estimated_tokens=0,
                threshold=0,
                context_limit=0,
                threshold_utilization=0.0,
                compacted=False,
                reason="Compaction service unavailable; skipping preflight.",
            )

        assessment = await self._context._compaction_service.assess_messages_for_model(
            self._context.get_messages(),
            target_model,
        )

        estimated_tokens = int(assessment["estimated_tokens"])
        threshold = int(assessment["threshold"])
        context_limit = int(assessment["context_limit"])
        threshold_utilization = float(assessment["threshold_utilization"])

        if threshold == 0 or threshold_utilization < utilization_limit:
            return ModelSwitchPreflightResult(
                ok=True,
                target_model=target_model,
                estimated_tokens=estimated_tokens,
                threshold=threshold,
                context_limit=context_limit,
                threshold_utilization=threshold_utilization,
                compacted=False,
            )

        if not self._context._compaction_service.config.enabled:
            return ModelSwitchPreflightResult(
                ok=False,
                target_model=target_model,
                estimated_tokens=estimated_tokens,
                threshold=threshold,
                context_limit=context_limit,
                threshold_utilization=threshold_utilization,
                compacted=False,
                reason="Context too large for target model and compaction is disabled.",
            )

        compacted = await self._compact_messages_now()
        if not compacted:
            return ModelSwitchPreflightResult(
                ok=False,
                target_model=target_model,
                estimated_tokens=estimated_tokens,
                threshold=threshold,
                context_limit=context_limit,
                threshold_utilization=threshold_utilization,
                compacted=False,
                reason="Context too large and automatic compaction failed.",
            )

        reassessment = await self._context._compaction_service.assess_messages_for_model(
            self._context.get_messages(),
            target_model,
        )

        new_estimated_tokens = int(reassessment["estimated_tokens"])
        new_threshold = int(reassessment["threshold"])
        new_context_limit = int(reassessment["context_limit"])
        new_utilization = float(reassessment["threshold_utilization"])

        if new_threshold > 0 and new_utilization >= utilization_limit:
            return ModelSwitchPreflightResult(
                ok=False,
                target_model=target_model,
                estimated_tokens=new_estimated_tokens,
                threshold=new_threshold,
                context_limit=new_context_limit,
                threshold_utilization=new_utilization,
                compacted=True,
                reason=(
                    "Context is still too large for target model after compaction. "
                    "Try a larger-context model."
                ),
            )

        return ModelSwitchPreflightResult(
            ok=True,
            target_model=target_model,
            estimated_tokens=new_estimated_tokens,
            threshold=new_threshold,
            context_limit=new_context_limit,
            threshold_utilization=new_utilization,
            compacted=True,
            reason="Context compacted before model switch.",
        )

    @observe(name="agent_query")
    async def query(self, message: str) -> str:
        """
        Send a message to the agent and get a response.

        Can be called multiple times for follow-up questions - message history
        is preserved between calls. System prompt is automatically added on
        first call.

        When compaction is enabled, the agent will automatically compress the
        conversation history when token usage exceeds the configured threshold.
        After compaction, the conversation continues from the summary.

        Args:
            message: The user message.

        Returns:
            The agent's response text.
        """
        # Add system prompt on first message
        if not self._context and self.system_prompt:
            # Cache the static system prompt when provider supports it (Anthropic).
            self._context.add_message(SystemMessage(content=self.system_prompt, cache=True))

        # Add the user message
        self._context.add_message(UserMessage(content=message))

        iterations = 0
        tool_calls_made = 0
        overflow_recovery_attempted = False
        incomplete_todos_prompted = (
            False  # Track if we've already prompted about incomplete todos
        )

        while iterations < self.max_iterations:
            iterations += 1

            # Destroy ephemeral messages from previous iteration before LLM sees them again
            # 清理一些历史冗余的工具输出结果，总体思想是对每种工具只保留最近N条结果
            self._destroy_ephemeral_messages()

            # Invoke the LLM
            try:
                response = await self._invoke_llm()
            except Exception as e:
                if (
                    not overflow_recovery_attempted
                    and self._is_context_overflow_error(e)
                ):
                    overflow_recovery_attempted = True
                    compacted = await self._compact_messages_now()
                    if compacted:
                        logger.warning(
                            "Recovered from context overflow by compacting history and retrying once."
                        )
                        continue
                raise

            # Add assistant message to history
            assistant_msg = AssistantMessage(
                content=response.content,
                tool_calls=response.tool_calls if response.tool_calls else None,
            )
            self._context.add_message(assistant_msg)

            # If no tool calls, check if should finish
            if not response.has_tool_calls:
                if not self.require_done_tool:
                    # 这种模型下，只要没有调用工具，就视为结束
                    # CLI mode: LLM stopped calling tools, check for incomplete todos before finishing
                    if not incomplete_todos_prompted:
                        # 检查未完成todo，防止llm忘了以前说过的todo（自动纠正）
                        # 留了一个未完成的hook，可以根据自己的plan格式自己实现
                        incomplete_prompt = await self._get_incomplete_todos_prompt()
                        if incomplete_prompt:
                            # 如果有没做完的，修改标识符，更新messages，继续循环
                            # 是一个最低程度的自我纠正，只会重复一次
                            incomplete_todos_prompted = True
                            self._context.add_message(
                                UserMessage(content=incomplete_prompt)
                            )
                            continue  # Give the LLM a chance to handle incomplete todos

                    # All done - return the response
                    # todo: 看看代码，大概就是做一下压缩
                    await self._maintain_context(response)
                    # 返回最终文本
                    return response.content or ""
                # Autonomous mode: require done tool, continue loop
                # LLM必须显式调用一个done tool，否则即使没有tool call，也不会退出
                continue

            # 如果有tool calls，执行工具
            # Execute all tool calls
            for tool_call in response.tool_calls:
                tool_calls_made += 1
                try:
                    # todo: 看看这个代码
                    tool_result = await self._execute_tool_call(tool_call)
                    self._context.add_message(tool_result)
                except TaskComplete as e:
                    # 工具可以通过抛异常来结束agent
                    self._context.add_message(
                        ToolMessage(
                            tool_call_id=tool_call.id,
                            tool_name=tool_call.function.name,
                            content=f"Task completed: {e.message}",
                            is_error=False,
                        )
                    )
                    return e.message

            # Check for compaction after tool execution
            await self._maintain_context(response)

        # Max iterations reached - generate summary of what was accomplished
        return await self._generate_max_iterations_summary()

    @observe(name="agent_query_stream")
    async def query_stream(
        self, message: str | list[ContentPartTextParam | ContentPartImageParam]
    ) -> AsyncIterator[AgentEvent]:
        """
        Send a message to the agent and stream events as they occur.

        Yields events for each step of the agent's execution, providing
        visibility into tool calls and intermediate results.

        Args:
            message: The user message. Can be a string or a list of content parts
                for multi-modal input (text + images).

        Yields:
            AgentEvent instances for each step:
            - TextEvent: When the assistant produces text
            - ThinkingEvent: When the model produces thinking content
            - ToolCallEvent: When a tool is being called
            - ToolResultEvent: When a tool returns a result
            - FinalResponseEvent: The final response (always last)

        Example:
            async for event in agent.query_stream("Schedule a meeting"):
                match event:
                    case ToolCallEvent(tool=name, args=args):
                        print(f"Calling {name}")
                    case ToolResultEvent(tool=name, result=result):
                        print(f"{name} returned: {result[:50]}")
                    case FinalResponseEvent(content=text):
                        print(f"Done: {text}")
        """
        # Add system prompt on first message
        if not self._context and self.system_prompt:
            # Cache the static system prompt when provider supports it (Anthropic).
            self._context.add_message(SystemMessage(content=self.system_prompt, cache=True))

        # Add the user message (supports both string and multi-modal content)
        self._context.add_message(UserMessage(content=message))

        iterations = 0
        overflow_recovery_attempted = False
        incomplete_todos_prompted = (
            False  # Track if already prompted about incomplete todos
        )

        while iterations < self.max_iterations:
            iterations += 1

            # Destroy ephemeral messages from previous iteration before LLM sees them again
            # _destroy_ephemeral_messages暂时留了一个入口hook，可以根据自己的plan工具自己实现
            self._destroy_ephemeral_messages()

            # Invoke the LLM
            try:
                response = await self._invoke_llm()
            except Exception as e:
                if (
                    not overflow_recovery_attempted
                    and self._is_context_overflow_error(e)
                ):
                    overflow_recovery_attempted = True
                    compacted = await self._compact_messages_now()
                    if compacted:
                        yield HiddenUserMessageEvent(
                            content=(
                                "Context exceeded model window. "
                                "Automatically compacted history and retried."
                            )
                        )
                        continue
                raise

            # Check for thinking content and yield it
            if response.thinking:
                yield ThinkingEvent(content=response.thinking)

            # Add assistant message to history
            assistant_msg = AssistantMessage(
                content=response.content,
                tool_calls=response.tool_calls if response.tool_calls else None,
            )
            self._context.add_message(assistant_msg)

            # If no tool calls, check if should finish
            if not response.has_tool_calls:
                if not self.require_done_tool:
                    # CLI mode: LLM stopped calling tools, check for incomplete todos before finishing
                    if not incomplete_todos_prompted:
                        incomplete_prompt = await self._get_incomplete_todos_prompt()
                        if incomplete_prompt:
                            incomplete_todos_prompted = True
                            self._context.add_message(
                                UserMessage(content=incomplete_prompt)
                            )
                            yield HiddenUserMessageEvent(content=incomplete_prompt)
                            continue  # Give the LLM a chance to handle incomplete todos

                    # All done - return the response
                    await self._maintain_context(response)
                    if response.content:
                        yield TextEvent(content=response.content)
                    yield FinalResponseEvent(content=response.content or "")
                    return
                # Autonomous mode: require done tool, yield text and continue loop
                if response.content:
                    yield TextEvent(content=response.content)
                continue

            # Yield text content if present alongside tool calls
            if response.content:
                yield TextEvent(content=response.content)

            # Execute all tool calls, yielding events for each
            step_number = 0
            for tool_call in response.tool_calls:
                step_number += 1
                tool_name = tool_call.function.name

                # Yield the tool call event
                try:
                    args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    args = {"_raw": tool_call.function.arguments}

                # Emit step start event
                yield StepStartEvent(
                    step_id=tool_call.id,
                    title=tool_name,
                    step_number=step_number,
                )

                yield ToolCallEvent(
                    tool=tool_name,
                    args=args,
                    tool_call_id=tool_call.id,
                    display_name=tool_name,
                )

                # Execute the tool
                step_start_time = time.time()
                try:
                    tool_result = await self._execute_tool_call(tool_call)
                    self._context.add_message(tool_result)

                    # Extract screenshot if present (for browser tools)
                    screenshot_base64 = self._extract_screenshot(tool_result)

                    # Yield the tool result event
                    yield ToolResultEvent(
                        tool=tool_name,
                        result=tool_result.text,
                        tool_call_id=tool_call.id,
                        is_error=tool_result.is_error,
                        screenshot_base64=screenshot_base64,
                    )

                    # Emit step complete event
                    step_duration_ms = (time.time() - step_start_time) * 1000
                    yield StepCompleteEvent(
                        step_id=tool_call.id,
                        status="error" if tool_result.is_error else "completed",
                        duration_ms=step_duration_ms,
                    )
                except TaskComplete as e:
                    # done_autonomous already validates todos before raising TaskComplete,
                    # so can complete immediately
                    self._context.add_message(
                        ToolMessage(
                            tool_call_id=tool_call.id,
                            tool_name=tool_call.function.name,
                            content=f"Task completed: {e.message}",
                            is_error=False,
                        )
                    )
                    yield ToolResultEvent(
                        tool=tool_call.function.name,
                        result=f"Task completed: {e.message}",
                        tool_call_id=tool_call.id,
                        is_error=False,
                    )
                    yield FinalResponseEvent(content=e.message)
                    return

            # Check for compaction after tool execution
            await self._maintain_context(response)

        # Max iterations reached - generate summary of what was accomplished
        summary = await self._generate_max_iterations_summary()
        yield FinalResponseEvent(content=summary)

    @observe(name="agent_query_stream_delta")
    async def query_stream_delta(
        self, message: str | list[ContentPartTextParam | ContentPartImageParam]
    ) -> AsyncIterator[AgentEvent]:
        """
        流式查询 - Token级别的实时输出

        智能流式策略：
        - 先尝试流式输出，逐 token 实时显示
        - 一旦检测到工具调用，立即取消流式，改用非流式获取完整响应

        这样可以避免：
        1. 重复调用 LLM
        2. 流式 API 下工具调用参数不完整的问题

        Args:
            message: 用户消息

        Yields:
            AgentEvent，包含：
            - TextDeltaEvent: 增量文本（打字机效果）
            - TextEvent: 完整文本（检测到工具调用时）
            - ToolCallEvent: 工具调用
            - ToolResultEvent: 工具结果
            - FinalResponseEvent: 最终响应
        """
        from bu_agent_sdk.agent.events import TextDeltaEvent

        # Add system prompt on first message
        if not self._messages and self.system_prompt:
            self._messages.append(SystemMessage(content=self.system_prompt, cache=True))

        # Add the user message
        self._messages.append(UserMessage(content=message))

        iterations = 0
        incomplete_todos_prompted = False

        while iterations < self.max_iterations:
            iterations += 1

            # Destroy ephemeral messages
            self._destroy_ephemeral_messages()

            # ========== 尝试流式调用 ==========
            accumulated_content = ""
            detected_tool_call = False
            response_usage = None

            try:
                # 先尝试流式输出
                stream_iter = self.llm.astream(
                    messages=self._messages,
                    tools=self.tool_definitions if self.tools else None,
                    tool_choice=self.tool_choice if self.tools else None,
                )

                async for chunk in stream_iter:
                    # 累积增量内容
                    if chunk.delta:
                        accumulated_content += chunk.delta
                        yield TextDeltaEvent(delta=chunk.delta)

                    # 检测到工具调用，立即中断流式
                    if chunk.has_tool_calls:
                        detected_tool_call = True
                        # 取消流式，准备用非流式重新获取
                        break

                    # 保存 usage
                    if chunk.usage:
                        response_usage = chunk.usage

                # 如果检测到工具调用，用非流式获取完整响应
                if detected_tool_call:
                    # 使用非流式 API 获取完整的工具调用信息
                    response = await self.llm.ainvoke(
                        messages=self._messages,
                        tools=self.tool_definitions if self.tools else None,
                        tool_choice=self.tool_choice if self.tools else None,
                    )
                    final_content = response.content or ""
                    has_tools = response.has_tool_calls
                    tool_calls = response.tool_calls
                    if response.usage:
                        response_usage = response.usage
                else:
                    # 纯文本输出，使用流式累积的内容
                    final_content = accumulated_content
                    has_tools = False
                    tool_calls = None

            except Exception as e:
                logger.error(f"Error in stream_delta: {e}", exc_info=True)
                raise

            # Track token usage
            if response_usage:
                self._token_cost.add_usage(self.llm.model, response_usage)

            # Add assistant message to history
            assistant_msg = AssistantMessage(
                content=final_content,
                tool_calls=tool_calls if has_tools else None,
            )
            self._messages.append(assistant_msg)

            # If no tool calls, check if should finish
            if not has_tools:
                if not self.require_done_tool:
                    if not incomplete_todos_prompted:
                        incomplete_prompt = await self._get_incomplete_todos_prompt()
                        if incomplete_prompt:
                            incomplete_todos_prompted = True
                            self._messages.append(UserMessage(content=incomplete_prompt))
                            yield HiddenUserMessageEvent(content=incomplete_prompt)
                            continue

                    await self._check_and_compact_with_usage(response_usage)
                    yield FinalResponseEvent(content=final_content or "")
                    return

                # Autonomous mode: continue loop
                continue

            # Execute all tool calls
            step_number = 0
            for tool_call in tool_calls:
                step_number += 1
                tool_name = tool_call.function.name

                try:
                    args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    args = {"_raw": tool_call.function.arguments}

                yield StepStartEvent(
                    step_id=tool_call.id,
                    title=tool_name,
                    step_number=step_number,
                )

                yield ToolCallEvent(
                    tool=tool_name,
                    args=args,
                    tool_call_id=tool_call.id,
                    display_name=tool_name,
                )

                step_start_time = time.time()
                try:
                    tool_result = await self._execute_tool_call(tool_call)
                    self._messages.append(tool_result)

                    screenshot_base64 = self._extract_screenshot(tool_result)

                    yield ToolResultEvent(
                        tool=tool_name,
                        result=tool_result.text,
                        tool_call_id=tool_call.id,
                        is_error=tool_result.is_error,
                        screenshot_base64=screenshot_base64,
                    )

                    step_duration_ms = (time.time() - step_start_time) * 1000
                    yield StepCompleteEvent(
                        step_id=tool_call.id,
                        status="error" if tool_result.is_error else "completed",
                        duration_ms=step_duration_ms,
                    )
                except TaskComplete as e:
                    self._messages.append(
                        ToolMessage(
                            tool_call_id=tool_call.id,
                            tool_name=tool_call.function.name,
                            content=f"Task completed: {e.message}",
                            is_error=False,
                        )
                    )
                    yield ToolResultEvent(
                        tool=tool_call.function.name,
                        result=f"Task completed: {e.message}",
                        tool_call_id=tool_call.id,
                        is_error=False,
                    )
                    yield FinalResponseEvent(content=e.message)
                    return

            # Check for compaction after tool execution
            await self._check_and_compact_with_usage(response_usage)

        # Max iterations reached
        summary = await self._generate_max_iterations_summary()
        yield FinalResponseEvent(content=summary)

    async def _check_and_compact_with_usage(self, usage):
        """Check and compact using provided usage info"""
        if self._compaction_service is None:
            return

        if usage:
            self._compaction_service.update_usage(usage)

        new_messages, result = await self._compaction_service.check_and_compact(
            self._messages,
            self.llm,
        )

        if result.compacted:
            self._messages = list(new_messages)
