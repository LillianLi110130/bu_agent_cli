"""
Compaction service for managing conversation context.

This service monitors token usage and automatically compresses conversation
history when it approaches the model's context window limit.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from uuid import uuid4

from agent_core.agent.budget import ContextBudgetEngine
from agent_core.agent.compaction.models import (
    CompactionConfig,
    CompactionResult,
    CompactionWorkingState,
    TokenUsage,
)
from agent_core.llm.messages import (
    AssistantMessage,
    BaseMessage,
    UserMessage,
)
if TYPE_CHECKING:
    from agent_core.llm.base import BaseChatModel
    from agent_core.llm.views import ChatInvokeUsage
    from agent_core.tokens import TokenCost

log = logging.getLogger(__name__)


@dataclass
class CompactionService:
    """Service for managing conversation context through compaction.

    The service monitors token usage after each LLM response and triggers
    compaction when the threshold is exceeded. During compaction:
    1. The conversation history is sent to an LLM with a summary prompt
    2. The LLM generates a structured summary
    3. The entire message history is replaced with the summary

    The threshold is calculated dynamically based on the model's context window:
    threshold = context_window * threshold_ratio

    Attributes:
            config: Configuration for compaction behavior.
            llm: The language model to use for generating summaries.
                 If None, must be set before calling check_and_compact.
            token_cost: TokenCost service for fetching model context limits.
    """

    config: CompactionConfig = field(default_factory=CompactionConfig)
    llm: BaseChatModel | None = None
    token_cost: TokenCost | None = None
    budget_engine: ContextBudgetEngine | None = None

    # Internal state
    _last_usage: TokenUsage = field(default_factory=TokenUsage, repr=False)

    def __post_init__(self) -> None:
        if self.budget_engine is None:
            self.budget_engine = ContextBudgetEngine(
                config=self.config,
                token_cost=self.token_cost,
            )

    def update_usage(self, usage: ChatInvokeUsage | None) -> None:
        """Update the tracked token usage from a response.

        Args:
                usage: The usage information from the last LLM response.
        """
        self._last_usage = TokenUsage.from_usage(usage)

    def estimate_tokens_for_messages(self, messages: list[BaseMessage]) -> int:
        """Estimate token usage for a message history."""
        if self.budget_engine is None:
            return 0
        return self.budget_engine.estimate_tokens_for_messages(messages)

    async def assess_messages_for_model(
        self,
        messages: list[BaseMessage],
        model: str,
    ) -> dict[str, int | float | bool | str | None]:
        """Assess whether a message history is likely to fit a target model."""
        if self.budget_engine is None:
            return {
                "context_limit": 0,
                "threshold": 0,
                "estimated_tokens": 0,
                "threshold_utilization": 0.0,
                "context_utilization": 0.0,
                "exceeds_threshold": False,
                "exceeds_context_limit": False,
                "warn_threshold": 0,
                "hard_threshold": 0,
                "trigger": None,
            }
        assessment = await self.budget_engine.assess(model=model, messages=messages)
        return {
            "context_limit": assessment.context_limit,
            "threshold": assessment.compact_threshold,
            "estimated_tokens": assessment.estimated_tokens,
            "threshold_utilization": assessment.threshold_utilization,
            "context_utilization": assessment.context_utilization,
            "exceeds_threshold": assessment.needs_compaction,
            "exceeds_context_limit": assessment.context_utilization >= 1.0,
            "warn_threshold": assessment.warn_threshold,
            "hard_threshold": assessment.hard_threshold,
            "trigger": assessment.trigger,
        }

    async def get_model_context_limit(self, model: str) -> int:
        """Get the context window limit for a model."""
        if self.budget_engine is None:
            return 0
        return await self.budget_engine.get_model_context_limit(model)

    async def get_threshold_for_model(self, model: str) -> int:
        """Get the compaction threshold for a specific model."""
        if self.budget_engine is None:
            return 0
        assessment = await self.budget_engine.assess(model=model, messages=[])
        return assessment.compact_threshold

    async def should_compact(self, model: str) -> bool:
        """Check if compaction should be triggered based on current token usage.

        Returns:
                True if token usage exceeds the threshold and compaction is enabled.
        """
        if not self.config.enabled:
            return False

        if self.budget_engine is None:
            return False

        assessment = await self.budget_engine.assess(model=model, messages=[])
        threshold = assessment.compact_threshold
        should = self._last_usage.total_tokens >= threshold

        if should:
            log.info(
                f"Compaction triggered: {self._last_usage.total_tokens} tokens >= {threshold} threshold "
                f"(model: {model}, ratio: {self.budget_engine.compact_threshold_ratio})"
            )

        return should

    async def compact(
        self,
        messages: list[BaseMessage],
        llm: BaseChatModel | None = None,
    ) -> CompactionResult:
        """Perform compaction on the message history.

        This method:
        1. Prepares the messages for summarization (removing pending tool calls)
        2. Appends the summary prompt as a user message
        3. Calls the LLM to generate a summary
        4. Extracts the summary and returns it

        Args:
                messages: The current message history to compact.
                llm: Optional LLM to use for summarization. Falls back to self.llm.

        Returns:
                CompactionResult containing the summary and token information.

        Raises:
                ValueError: If no LLM is available for summarization.
        """
        model = llm or self.llm
        if model is None:
            raise ValueError("No LLM available for compaction. Provide an LLM or set self.llm.")

        original_tokens = self._last_usage.total_tokens
        threshold = await self.get_threshold_for_model(model.model)

        log.info(
            f"Token usage {original_tokens} has exceeded the threshold of "
            f"{threshold}. Performing compaction."
        )

        # Prepare messages for summarization
        prepared_messages = self._prepare_messages_for_summary(messages)

        # Add the summary prompt
        prepared_messages.append(UserMessage(content=self.config.summary_prompt))

        # Generate the summary
        response = await model.ainvoke(messages=prepared_messages)

        summary_text = response.content or ""

        # Extract summary from tags if present
        extracted_summary = self._extract_summary(summary_text)
        extracted_working_state = self._extract_working_state(summary_text)
        checkpoint_ref = self._extract_checkpoint_ref(summary_text) or self._generate_checkpoint_ref()

        new_tokens = response.usage.completion_tokens if response.usage else 0

        log.info(f"Compaction complete. New token usage: {new_tokens}")

        return CompactionResult(
            compacted=True,
            original_tokens=original_tokens,
            new_tokens=new_tokens,
            summary=extracted_summary,
            working_state=extracted_working_state,
            checkpoint_ref=checkpoint_ref,
        )

    async def check_and_compact(
        self,
        messages: list[BaseMessage],
        llm: BaseChatModel | None = None,
    ) -> tuple[list[BaseMessage], CompactionResult]:
        """Check token usage and compact if threshold exceeded.

        This is the main entry point for the compaction service. It checks
        if compaction is needed and performs it if so.

        Args:
                messages: The current message history.
                llm: Optional LLM to use for summarization.

        Returns:
                A tuple of (new_messages, result) where new_messages is either
                the original messages (if no compaction) or a single summary
                message (if compacted).
        """
        model = llm or self.llm
        if model is None:
            return messages, CompactionResult(compacted=False)

        if not await self.should_compact(model.model):
            return messages, CompactionResult(compacted=False)

        result = await self.compact(messages, llm)

        # Replace entire history with summary as a user message
        # This matches the Anthropic SDK behavior
        new_messages: list[BaseMessage] = [
            UserMessage(content=result.summary or ""),
        ]

        return new_messages, result

    def create_compacted_messages(self, result: CompactionResult | str) -> list[BaseMessage]:
        """Create a new message list from a summary.

        Args:
                result: A compaction result or raw summary string.

        Returns:
                A list containing a single user message with the structured working set.
        """
        if isinstance(result, str):
            result = CompactionResult(
                compacted=True,
                summary=result,
                working_state=CompactionWorkingState(),
            )
        return [UserMessage(content=self.render_compacted_working_set(result))]

    def render_compacted_working_set(self, result: CompactionResult) -> str:
        """Render a structured working set message for reinjection into context."""
        working_state = result.working_state or CompactionWorkingState()
        lines = [
            "[Compacted Working Set]",
            "",
            "Summary:",
            result.summary or "",
            "",
            f"User Goal: {working_state.user_goal or '(unknown)'}",
            "User Constraints:",
            *self._render_list(working_state.user_constraints),
            "Confirmed Conclusions:",
            *self._render_list(working_state.confirmed_conclusions),
            "Files Reviewed:",
            *self._render_list(working_state.files_reviewed),
            "Files Modified:",
            *self._render_list(working_state.files_modified),
            "Failed Attempts:",
            *self._render_list(working_state.failed_attempts),
            "Remaining Actions:",
            *self._render_list(working_state.remaining_actions),
            "Artifact Refs:",
            *self._render_list(working_state.artifact_refs),
            "Recent History Notes:",
            *self._render_list(working_state.recent_history_notes),
            f"Checkpoint Ref: {result.checkpoint_ref or '(none)'}",
        ]
        return "\n".join(lines).strip()

    def _prepare_messages_for_summary(
        self,
        messages: list[BaseMessage],
    ) -> list[BaseMessage]:
        """Prepare messages for summarization.

        This removes tool_calls from the last assistant message to avoid
        API errors (tool_use requires tool_result which we won't have).

        Args:
                messages: The original message history.

        Returns:
                A cleaned copy of the messages suitable for summarization.
        """
        if not messages:
            return []

        # Make a copy to avoid modifying the original
        prepared: list[BaseMessage] = []

        for i, msg in enumerate(messages):
            is_last = i == len(messages) - 1

            if is_last and isinstance(msg, AssistantMessage) and msg.tool_calls:
                # Remove tool_calls from the last assistant message
                # Keep the content if there is any text
                if msg.content:
                    prepared.append(
                        AssistantMessage(
                            content=msg.content,
                            tool_calls=None,
                        )
                    )
                # If no content, skip this message entirely
            else:
                prepared.append(msg)

        return prepared

    def _extract_summary(self, text: str) -> str:
        """Extract summary content from <summary></summary> tags.

        If tags are not found, returns the original text.

        Args:
                text: The response text that may contain summary tags.

        Returns:
                The extracted summary or the original text.
        """
        # Try to extract content between <summary> tags
        pattern = r"<summary>(.*?)</summary>"
        match = re.search(pattern, text, re.DOTALL)

        if match:
            return match.group(1).strip()

        # No tags found, return original text
        return text.strip()

    def _extract_working_state(self, text: str) -> CompactionWorkingState:
        """Extract structured working-state JSON from a compaction response."""
        raw_state = self._extract_tag_block(text, "working_state")
        if not raw_state:
            return CompactionWorkingState()

        try:
            payload = json.loads(raw_state)
        except json.JSONDecodeError:
            log.warning("Failed to parse compaction working_state JSON.")
            return CompactionWorkingState()

        return CompactionWorkingState(
            user_goal=self._coerce_string(payload.get("user_goal")),
            user_constraints=self._coerce_string_list(payload.get("user_constraints")),
            confirmed_conclusions=self._coerce_string_list(
                payload.get("confirmed_conclusions")
            ),
            files_reviewed=self._coerce_string_list(payload.get("files_reviewed")),
            files_modified=self._coerce_string_list(payload.get("files_modified")),
            failed_attempts=self._coerce_string_list(payload.get("failed_attempts")),
            remaining_actions=self._coerce_string_list(payload.get("remaining_actions")),
            artifact_refs=self._coerce_string_list(payload.get("artifact_refs")),
            recent_history_notes=self._coerce_string_list(payload.get("recent_history_notes")),
        )

    def _extract_checkpoint_ref(self, text: str) -> str | None:
        """Extract checkpoint ref from response text."""
        checkpoint_ref = self._extract_tag_block(text, "checkpoint_ref")
        if not checkpoint_ref:
            return None
        return checkpoint_ref.strip() or None

    @staticmethod
    def _extract_tag_block(text: str, tag: str) -> str | None:
        pattern = rf"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, text, re.DOTALL)
        if not match:
            return None
        return match.group(1).strip()

    @staticmethod
    def _generate_checkpoint_ref() -> str:
        return f"compaction-inline-{uuid4().hex[:8]}"

    @staticmethod
    def _coerce_string(value: object) -> str:
        if value is None:
            return ""
        return str(value).strip()

    @staticmethod
    def _coerce_string_list(value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            items = [str(item).strip() for item in value if str(item).strip()]
            return CompactionService._dedupe_preserve_order(items)
        coerced = str(value).strip()
        return [coerced] if coerced else []

    @staticmethod
    def _render_list(values: list[str]) -> list[str]:
        if not values:
            return ["- (none)"]
        return [f"- {value}" for value in values]

    @staticmethod
    def _dedupe_preserve_order(values: list[str]) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            deduped.append(value)
        return deduped

    def merge_results(
        self,
        previous: CompactionResult | None,
        current: CompactionResult,
    ) -> CompactionResult:
        """Merge a newly compacted interval into the existing working set."""
        if previous is None:
            return current

        previous_state = previous.working_state or CompactionWorkingState()
        current_state = current.working_state or CompactionWorkingState()

        merged_state = CompactionWorkingState(
            user_goal=current_state.user_goal or previous_state.user_goal,
            user_constraints=self._dedupe_preserve_order(
                [*previous_state.user_constraints, *current_state.user_constraints]
            ),
            confirmed_conclusions=self._dedupe_preserve_order(
                [*previous_state.confirmed_conclusions, *current_state.confirmed_conclusions]
            ),
            files_reviewed=self._dedupe_preserve_order(
                [*previous_state.files_reviewed, *current_state.files_reviewed]
            ),
            files_modified=self._dedupe_preserve_order(
                [*previous_state.files_modified, *current_state.files_modified]
            ),
            failed_attempts=self._dedupe_preserve_order(
                [*previous_state.failed_attempts, *current_state.failed_attempts]
            ),
            remaining_actions=self._dedupe_preserve_order(
                [*previous_state.remaining_actions, *current_state.remaining_actions]
            ),
            artifact_refs=self._dedupe_preserve_order(
                [*previous_state.artifact_refs, *current_state.artifact_refs]
            ),
            recent_history_notes=self._dedupe_preserve_order(
                [*previous_state.recent_history_notes, *current_state.recent_history_notes]
            ),
        )
        summary_parts = [part.strip() for part in [previous.summary or "", current.summary or ""] if part.strip()]
        return CompactionResult(
            compacted=True,
            original_tokens=current.original_tokens,
            new_tokens=current.new_tokens,
            summary="\n\n".join(self._dedupe_preserve_order(summary_parts)),
            working_state=merged_state,
            checkpoint_ref=current.checkpoint_ref or previous.checkpoint_ref,
        )

    def reset(self) -> None:
        """Reset the service state.

        Clears tracked token usage and cached thresholds.
        """
        self._last_usage = TokenUsage()
        if self.budget_engine is not None:
            self.budget_engine.reset()
