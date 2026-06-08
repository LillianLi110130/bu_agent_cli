"""
Context manager for conversation messages.

Keeps an ordered list of messages (the true source of context),
and a role-based index for fast grouping.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from inspect import signature
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable, Iterator, Sequence

from agent_core.agent.budget import BudgetAssessment, ContextBudgetEngine
from agent_core.agent.compaction import (
    CompactionConfig,
    CompactionResult,
    CompactionService,
    CompactionWorkingState,
)
from agent_core.agent.context_store import ArtifactStore, CheckpointStore, WorkingStateStore
from agent_core.llm.messages import (
    AssistantMessage,
    BaseMessage,
    ContentPartTextParam,
    ToolMessage,
    UserMessage,
)

logger = logging.getLogger("agent_core.agent")

if TYPE_CHECKING:
    from agent_core.llm.base import BaseChatModel
    from agent_core.llm.views import ChatInvokeUsage
    from agent_core.tokens import TokenCost
    from agent_core.tools.decorator import Tool


ACTIVE_TODO_SNAPSHOT_HEADER = "[Active Todo List Preserved Across Context Compression]"
FINISH_GUARD_PROMPT_PREFIX = "There are unfinished todo items in the current task list."
INVALID_TOOL_CALL_RECOVERY_PREFIX = (
    "The previous model response contained invalid tool call arguments "
)
COMPACTED_WORKING_SET_HEADER = "[Compacted Working Set]"
MICROCOMPACTED_TOOL_RESULT_HEADER = "[Previous tool result microcompacted]"
DEFAULT_MICROCOMPACT_EXCLUDED_TOOL_NAMES = frozenset({"skill_view"})


@dataclass(slots=True)
class ContextSnipResult:
    """Result of deterministic runtime-context snipping."""

    removed_count: int = 0
    removed_by_kind: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class ToolMicrocompactResult:
    """Result of deterministic aged tool-result microcompaction."""

    microcompacted_count: int = 0
    skipped_count: int = 0
    artifact_created_count: int = 0


class ContextManager:
    """Manage message history with order + role-based index."""

    def __init__(
        self,
        messages: Iterable[BaseMessage] | None = None,
        sliding_window_messages: int | None = 50,
    ) -> None:
        self._messages: list[BaseMessage] = []
        self._by_role: dict[str, list[BaseMessage]] = defaultdict(list)
        self.sliding_window_messages: int | None = sliding_window_messages
        self._compaction_service: CompactionService | None = None
        self._budget_engine: ContextBudgetEngine | None = None
        self._compacted_result: CompactionResult | None = None
        self._summarized_boundary: int = 0
        self._artifact_store: ArtifactStore | None = None
        self._checkpoint_store: CheckpointStore | None = None
        self._working_state_store: WorkingStateStore | None = None
        self._artifact_refs: list[str] = []
        self._conversation_round: int = 0
        self._message_rounds: dict[int, int] = {}
        self._message_seen_by_model: set[int] = set()
        self.compaction_start_handler: Callable[[str], None] | None = None
        if messages:
            self.replace_messages(messages)

    def __iter__(self) -> Iterator[BaseMessage]:
        return iter(self._messages)

    def __len__(self) -> int:
        return len(self._messages)

    def __getitem__(self, index: int) -> BaseMessage:
        return self._messages[index]

    def get_messages(self) -> list[BaseMessage]:
        """Return a shallow copy of messages in order."""
        return list(self._messages)

    @property
    def conversation_round(self) -> int:
        """Return the current user conversation round."""
        return self._conversation_round

    def add_message(self, message: BaseMessage, *, new_user_round: bool = False) -> None:
        if new_user_round:
            self._conversation_round += 1
        self._messages.append(message)
        self._by_role[message.role].append(message)
        self._message_rounds[id(message)] = self._conversation_round

    def add_messages(self, messages: Iterable[BaseMessage]) -> None:
        for msg in messages:
            self.add_message(msg)

    def remove_message_at(self, index: int = -1) -> BaseMessage:
        msg = self._messages.pop(index)
        self._message_rounds.pop(id(msg), None)
        self._message_seen_by_model.discard(id(msg))
        role_list = self._by_role.get(msg.role)
        if role_list is not None:
            try:
                role_list.remove(msg)
            except ValueError:
                pass
        self.clear_compaction_state()
        self.invalidate_budget_baseline()
        return msg

    def clear_messages(self) -> None:
        self._messages.clear()
        self._by_role.clear()
        self._conversation_round = 0
        self._message_rounds.clear()
        self._message_seen_by_model.clear()
        self.clear_compaction_state()
        self.invalidate_budget_baseline()

    def invalidate_budget_baseline(self) -> None:
        """Drop incremental prompt baseline after in-place history rewrites."""
        if self._budget_engine is not None:
            self._budget_engine.reset()

    def clear_compaction_state(self) -> None:
        """Forget the current compacted working set and summarized boundary."""
        self._compacted_result = None
        self._summarized_boundary = 0

    @staticmethod
    def _dedupe_preserve_order(values: Iterable[str]) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for value in values:
            cleaned = str(value).strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            deduped.append(cleaned)
        return deduped

    def bind_filesystem_stores(
        self,
        *,
        artifact_store: ArtifactStore | None = None,
        checkpoint_store: CheckpointStore | None = None,
        working_state_store: WorkingStateStore | None = None,
    ) -> None:
        """Attach rollout-local filesystem stores used by context engineering."""
        self._artifact_store = artifact_store
        self._checkpoint_store = checkpoint_store
        self._working_state_store = working_state_store
        if self._working_state_store is not None:
            self._working_state_store.ensure_initialized()

    def _register_artifact_ref(self, path: str) -> str:
        self._artifact_refs = self._dedupe_preserve_order([*self._artifact_refs, path])
        return path

    def persist_tool_artifact(
        self,
        message: ToolMessage,
        *,
        force: bool = False,
        summary_text: str | None = None,
    ) -> str | None:
        """Persist a large tool result to the rollout artifact store when configured."""
        if self._artifact_store is None:
            return None
        saved_path = self._artifact_store.save_tool_message(
            message,
            force=force,
            summary_text=summary_text,
        )
        if saved_path is None:
            return None
        return self._register_artifact_ref(str(saved_path))

    def persist_image_detail_artifact(
        self,
        detail_text: str,
        *,
        source_hint: str = "",
    ) -> str | None:
        """Persist detailed image extraction text for later recovery."""
        if self._artifact_store is None:
            return None
        saved_path = self._artifact_store.save_image_detail(detail_text, source_hint=source_hint)
        return self._register_artifact_ref(str(saved_path))

    async def compact_messages(
        self,
        messages: Sequence[BaseMessage],
        llm: "BaseChatModel",
        *,
        reference_messages: Sequence[BaseMessage] | None = None,
    ) -> CompactionResult:
        """Compact a message segment, checkpointing the raw messages first when configured."""
        if self._compaction_service is None:
            raise RuntimeError("Compaction service unavailable.")

        checkpoint_record = None
        if self._checkpoint_store is not None and messages:
            checkpoint_record = self._checkpoint_store.save_messages(messages)

        compact_kwargs = {}
        if (
            reference_messages
            and "reference_messages" in signature(self._compaction_service.compact).parameters
        ):
            compact_kwargs["reference_messages"] = list(reference_messages)

        result = await self._compaction_service.compact(list(messages), llm, **compact_kwargs)
        if checkpoint_record is not None:
            result.checkpoint_ref = checkpoint_record.reference
            result.checkpoint_path = str(checkpoint_record.path)
        return result

    def strip_user_image_inputs(
        self,
        placeholder_text: str = "[Image omitted after vision turn]",
    ) -> int:
        """Replace user image parts with a text placeholder.

        This is useful before switching from a vision-capable model back to a
        text-only model in the same session.

        Returns:
            Number of image parts replaced.
        """
        replaced_count = 0
        for msg in self._messages:
            if not isinstance(msg, UserMessage):
                continue
            if not isinstance(msg.content, list):
                continue

            new_parts = []
            inserted_placeholder = False
            changed = False
            for part in msg.content:
                if getattr(part, "type", None) == "image_url":
                    replaced_count += 1
                    changed = True
                    if not inserted_placeholder:
                        new_parts.append(ContentPartTextParam(text=placeholder_text))
                        inserted_placeholder = True
                    continue
                new_parts.append(part)

            if changed:
                msg.content = new_parts

        if replaced_count:
            self.rebuild_role_index()
            self.invalidate_budget_baseline()
        return replaced_count

    def replace_messages(self, messages: Iterable[BaseMessage]) -> None:
        self._messages = list(messages)
        self.rebuild_role_index()
        self._reconstruct_rounds_from_messages()
        self.clear_compaction_state()
        self.invalidate_budget_baseline()

    @staticmethod
    def _message_text(message: BaseMessage) -> str:
        text = getattr(message, "text", None)
        if isinstance(text, str):
            return text
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                str(getattr(part, "text", ""))
                for part in content
                if getattr(part, "type", None) == "text"
            )
        return ""

    @classmethod
    def _is_runtime_injected_user_message(cls, message: BaseMessage) -> bool:
        if not isinstance(message, UserMessage):
            return False
        text = cls._message_text(message).lstrip()
        return text.startswith(
            (
                ACTIVE_TODO_SNAPSHOT_HEADER,
                FINISH_GUARD_PROMPT_PREFIX,
                INVALID_TOOL_CALL_RECOVERY_PREFIX,
                COMPACTED_WORKING_SET_HEADER,
            )
        )

    def _reconstruct_rounds_from_messages(self) -> None:
        self._message_rounds.clear()
        self._message_seen_by_model.clear()
        current_round = 0
        for message in self._messages:
            if isinstance(message, UserMessage) and not self._is_runtime_injected_user_message(
                message
            ):
                current_round += 1
            self._message_rounds[id(message)] = current_round
        self._conversation_round = current_round

    @staticmethod
    def _find_matching_assistant_index(
        messages: Sequence[BaseMessage],
        tool_call_id: str,
        tool_index: int,
    ) -> int | None:
        """Find the assistant message that issued a specific tool call."""
        for i in range(tool_index - 1, -1, -1):
            msg = messages[i]
            if isinstance(msg, ToolMessage):
                continue
            if isinstance(msg, AssistantMessage) and msg.tool_calls:
                if any(tool_call.id == tool_call_id for tool_call in msg.tool_calls):
                    return i
            break
        return None

    @staticmethod
    def _expand_keep_indices_for_tool_pairs(
        messages: Sequence[BaseMessage],
        keep_indices: set[int],
    ) -> set[int]:
        """Expand keep set so tool transactions are preserved atomically."""
        expanded = set(keep_indices)
        changed = True

        while changed:
            changed = False
            for index in list(expanded):
                msg = messages[index]

                if isinstance(msg, ToolMessage):
                    assistant_index = ContextManager._find_matching_assistant_index(
                        messages, msg.tool_call_id, index
                    )
                    if assistant_index is not None and assistant_index not in expanded:
                        expanded.add(assistant_index)
                        changed = True
                    continue

                if not isinstance(msg, AssistantMessage) or not msg.tool_calls:
                    continue

                expected_tool_ids = {tool_call.id for tool_call in msg.tool_calls}
                next_index = index + 1
                while next_index < len(messages) and isinstance(messages[next_index], ToolMessage):
                    tool_msg = messages[next_index]
                    if tool_msg.tool_call_id in expected_tool_ids and next_index not in expanded:
                        expanded.add(next_index)
                        changed = True
                    next_index += 1

        return expanded

    @staticmethod
    def _should_keep_destroyed_tool_message(
        messages: Sequence[BaseMessage],
        keep_indices: set[int],
        index: int,
    ) -> bool:
        """Keep destroyed tool messages only when their assistant tool call is also kept."""
        msg = messages[index]
        if not isinstance(msg, ToolMessage):
            return False

        assistant_index = ContextManager._find_matching_assistant_index(
            messages, msg.tool_call_id, index
        )
        return assistant_index is not None and assistant_index in keep_indices

    def get_messages_by_role(self, role: str) -> list[BaseMessage]:
        """Return a shallow copy of messages for a given role."""
        return list(self._by_role.get(role, []))

    def rebuild_role_index(self) -> None:
        """Rebuild role index from ordered messages (after bulk edits)."""
        self._by_role = defaultdict(list)
        for msg in self._messages:
            self._by_role[msg.role].append(msg)

    def replace_message_range(
        self,
        start: int,
        end: int,
        new_messages: Iterable[BaseMessage],
    ) -> None:
        """Replace a slice [start:end] with new messages, keeping order."""
        self._messages[start:end] = list(new_messages)
        self.rebuild_role_index()
        self._reconstruct_rounds_from_messages()
        self.clear_compaction_state()
        self.invalidate_budget_baseline()

    def inject_message(
        self,
        message: BaseMessage,
        pinned: bool = True,
        after_roles: Iterable[str] = ("system", "developer"),
    ) -> None:
        """Inject a message into context, optionally pinning it."""
        msg = message
        if pinned and message.role not in ("system", "developer"):
            msg = UserMessage(content=getattr(message, "content", ""))

        after_roles_set = set(after_roles)
        insert_at = 0
        for i, m in enumerate(self._messages):
            if m.role in after_roles_set:
                insert_at = i + 1

        self._messages.insert(insert_at, msg)
        self.rebuild_role_index()
        self._message_rounds[id(msg)] = self._conversation_round
        self.clear_compaction_state()
        self.invalidate_budget_baseline()

    @staticmethod
    def _increment_count(counts: dict[str, int], key: str) -> None:
        counts[key] = counts.get(key, 0) + 1

    def _remove_indices(
        self,
        indices: set[int],
        *,
        removed_by_kind: dict[str, int] | None = None,
    ) -> int:
        if not indices:
            return 0
        for index in sorted(indices, reverse=True):
            if removed_by_kind is not None:
                message = self._messages[index]
                text = self._message_text(message).lstrip()
                if text.startswith(ACTIVE_TODO_SNAPSHOT_HEADER):
                    self._increment_count(removed_by_kind, "active_todo_snapshot")
                elif text.startswith(FINISH_GUARD_PROMPT_PREFIX):
                    self._increment_count(removed_by_kind, "finish_guard_prompt")
                elif text.startswith(INVALID_TOOL_CALL_RECOVERY_PREFIX):
                    self._increment_count(removed_by_kind, "tool_call_recovery_prompt")
            msg = self._messages.pop(index)
            self._message_rounds.pop(id(msg), None)
            self._message_seen_by_model.discard(id(msg))
        self.rebuild_role_index()
        self.clear_compaction_state()
        self.invalidate_budget_baseline()
        return len(indices)

    def snip_redundant_runtime_context(self) -> ContextSnipResult:
        """Remove repeated or consumed runtime-injected control messages."""
        todo_indices: list[int] = []
        finish_guard_indices: list[int] = []
        recovery_indices: list[int] = []

        for index, message in enumerate(self._messages):
            if not isinstance(message, UserMessage):
                continue
            text = self._message_text(message).lstrip()
            if text.startswith(ACTIVE_TODO_SNAPSHOT_HEADER):
                todo_indices.append(index)
            elif text.startswith(FINISH_GUARD_PROMPT_PREFIX):
                finish_guard_indices.append(index)
            elif text.startswith(INVALID_TOOL_CALL_RECOVERY_PREFIX):
                recovery_indices.append(index)

        remove_indices: set[int] = set()
        if len(todo_indices) > 1:
            remove_indices.update(todo_indices[:-1])
        if len(finish_guard_indices) > 1:
            remove_indices.update(finish_guard_indices[:-1])

        pending_recovery_indices: list[int] = []
        for index in recovery_indices:
            consumed = any(
                isinstance(message, (AssistantMessage, ToolMessage))
                for message in self._messages[index + 1 :]
            )
            if consumed:
                remove_indices.add(index)
            else:
                pending_recovery_indices.append(index)
        if len(pending_recovery_indices) > 1:
            remove_indices.update(pending_recovery_indices[:-1])

        removed_by_kind: dict[str, int] = {}
        removed_count = self._remove_indices(remove_indices, removed_by_kind=removed_by_kind)
        return ContextSnipResult(
            removed_count=removed_count,
            removed_by_kind=removed_by_kind,
        )

    @staticmethod
    def _tool_message_text_size(message: ToolMessage) -> int:
        content = message.content
        if isinstance(content, str):
            return len(content)
        try:
            return len(json.dumps([part.model_dump(mode="json") for part in content]))
        except Exception:
            return len(message.text)

    @staticmethod
    def _extract_artifact_path_from_text(text: str) -> str | None:
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped.startswith("Artifact file:"):
                continue
            _, _, value = stripped.partition(":")
            cleaned = value.strip()
            return cleaned or None
        return None

    @staticmethod
    def _is_runtime_artifact_read_result(message: ToolMessage) -> bool:
        if message.tool_name != "read":
            return False
        text = message.text
        return "# artifact_meta" in text and "--- artifact_body ---" in text

    def _tool_message_is_microcompact_eligible(
        self,
        message: ToolMessage,
        *,
        min_chars: int,
        preserve_recent_rounds: int,
        excluded_tool_names: frozenset[str],
    ) -> bool:
        if message.destroyed or message.microcompacted:
            return False
        if message.tool_name in excluded_tool_names:
            return False
        if not isinstance(message.content, str):
            return False
        if id(message) not in self._message_seen_by_model:
            return False
        message_round = self._message_rounds.get(id(message), self._conversation_round)
        if message_round > self._conversation_round - preserve_recent_rounds:
            return False
        if self._tool_message_text_size(message) <= min_chars:
            return False
        if self._is_runtime_artifact_read_result(message):
            return False
        return True

    def microcompact_tool_messages(
        self,
        *,
        summarize_tool_message: Callable[[ToolMessage, str], str],
        min_chars: int = 1200,
        preserve_recent_rounds: int = 3,
        excluded_tool_names: frozenset[str] = DEFAULT_MICROCOMPACT_EXCLUDED_TOOL_NAMES,
    ) -> ToolMicrocompactResult:
        """Replace aged tool results with micro summaries plus artifact refs."""
        result = ToolMicrocompactResult()
        changed = False

        for message in self._messages:
            if not isinstance(message, ToolMessage):
                continue
            if not self._tool_message_is_microcompact_eligible(
                message,
                min_chars=min_chars,
                preserve_recent_rounds=preserve_recent_rounds,
                excluded_tool_names=excluded_tool_names,
            ):
                continue

            artifact_path = message.context_artifact_path or self._extract_artifact_path_from_text(
                message.text
            )
            if artifact_path is None:
                policy = (message.context_policy or "raw").lower()
                if policy in {"trim", "summarize", "store_only"}:
                    result.skipped_count += 1
                    continue
                artifact_path = self.persist_tool_artifact(
                    message,
                    force=True,
                    summary_text=f"{message.tool_name} output before microcompaction.",
                )
                if artifact_path is not None:
                    result.artifact_created_count += 1

            if artifact_path is None:
                result.skipped_count += 1
                continue

            micro_summary = summarize_tool_message(message, artifact_path)
            if not micro_summary.strip():
                result.skipped_count += 1
                continue

            message.content = micro_summary
            message.context_artifact_path = artifact_path
            message.context_summary = micro_summary
            message.microcompacted = True
            message.microcompacted_at_turn = self._conversation_round
            result.microcompacted_count += 1
            changed = True

        if changed:
            self.rebuild_role_index()
            self.clear_compaction_state()
            self.invalidate_budget_baseline()

        return result

    def prune_ephemeral(
        self,
        tool_map: dict[str, "Tool"],
        storage_path: "Path | None" = None,
    ) -> None:
        """Destroy old ephemeral tool outputs, keeping last N per tool."""
        # Group ephemeral messages by tool name, preserving order
        ephemeral_by_tool: dict[str, list[ToolMessage]] = {}

        for msg in self._messages:
            # 过滤条件：必须是toolmessage、必须标记为ephemeral、必须还没被销毁
            if not isinstance(msg, ToolMessage):
                continue
            if not msg.ephemeral:
                continue
            if msg.destroyed:
                continue

            # 按tool_name分组，每个tool有自己独立的保留窗口
            if msg.tool_name not in ephemeral_by_tool:
                ephemeral_by_tool[msg.tool_name] = []
            ephemeral_by_tool[msg.tool_name].append(msg)

        for tool_name, messages in ephemeral_by_tool.items():
            # Get the keep limit from the tool's ephemeral attribute
            tool = tool_map.get(tool_name)
            if tool is None:
                keep_count = 1
            else:
                keep_count = tool.ephemeral if isinstance(tool.ephemeral, int) else 1

            # Destroy messages beyond the keep limit (older ones first)
            messages_to_destroy = messages[:-keep_count] if keep_count > 0 else messages

            for msg in messages_to_destroy:
                # Log which message is being destroyed
                logger.debug(
                    f"🗑️  Destroying ephemeral: {msg.tool_name} (keeping last {keep_count})"
                )

                # Save to disk if storage path is configured
                # 存储用于log、故障排查、训练数据收集等
                if storage_path is not None:
                    Path(storage_path).mkdir(parents=True, exist_ok=True)
                    filename = f"{msg.tool_call_id}.json"
                    filepath = Path(storage_path) / filename

                    if isinstance(msg.content, str):
                        content_data = msg.content
                    else:
                        content_data = [part.model_dump() for part in msg.content]

                    saved_data = {
                        "tool_call_id": msg.tool_call_id,
                        "tool_name": msg.tool_name,
                        "content": content_data,
                        "is_error": msg.is_error,
                    }
                    filepath.write_text(json.dumps(saved_data, indent=2))

                # Mark as destroyed - serializers will use placeholder instead of content
                msg.destroyed = True

        if ephemeral_by_tool:
            self.invalidate_budget_baseline()

    def configure_compaction(
        self,
        config: CompactionConfig,
        llm: "BaseChatModel",
        token_cost: "TokenCost | None",
    ) -> None:
        """Attach a compaction service to this context manager."""
        self._budget_engine = ContextBudgetEngine(
            config=config,
            token_cost=token_cost,
        )
        self._compaction_service = CompactionService(
            config=config,
            llm=llm,
            token_cost=token_cost,
            budget_engine=self._budget_engine,
        )

    def record_prompt_usage(
        self,
        *,
        model: str,
        messages: Sequence[BaseMessage],
        usage: "ChatInvokeUsage | None",
    ) -> None:
        """Store the latest real prompt usage for later incremental estimates."""
        if self._compaction_service is not None:
            self._compaction_service.update_usage(usage)
        if self._budget_engine is not None:
            self._budget_engine.record_usage(model=model, messages=messages, usage=usage)
        for message in messages:
            self._message_seen_by_model.add(id(message))

    @property
    def summarized_boundary(self) -> int:
        """Return the prefix boundary that is already compacted."""
        return self._summarized_boundary

    def _instruction_prefix(self) -> list[BaseMessage]:
        prefix: list[BaseMessage] = []
        for message in self._messages:
            if message.role in ("system", "developer"):
                prefix.append(message)
                continue
            break
        return prefix

    def _countable_indices_from(
        self,
        messages: Sequence[BaseMessage],
        *,
        start_index: int,
        pin_roles: Iterable[str] = ("system", "developer"),
    ) -> list[int]:
        pinned_roles = set(pin_roles)
        countable_indices: list[int] = []
        for i, msg in enumerate(messages):
            if i < start_index:
                continue
            if msg.role in pinned_roles:
                continue
            is_destroyed_tool = msg.role == "tool" and bool(getattr(msg, "destroyed", False))
            if is_destroyed_tool:
                continue
            countable_indices.append(i)
        return countable_indices

    def _build_recent_keep_indices(
        self,
        messages: Sequence[BaseMessage],
        countable_indices: Sequence[int],
        keep_count: int,
        *,
        token_budget: int | None = None,
    ) -> set[int]:
        if keep_count <= 0 or not countable_indices:
            return set()
        keep_indices: set[int] = set()
        kept_count = 0
        kept_tokens = 0

        def estimate_subset_tokens(indices: Sequence[int]) -> int:
            subset = [messages[i] for i in indices]
            if self._budget_engine is not None:
                return self._budget_engine.estimate_tokens_for_messages(subset)

            total = 0
            for item in subset:
                try:
                    serialized = item.model_dump_json()
                except Exception:
                    serialized = str(item)
                total += max(1, len(serialized) // 4) + 4
            return total

        for index in reversed(countable_indices):
            candidate_indices = self._expand_keep_indices_for_tool_pairs(
                messages,
                keep_indices | {index},
            )
            new_indices = [
                i for i in candidate_indices if i not in keep_indices and i in countable_indices
            ]
            if not new_indices:
                continue

            candidate_tokens = kept_tokens + estimate_subset_tokens(new_indices)
            if kept_count >= keep_count:
                break
            if token_budget is not None and candidate_tokens > token_budget:
                break

            keep_indices = candidate_indices
            kept_count += len(new_indices)
            kept_tokens = candidate_tokens

        return keep_indices

    def _collect_recent_messages(
        self,
        messages: Sequence[BaseMessage],
        *,
        start_index: int,
        keep_indices: set[int],
    ) -> list[BaseMessage]:
        if not keep_indices:
            return []

        cutoff_index = min(keep_indices)
        recent_messages: list[BaseMessage] = []
        for i, msg in enumerate(messages):
            if i < start_index:
                continue
            if i in keep_indices:
                recent_messages.append(msg)
                continue

            is_destroyed_tool = msg.role == "tool" and bool(getattr(msg, "destroyed", False))
            if (
                is_destroyed_tool
                and i >= cutoff_index
                and self._should_keep_destroyed_tool_message(messages, keep_indices, i)
            ):
                recent_messages.append(msg)
        return recent_messages

    def apply_compaction_result(
        self,
        result: CompactionResult,
        *,
        recent_messages: Sequence[BaseMessage] | None = None,
    ) -> None:
        """Replace older history with a merged compacted working set plus recent tail."""
        if self._compaction_service is None:
            raise RuntimeError("Compaction service unavailable.")

        merged_result = self._compaction_service.merge_results(self._compacted_result, result)
        working_state = getattr(merged_result, "working_state", None) or CompactionWorkingState()
        if self._artifact_refs and hasattr(merged_result, "working_state"):
            working_state.artifact_refs = self._dedupe_preserve_order(
                [*working_state.artifact_refs, *self._artifact_refs]
            )
            merged_result.working_state = working_state
        prefix = self._instruction_prefix()
        compacted_messages = self._compaction_service.create_compacted_messages(merged_result)
        tail = list(recent_messages or [])
        self.replace_messages([*prefix, *compacted_messages, *tail])
        self._compacted_result = merged_result
        self._summarized_boundary = len(prefix) + len(compacted_messages)
        if self._working_state_store is not None and isinstance(merged_result, CompactionResult):
            self._working_state_store.write_result(
                merged_result,
                artifact_refs=self._artifact_refs,
            )

    def _recent_keep_token_budget(self, assessment: BudgetAssessment) -> int:
        """Cap the preserved recent tail so it cannot dominate the compacted context."""
        if self._compaction_service is None:
            return 0

        ratio = max(0.0, float(self._compaction_service.config.preserve_recent_token_ratio))
        compact_budget = int(assessment.compact_threshold * ratio)
        if assessment.warn_threshold > 0:
            compact_budget = min(compact_budget, assessment.warn_threshold)
        return max(0, compact_budget)

    async def assess_budget(
        self,
        *,
        model: str,
        usage: "ChatInvokeUsage | None" = None,
        trigger: str | None = None,
    ) -> BudgetAssessment:
        """Assess the current context budget for one model."""
        if self._budget_engine is None:
            return BudgetAssessment(
                model=model,
                context_limit=0,
                warn_threshold=0,
                compact_threshold=0,
                hard_threshold=0,
                baseline_prompt_tokens=0,
                incremental_tokens=0,
                estimated_tokens=0,
                message_count=len(self._messages),
                warn_threshold_ratio=0.0,
                compact_threshold_ratio=0.0,
                hard_threshold_ratio=0.0,
                threshold_utilization=0.0,
                context_utilization=0.0,
                trigger=trigger,
                token_estimate_source="unknown",
            )
        return await self._budget_engine.assess(
            model=model,
            messages=self._messages,
            usage=usage,
            trigger=trigger,
        )

    async def maintain_budget(
        self,
        llm: "BaseChatModel",
        *,
        trigger: str | None = None,
    ) -> BudgetAssessment:
        """Apply formal compaction from one budget assessment path."""
        assessment = await self.assess_budget(model=llm.model, trigger=trigger)

        if assessment.needs_compaction:
            compacted = await self.check_and_compact(llm, trigger="compact_threshold")
            if compacted:
                assessment = await self.assess_budget(
                    model=llm.model,
                    trigger="post_compaction",
                )

        return assessment

    async def check_and_compact(
        self,
        llm: "BaseChatModel",
        usage: "ChatInvokeUsage | None" = None,
        *,
        trigger: str | None = None,
    ) -> bool:
        """Compact when the unified budget engine says the context is too large."""
        if self._compaction_service is None:
            return False
        if not self._compaction_service.config.enabled:
            return False
        if usage is not None:
            self._compaction_service.update_usage(usage)

        assessment = await self.assess_budget(model=llm.model, usage=usage, trigger=trigger)
        if not assessment.needs_compaction:
            return False

        countable_indices = self._countable_indices_from(
            self._messages,
            start_index=self._summarized_boundary,
        )
        if not countable_indices:
            return False

        keep_indices = self._build_recent_keep_indices(
            self._messages,
            countable_indices,
            self._compaction_service.config.preserve_recent_messages,
            token_budget=self._recent_keep_token_budget(assessment),
        )
        compacted_indices = [i for i in countable_indices if i not in keep_indices]
        if not compacted_indices:
            return False

        recent_messages = self._collect_recent_messages(
            self._messages,
            start_index=self._summarized_boundary,
            keep_indices=keep_indices,
        )
        if self.compaction_start_handler is not None:
            self.compaction_start_handler("Compaction start")
        result = await self.compact_messages(
            [self._messages[i] for i in compacted_indices],
            llm,
            reference_messages=recent_messages,
        )
        if not (result.summary or "").strip():
            return False

        self.apply_compaction_result(result, recent_messages=recent_messages)
        if self._budget_engine is not None:
            self._budget_engine.note_trigger(trigger or "compact_threshold")
        return True

    def estimate_tokens_simple(self) -> int:
        """Very rough token estimate with basic CJK awareness."""
        total_chars = 0
        cjk_chars = 0
        cjk_re = re.compile(r"[\u4E00-\u9FFF\u3400-\u4DBF]")
        for msg in self._messages:
            if msg.role == "tool" and bool(getattr(msg, "destroyed", False)):
                placeholder = "<removed to save context>"
                total_chars += len(placeholder)
                cjk_chars += len(cjk_re.findall(placeholder))
                continue

            content = getattr(msg, "content", "")
            if isinstance(content, str):
                total_chars += len(content)
                cjk_chars += len(cjk_re.findall(content))
            elif isinstance(content, list):
                for part in content:
                    text = getattr(part, "text", "")
                    if text:
                        total_chars += len(text)
                        cjk_chars += len(cjk_re.findall(text))

        non_cjk = max(0, total_chars - cjk_chars)
        estimated = cjk_chars + (non_cjk // 4)
        return max(1, int(estimated))

    async def check_and_compact_estimated(self, llm: "BaseChatModel") -> bool:
        """Estimate token usage and compact if threshold exceeded."""
        return await self.check_and_compact(llm, trigger="estimated_compaction")

    async def apply_sliding_window_by_messages(
        self,
        keep_count: int,
        llm: "BaseChatModel",
        pin_roles: Iterable[str] = ("system", "developer"),
        buffer: int = 10,
    ) -> bool:
        """Summarize older messages and keep only the most recent N messages.

        Returns True if compaction happened, False otherwise.
        """
        if keep_count <= 0:
            return False
        if self._compaction_service is None:
            return False

        messages = self._messages
        if len(messages) <= keep_count:
            return False

        countable_indices = self._countable_indices_from(
            messages,
            start_index=self._summarized_boundary,
            pin_roles=pin_roles,
        )

        if not countable_indices:
            return False

        tail_budget: int | None = None
        if self._budget_engine is not None:
            assessment = await self._budget_engine.assess(model=llm.model, messages=messages)
            tail_budget = assessment.warn_threshold

        if len(countable_indices) <= keep_count + max(0, buffer) and tail_budget is None:
            return False

        keep_indices: set[int] = set()
        kept_count = 0
        kept_tokens = 0
        stopped_for_budget = False

        def estimate_subset_tokens(indices: Sequence[int]) -> int:
            subset = [messages[i] for i in indices]
            if self._budget_engine is not None:
                return self._budget_engine.estimate_tokens_for_messages(subset)
            total = 0
            for item in subset:
                try:
                    serialized = item.model_dump_json()
                except Exception:
                    serialized = str(item)
                total += max(1, len(serialized) // 4) + 4
            return total

        for index in reversed(countable_indices):
            candidate_indices = self._expand_keep_indices_for_tool_pairs(
                messages,
                keep_indices | {index},
            )
            new_indices = [
                i for i in candidate_indices if i not in keep_indices and i in countable_indices
            ]
            if not new_indices:
                continue

            candidate_tokens = kept_tokens + estimate_subset_tokens(new_indices)
            if kept_count >= keep_count:
                break
            if tail_budget is not None and kept_count > 0 and candidate_tokens > tail_budget:
                stopped_for_budget = True
                break

            keep_indices = candidate_indices
            kept_count += len(new_indices)
            kept_tokens = candidate_tokens

        if not keep_indices:
            keep_indices = {countable_indices[-1]}
            keep_indices = self._expand_keep_indices_for_tool_pairs(messages, keep_indices)

        summarized_indices = [i for i in countable_indices if i not in keep_indices]
        if not summarized_indices:
            return False

        if (
            len(countable_indices) <= keep_count + max(0, buffer)
            and tail_budget is not None
            and kept_tokens < tail_budget
            and not stopped_for_budget
        ):
            return False

        head = [messages[i] for i in summarized_indices]

        result = await self.compact_messages(head, llm)
        if not (result.summary or "").strip():
            return False

        recent_messages = self._collect_recent_messages(
            messages,
            start_index=self._summarized_boundary,
            keep_indices=keep_indices,
        )
        self.apply_compaction_result(result, recent_messages=recent_messages)
        if self._budget_engine is not None:
            self._budget_engine.note_trigger("sliding_window")
        return True

    async def apply_sliding_window_by_rounds(
        self,
        keep_rounds: int,
        llm: "BaseChatModel",
        pin_roles: Iterable[str] = ("system", "developer"),
        buffer: int = 10,
    ) -> bool:
        """Summarize older rounds and keep only the most recent N rounds."""
        if keep_rounds <= 0:
            return False
        if self._compaction_service is None:
            return False

        messages = self._messages
        if not messages:
            return False

        pinned_roles = set(pin_roles)
        rest: list[BaseMessage] = [
            m
            for i, m in enumerate(messages)
            if i >= self._summarized_boundary and m.role not in pinned_roles
        ]

        # Build rounds: each round starts with a user message
        rounds: list[list[BaseMessage]] = []
        current: list[BaseMessage] = []
        for msg in rest:
            if msg.role == "user" and current:
                rounds.append(current)
                current = []
            current.append(msg)
        if current:
            rounds.append(current)

        if len(rounds) <= keep_rounds + max(0, buffer):
            return False

        head_rounds = rounds[:-keep_rounds]
        tail_rounds = rounds[-keep_rounds:]
        head = [msg for r in head_rounds for msg in r]

        result = await self.compact_messages(head, llm)
        if not (result.summary or "").strip():
            return False

        self.apply_compaction_result(
            result, recent_messages=[msg for r in tail_rounds for msg in r]
        )
        return True
