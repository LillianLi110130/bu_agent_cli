"""
Context manager for conversation messages.

Keeps an ordered list of messages (the true source of context),
and a role-based index for fast grouping.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Iterator, Sequence, TYPE_CHECKING

import logging

from bu_agent_sdk.agent.compaction import CompactionConfig, CompactionService
import re

logger = logging.getLogger("bu_agent_sdk.agent")

from bu_agent_sdk.llm.messages import (
    BaseMessage,
    DeveloperMessage,
    ToolMessage,
    UserMessage,
)
from bu_agent_sdk.llm.views import ChatInvokeUsage

if TYPE_CHECKING:
    from bu_agent_sdk.llm.base import BaseChatModel
    from bu_agent_sdk.llm.views import ChatInvokeUsage
    from bu_agent_sdk.tokens import TokenCost
    from bu_agent_sdk.tools.decorator import Tool
    from pathlib import Path


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

    def add_message(self, message: BaseMessage) -> None:
        self._messages.append(message)
        self._by_role[message.role].append(message)

    def add_messages(self, messages: Iterable[BaseMessage]) -> None:
        for msg in messages:
            self.add_message(msg)

    def remove_message_at(self, index: int = -1) -> BaseMessage:
        msg = self._messages.pop(index)
        role_list = self._by_role.get(msg.role)
        if role_list is not None:
            try:
                role_list.remove(msg)
            except ValueError:
                pass
        return msg

    def clear_messages(self) -> None:
        self._messages.clear()
        self._by_role.clear()

    def replace_messages(self, messages: Iterable[BaseMessage]) -> None:
        self._messages = list(messages)
        self._by_role = defaultdict(list)
        for msg in self._messages:
            self._by_role[msg.role].append(msg)

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

    def inject_message(
        self,
        message: BaseMessage,
        pinned: bool = True,
        after_roles: Iterable[str] = ("system", "developer"),
    ) -> None:
        """Inject a message into context, optionally pinning it."""
        msg = message
        if pinned and message.role not in ("system", "developer"):
            msg = DeveloperMessage(content=getattr(message, "content", ""))

        after_roles_set = set(after_roles)
        insert_at = 0
        for i, m in enumerate(self._messages):
            if m.role in after_roles_set:
                insert_at = i + 1

        self._messages.insert(insert_at, msg)
        self.rebuild_role_index()

    def prune_ephemeral(
        self,
        tool_map: dict[str, "Tool"],
        storage_path: "Path | None" = None,
    ) -> None:
        """Destroy old ephemeral tool outputs, keeping last N per tool."""
        from pathlib import Path
        import json

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

    def configure_compaction(
        self,
        config: CompactionConfig,
        llm: "BaseChatModel",
        token_cost: "TokenCost | None",
    ) -> None:
        """Attach a compaction service to this context manager."""
        self._compaction_service = CompactionService(
            config=config,
            llm=llm,
            token_cost=token_cost,
        )

    async def check_and_compact(
        self,
        llm: "BaseChatModel",
        usage: "ChatInvokeUsage | None",
    ) -> bool:
        """Check token usage and compact if threshold exceeded."""
        if self._compaction_service is None:
            return False

        self._compaction_service.update_usage(usage)

        new_messages, result = await self._compaction_service.check_and_compact(
            self._messages,
            llm,
        )

        if result.compacted:
            summary_text = (result.summary or "").strip()
            if not summary_text:
                return False
            pinned = [
                m for m in self._messages if m.role in ("system", "developer")
            ]
            summary_message = UserMessage(content=summary_text)
            self.replace_messages([*pinned, summary_message])
            return True

        return False

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
        if self._compaction_service is None:
            return False

        estimated = self.estimate_tokens_simple()
        threshold = await self._compaction_service.get_threshold_for_model(llm.model)
        if estimated < threshold:
            return False

        usage = ChatInvokeUsage(
            prompt_tokens=estimated,
            prompt_cached_tokens=0,
            prompt_cache_creation_tokens=0,
            prompt_image_tokens=0,
            completion_tokens=0,
            total_tokens=estimated,
        )
        return await self.check_and_compact(llm, usage)

    async def apply_sliding_window_by_messages(
        self,
        keep_count: int,
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

        pinned_roles = set(pin_roles)

        # Countable messages exclude pinned roles and destroyed tool messages
        countable_indices: list[int] = []
        for i, msg in enumerate(messages):
            if msg.role in pinned_roles:
                continue
            is_destroyed_tool = (
                msg.role == "tool" and bool(getattr(msg, "destroyed", False))
            )
            if is_destroyed_tool:
                continue
            countable_indices.append(i)

        if len(countable_indices) <= keep_count + max(0, buffer):
            return False

        keep_indices = set(countable_indices[-keep_count:])
        cutoff_index = min(keep_indices)
        head = [messages[i] for i in countable_indices if i not in keep_indices]

        result = await self._compaction_service.compact(head)
        summary_text = (result.summary or "").strip()
        if not summary_text:
            return False

        new_messages: list[BaseMessage] = []
        summary_inserted = False
        for i, msg in enumerate(messages):
            if msg.role in pinned_roles:
                new_messages.append(msg)
                continue

            if i in keep_indices:
                new_messages.append(msg)
                continue

            is_destroyed_tool = (
                msg.role == "tool" and bool(getattr(msg, "destroyed", False))
            )
            if is_destroyed_tool and i >= cutoff_index:
                new_messages.append(msg)
                continue

            # This message is summarized into the rolling summary
            if not summary_inserted:
                new_messages.append(UserMessage(content=summary_text))
                summary_inserted = True
            # Skip the original message

        if not summary_inserted:
            return False

        self.replace_messages(new_messages)
        return True

    async def apply_sliding_window_by_rounds(
        self,
        keep_rounds: int,
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
        pinned: list[BaseMessage] = [m for m in messages if m.role in pinned_roles]
        rest: list[BaseMessage] = [m for m in messages if m.role not in pinned_roles]

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

        result = await self._compaction_service.compact(head)
        summary_text = (result.summary or "").strip()
        if not summary_text:
            return False

        new_messages: list[BaseMessage] = []
        new_messages.extend(pinned)
        new_messages.append(UserMessage(content=summary_text))
        new_messages.extend([msg for r in tail_rounds for msg in r])

        self.replace_messages(new_messages)
        return True
