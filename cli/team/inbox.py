"""In-process inbox buffer and attachment hook for team members."""

from __future__ import annotations

import asyncio
from dataclasses import replace

from agent_core.agent.hooks import BaseAgentHook, HookAction, HookDecision
from agent_core.agent.runtime_events import LLMCallRequested
from agent_core.llm.messages import UserMessage
from agent_core.team.models import TeamMessage


class TeamInboxBuffer:
    """Async-safe buffer for team messages awaiting agent context injection."""

    def __init__(self) -> None:
        self._messages: list[TeamMessage] = []
        self._lock = asyncio.Lock()

    async def push(self, message: TeamMessage) -> None:
        async with self._lock:
            self._messages.append(message)

    async def drain(self, limit: int | None = None) -> list[TeamMessage]:
        async with self._lock:
            if limit is None or limit >= len(self._messages):
                messages = list(self._messages)
                self._messages.clear()
                return messages
            messages = self._messages[:limit]
            del self._messages[:limit]
            return messages

    async def has_messages(self) -> bool:
        async with self._lock:
            return bool(self._messages)

    async def count(self) -> int:
        async with self._lock:
            return len(self._messages)


def format_team_messages_for_context(
    messages: list[TeamMessage],
    *,
    member_id: str,
    idle: bool = False,
) -> str:
    mode = "as a new turn" if idle else "before the next model call"
    lines = [
        "## Team Messages",
        "",
        f"You received team messages while running as teammate `{member_id}`.",
        f"These messages are being delivered {mode}.",
        "",
        "Messages:",
    ]
    for message in messages:
        lines.extend(
            [
                f"- From {message.sender} [{message.type}] {message.message_id}:",
                f"  {message.body}",
            ]
        )
        if message.metadata:
            lines.append(f"  metadata: {message.metadata}")
    lines.extend(
        [
            "",
            "Treat these messages as team coordination context. Continue your current work if applicable.",
        ]
    )
    return "\n".join(lines)


class TeamInboxAttachmentHook(BaseAgentHook):
    """Attach pending team messages immediately before a team member LLM call."""

    priority: int = 15

    def __init__(self, *, buffer: TeamInboxBuffer, member_id: str) -> None:
        self.buffer = buffer
        self.member_id = member_id

    async def before_event(self, event, ctx) -> HookDecision | None:
        if not isinstance(event, LLMCallRequested):
            return None

        messages = await self.buffer.drain()
        if not messages:
            return None

        user_message = UserMessage(
            content=format_team_messages_for_context(
                messages,
                member_id=self.member_id,
                idle=False,
            )
        )
        return HookDecision(
            action=HookAction.REPLACE_EVENT,
            replacement_event=replace(event, messages=[*event.messages, user_message]),
        )
