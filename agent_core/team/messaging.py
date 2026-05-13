"""Shared team message bus backed by filesystem mailboxes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from agent_core.team.mailbox import Mailbox
from agent_core.team.models import TeamMessage

AppendEvent = Callable[[str, str, dict], None]


class TeamMessenger:
    """Symmetric message sender for lead and teammate processes."""

    def __init__(
        self,
        *,
        team_id: str,
        team_dir: Path,
        append_event: AppendEvent | None = None,
    ) -> None:
        self.team_id = team_id
        self.team_dir = team_dir
        self.mailbox = Mailbox(team_dir)
        self._append_event = append_event

    def deliver(
        self,
        *,
        sender: str,
        recipient: str,
        body: str,
        type: str = "message",
        metadata: dict[str, Any] | None = None,
    ) -> TeamMessage:
        message = self.mailbox.write_message(
            team_id=self.team_id,
            sender=sender,
            recipient=recipient,
            type=type,
            body=body,
            metadata=metadata,
        )
        if self._append_event is not None:
            self._append_event("message_sent", sender, message.to_dict())
        return message

    def receive(
        self,
        member_id: str,
        *,
        ack: bool = True,
        limit: int | None = None,
    ) -> list[TeamMessage]:
        return self.mailbox.receive(member_id, ack=ack, limit=limit)

    def ack(self, member_id: str, message_ids: set[str]) -> list[TeamMessage]:
        return self.mailbox.ack(member_id, message_ids)
