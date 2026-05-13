"""File-per-message mailbox implementation."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from agent_core.team.atomic_io import atomic_write_json, read_json
from agent_core.team.models import TeamMessage, utc_now_iso
from agent_core.team.protocol import normalize_message_type


class Mailbox:
    """Mailbox backed by one JSON file per message."""

    def __init__(self, team_dir: Path):
        self.team_dir = team_dir

    def ensure(self, member_id: str) -> None:
        (self._box(member_id) / "inbox").mkdir(parents=True, exist_ok=True)
        (self._box(member_id) / "read").mkdir(parents=True, exist_ok=True)

    def write_message(
        self,
        *,
        team_id: str,
        sender: str,
        recipient: str,
        type: str,
        body: str,
        metadata: dict | None = None,
    ) -> TeamMessage:
        self.ensure(recipient)
        message = TeamMessage(
            message_id=f"msg_{uuid.uuid4().hex[:12]}",
            team_id=team_id,
            sender=sender,
            recipient=recipient,
            type=normalize_message_type(type),
            body=body,
            metadata=dict(metadata or {}),
        )
        path = self._box(recipient) / "inbox" / f"{message.message_id}.json"
        atomic_write_json(path, message.to_dict())
        return message

    def receive(
        self,
        member_id: str,
        *,
        ack: bool = True,
        limit: int | None = None,
    ) -> list[TeamMessage]:
        self.ensure(member_id)
        inbox = self._box(member_id) / "inbox"
        read_dir = self._box(member_id) / "read"
        messages: list[TeamMessage] = []
        for path in sorted(inbox.glob("*.json")):
            if limit is not None and len(messages) >= limit:
                break
            lock_dir = path.with_suffix(path.suffix + ".lock")
            try:
                lock_dir.mkdir()
            except FileExistsError:
                continue
            try:
                payload = read_json(path, None)
                if payload is None:
                    continue
                message = TeamMessage.from_dict(payload)
                messages.append(message)
                if ack:
                    message.read_at = utc_now_iso()
                    target = read_dir / path.name
                    atomic_write_json(target, message.to_dict())
                    path.unlink(missing_ok=True)
            finally:
                try:
                    lock_dir.rmdir()
                except OSError:
                    pass
        return messages

    def ack(self, member_id: str, message_ids: set[str]) -> list[TeamMessage]:
        """Move specific unread messages to the read mailbox."""
        if not message_ids:
            return []
        self.ensure(member_id)
        inbox = self._box(member_id) / "inbox"
        read_dir = self._box(member_id) / "read"
        messages: list[TeamMessage] = []
        for message_id in sorted(message_ids):
            path = inbox / f"{message_id}.json"
            lock_dir = path.with_suffix(path.suffix + ".lock")
            try:
                lock_dir.mkdir()
            except FileExistsError:
                continue
            try:
                payload = read_json(path, None)
                if payload is None:
                    continue
                message = TeamMessage.from_dict(payload)
                message.read_at = utc_now_iso()
                target = read_dir / path.name
                atomic_write_json(target, message.to_dict())
                path.unlink(missing_ok=True)
                messages.append(message)
            finally:
                try:
                    lock_dir.rmdir()
                except OSError:
                    pass
        return messages

    def _box(self, member_id: str) -> Path:
        return self.team_dir / "mailboxes" / member_id
