"""SQLite-backed CLI conversation session storage."""

from __future__ import annotations

import json
import re
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from agent_core.llm.messages import (
    AssistantMessage,
    BaseMessage,
    DeveloperMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)


class SessionStoreError(RuntimeError):
    """Raised when a persisted session cannot be read safely."""


@dataclass(frozen=True, slots=True)
class SessionMeta:
    id: str
    workspace_root: str
    workspace_key: str
    title: str | None
    model: str | None
    system_prompt: str | None
    started_at: float
    updated_at: float
    ended_at: float | None
    end_reason: str | None
    message_count: int


@dataclass(frozen=True, slots=True)
class SessionContextSnapshot:
    session_id: str
    messages: list[BaseMessage]
    message_count: int
    compacted: bool
    updated_at: float


@dataclass(frozen=True, slots=True)
class ConversationRound:
    user: str
    assistant: str


_ROLE_MODELS = {
    "user": UserMessage,
    "system": SystemMessage,
    "developer": DeveloperMessage,
    "assistant": AssistantMessage,
    "tool": ToolMessage,
}


def workspace_identity(working_dir: Path) -> tuple[str, str]:
    """Return display root and comparison key for a workspace path."""
    workspace_root = str(working_dir.resolve())
    return workspace_root, workspace_root.casefold()


def _message_to_json(message: BaseMessage) -> str:
    return json.dumps(message.model_dump(mode="json"), ensure_ascii=False)


def _message_from_json(raw: str) -> BaseMessage:
    try:
        payload = json.loads(raw)
        role = payload["role"]
        model = _ROLE_MODELS[role]
        return model.model_validate(payload)
    except Exception as exc:  # noqa: BLE001 - converted to domain error for callers.
        raise SessionStoreError("invalid persisted message") from exc


def _messages_to_json(messages: Iterable[BaseMessage]) -> str:
    payload = [message.model_dump(mode="json") for message in messages]
    return json.dumps(payload, ensure_ascii=False)


def _messages_from_json(raw: str) -> list[BaseMessage]:
    try:
        payload = json.loads(raw)
        if not isinstance(payload, list):
            raise TypeError("context payload is not a list")
        return [_message_from_json(json.dumps(item, ensure_ascii=False)) for item in payload]
    except SessionStoreError:
        raise
    except Exception as exc:  # noqa: BLE001 - converted to domain error for callers.
        raise SessionStoreError("invalid persisted context") from exc


def readable_message_text(message: BaseMessage) -> str:
    """Render the user/assistant visible text used by titles and previews."""
    if isinstance(message, UserMessage):
        content = message.content
        if isinstance(content, str):
            return content
        parts: list[str] = []
        for part in content:
            if part.type == "text":
                parts.append(part.text)
            elif part.type == "image_url":
                parts.append("[image]")
            else:
                parts.append("[attachment]")
        return "\n".join(parts)

    if isinstance(message, AssistantMessage):
        text = message.text
        if text:
            return text
        if message.tool_calls:
            names = [tool_call.function.name for tool_call in message.tool_calls]
            return f"[tool calls: {', '.join(names)}]"
        return ""

    text = getattr(message, "text", None)
    if isinstance(text, str):
        return text
    return ""


def make_session_title(message: BaseMessage, *, max_chars: int = 40) -> str:
    text = readable_message_text(message)
    text = "".join(ch for ch in text if ch.isprintable())
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return "Untitled session"
    return text[:max_chars]


class CLISessionStore:
    """Small synchronous SQLite store for CLI resume metadata and context."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def _connection(self):
        conn = self._connect()
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _initialize_schema(self) -> None:
        with self._connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                  id TEXT PRIMARY KEY,
                  workspace_root TEXT NOT NULL,
                  workspace_key TEXT NOT NULL,
                  title TEXT,
                  model TEXT,
                  system_prompt TEXT,
                  started_at REAL NOT NULL,
                  updated_at REAL NOT NULL,
                  ended_at REAL,
                  end_reason TEXT,
                  message_count INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS messages (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  session_id TEXT NOT NULL,
                  role TEXT NOT NULL,
                  message_json TEXT NOT NULL,
                  created_at REAL NOT NULL,
                  FOREIGN KEY(session_id) REFERENCES sessions(id)
                );

                CREATE TABLE IF NOT EXISTS session_context (
                  session_id TEXT PRIMARY KEY,
                  context_json TEXT NOT NULL,
                  message_count INTEGER NOT NULL,
                  compacted INTEGER NOT NULL DEFAULT 0,
                  updated_at REAL NOT NULL,
                  FOREIGN KEY(session_id) REFERENCES sessions(id)
                );

                CREATE INDEX IF NOT EXISTS idx_sessions_workspace_updated
                  ON sessions(workspace_key, updated_at DESC);

                CREATE INDEX IF NOT EXISTS idx_messages_session_order
                  ON messages(session_id, id);
                """
            )

    def create_session(
        self,
        *,
        session_id: str,
        workspace_root: str,
        workspace_key: str,
        model: str | None,
        system_prompt: str | None,
        now: float | None = None,
    ) -> None:
        ts = time.time() if now is None else now
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO sessions (
                  id, workspace_root, workspace_key, title, model, system_prompt,
                  started_at, updated_at, ended_at, end_reason, message_count
                )
                VALUES (?, ?, ?, NULL, ?, ?, ?, ?, NULL, NULL, 0)
                ON CONFLICT(id) DO UPDATE SET
                  workspace_root = excluded.workspace_root,
                  workspace_key = excluded.workspace_key,
                  model = excluded.model,
                  system_prompt = COALESCE(sessions.system_prompt, excluded.system_prompt),
                  updated_at = excluded.updated_at,
                  ended_at = NULL,
                  end_reason = NULL
                """,
                (
                    session_id,
                    workspace_root,
                    workspace_key,
                    model,
                    system_prompt,
                    ts,
                    ts,
                ),
            )

    def list_sessions(
        self,
        *,
        workspace_key: str,
        exclude_session_id: str | None = None,
        limit: int = 20,
    ) -> list[SessionMeta]:
        query = "SELECT * FROM sessions WHERE workspace_key = ?"
        params: list[object] = [workspace_key]
        if exclude_session_id is not None:
            query += " AND id <> ?"
            params.append(exclude_session_id)
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)
        with self._connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_session(row) for row in rows]

    def get_session(self, session_id: str) -> SessionMeta | None:
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return self._row_to_session(row) if row is not None else None

    def append_messages(self, session_id: str, messages: Iterable[BaseMessage]) -> int:
        materialized = list(messages)
        if not materialized:
            return self.count_messages(session_id)
        created_at = time.time()
        rows = [
            (session_id, message.role, _message_to_json(message), created_at)
            for message in materialized
        ]
        with self._connection() as conn:
            conn.executemany(
                """
                INSERT INTO messages (session_id, role, message_json, created_at)
                VALUES (?, ?, ?, ?)
                """,
                rows,
            )
            count = self._count_messages(conn, session_id)
            title = self._first_user_title(conn, session_id)
            conn.execute(
                """
                UPDATE sessions
                SET message_count = ?, updated_at = ?, title = COALESCE(title, ?)
                WHERE id = ?
                """,
                (count, created_at, title, session_id),
            )
        return count

    def count_messages(self, session_id: str) -> int:
        with self._connection() as conn:
            return self._count_messages(conn, session_id)

    def upsert_context_snapshot(
        self,
        *,
        session_id: str,
        messages: Iterable[BaseMessage],
        compacted: bool,
        now: float | None = None,
    ) -> None:
        materialized = list(messages)
        ts = time.time() if now is None else now
        compacted_value = 1 if compacted else 0
        with self._connection() as conn:
            existing = conn.execute(
                "SELECT compacted FROM session_context WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if existing is not None and int(existing["compacted"] or 0):
                compacted_value = 1
            conn.execute(
                """
                INSERT INTO session_context (
                  session_id, context_json, message_count, compacted, updated_at
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                  context_json = excluded.context_json,
                  message_count = excluded.message_count,
                  compacted = MAX(session_context.compacted, excluded.compacted),
                  updated_at = excluded.updated_at
                """,
                (
                    session_id,
                    _messages_to_json(materialized),
                    len(materialized),
                    compacted_value,
                    ts,
                ),
            )
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (ts, session_id),
            )

    def load_context_snapshot(self, session_id: str) -> SessionContextSnapshot | None:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM session_context WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return SessionContextSnapshot(
            session_id=str(row["session_id"]),
            messages=_messages_from_json(str(row["context_json"])),
            message_count=int(row["message_count"]),
            compacted=bool(row["compacted"]),
            updated_at=float(row["updated_at"]),
        )

    def recent_user_assistant_rounds(
        self,
        session_id: str,
        *,
        limit: int = 10,
    ) -> list[ConversationRound]:
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT message_json FROM messages
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (session_id,),
            ).fetchall()
        messages = [_message_from_json(str(row["message_json"])) for row in rows]
        rounds: list[ConversationRound] = []
        current_user: str | None = None
        assistant_parts: list[str] = []
        for message in messages:
            if isinstance(message, UserMessage):
                if current_user is not None:
                    rounds.append(
                        ConversationRound(
                            user=current_user,
                            assistant="\n".join(part for part in assistant_parts if part),
                        )
                    )
                current_user = readable_message_text(message)
                assistant_parts = []
                continue
            if current_user is not None and isinstance(message, AssistantMessage):
                assistant_parts.append(readable_message_text(message))
        if current_user is not None:
            rounds.append(
                ConversationRound(
                    user=current_user,
                    assistant="\n".join(part for part in assistant_parts if part),
                )
            )
        return rounds[-limit:]

    def end_session(
        self,
        session_id: str,
        *,
        reason: str,
        now: float | None = None,
    ) -> None:
        ts = time.time() if now is None else now
        with self._connection() as conn:
            conn.execute(
                "UPDATE sessions SET ended_at = ?, end_reason = ?, updated_at = ? WHERE id = ?",
                (ts, reason, ts, session_id),
            )

    def reopen_session(self, session_id: str, *, now: float | None = None) -> None:
        ts = time.time() if now is None else now
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE sessions
                SET ended_at = NULL, end_reason = NULL, updated_at = ?
                WHERE id = ?
                """,
                (ts, session_id),
            )

    def touch_session(
        self,
        session_id: str,
        *,
        model: str | None,
        message_count: int | None = None,
        now: float | None = None,
    ) -> None:
        ts = time.time() if now is None else now
        with self._connection() as conn:
            count = (
                self._count_messages(conn, session_id)
                if message_count is None
                else message_count
            )
            conn.execute(
                "UPDATE sessions SET model = ?, message_count = ?, updated_at = ? WHERE id = ?",
                (model, count, ts, session_id),
            )

    @staticmethod
    def _row_to_session(row: sqlite3.Row) -> SessionMeta:
        return SessionMeta(
            id=str(row["id"]),
            workspace_root=str(row["workspace_root"]),
            workspace_key=str(row["workspace_key"]),
            title=row["title"],
            model=row["model"],
            system_prompt=row["system_prompt"],
            started_at=float(row["started_at"]),
            updated_at=float(row["updated_at"]),
            ended_at=float(row["ended_at"]) if row["ended_at"] is not None else None,
            end_reason=row["end_reason"],
            message_count=int(row["message_count"]),
        )

    @staticmethod
    def _count_messages(conn: sqlite3.Connection, session_id: str) -> int:
        row = conn.execute(
            "SELECT COUNT(*) AS count FROM messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return int(row["count"])

    @staticmethod
    def _first_user_title(conn: sqlite3.Connection, session_id: str) -> str | None:
        row = conn.execute(
            """
            SELECT message_json FROM messages
            WHERE session_id = ? AND role = 'user'
            ORDER BY id ASC
            LIMIT 1
            """,
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        return make_session_title(_message_from_json(str(row["message_json"])))
