"""
Session management for agent instances.

This module provides a session manager that creates and manages
agent instances with unique session IDs.
"""

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from typing import Callable
from uuid import uuid4

from agent_core.agent import Agent
from agent_core.agent.events import FinalResponseEvent as AgentFinalResponseEvent
from agent_core.llm.messages import BaseMessage, DeveloperMessage, SystemMessage

logger = logging.getLogger("agent_core.server.session")


class AgentSession:
    """A single agent session with its own agent instance."""

    def __init__(
        self,
        session_id: str,
        agent: Agent,
        created_at: datetime | None = None,
        user_id: str | None = None,
        memory_service: object | None = None,
    ):
        self.session_id = session_id
        self.agent = agent
        self.user_id = user_id
        self.memory_service = memory_service
        self.history_loaded = False
        self.memory_context_injected = False
        self.created_at = created_at or datetime.now(UTC)
        self.last_used_at = self.created_at
        self._lock = asyncio.Lock()

    def bind_user(self, user_id: str | None) -> None:
        """Bind a user ID to the session or validate an existing binding."""
        if user_id is None:
            return
        if self.user_id is None:
            self.user_id = user_id
            return
        if self.user_id != user_id:
            raise ValueError("session user_id mismatch")

    def _build_loaded_history(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        loaded_messages: list[BaseMessage] = []
        if getattr(self.agent, "system_prompt", None):
            loaded_messages.append(SystemMessage(content=self.agent.system_prompt, cache=True))
        loaded_messages.extend(messages)
        return loaded_messages

    async def _ensure_history_loaded(self) -> None:
        if self.history_loaded or self.memory_service is None or self.user_id is None:
            return

        loader = getattr(self.memory_service, "load_history", None)
        if loader is None:
            self.history_loaded = True
            return

        messages = await loader(self.session_id, self.user_id)
        self.agent.load_history(self._build_loaded_history(messages))
        self.history_loaded = True

    async def _load_user_memory_context(self) -> DeveloperMessage | None:
        if self.memory_service is None or self.user_id is None:
            return None

        loader = getattr(self.memory_service, "load_user_memory_context", None)
        if loader is None:
            return None

        return await loader(self.user_id)

    def _inject_temporary_message(self, message: BaseMessage | None) -> BaseMessage | None:
        if message is None:
            return None

        context = getattr(self.agent, "_context", None)
        if context is not None:
            context.inject_message(message, pinned=True, after_roles=("system", "developer"))
            return message

        agent_messages = getattr(self.agent, "messages", None)
        if isinstance(agent_messages, list):
            insert_at = 0
            for i, existing in enumerate(agent_messages):
                if existing.role in {"system", "developer"}:
                    insert_at = i + 1
            agent_messages.insert(insert_at, message)
            return message

        return None

    async def _ensure_user_memory_context_injected(self) -> None:
        if self.memory_context_injected:
            return
        if self.memory_service is None or self.user_id is None:
            return

        memory_message = await self._load_user_memory_context()
        if memory_message is not None:
            self._inject_temporary_message(memory_message)
        self.memory_context_injected = True

    async def _append_round(self, user_message: str, assistant_message: str) -> None:
        if self.memory_service is None or self.user_id is None:
            return

        appender = getattr(self.memory_service, "append_round", None)
        if appender is None:
            return

        await appender(
            self.session_id,
            self.user_id,
            user_message,
            assistant_message,
        )

    async def query(self, message: str) -> str:
        """Execute a query on this session's agent."""
        async with self._lock:
            await self._ensure_history_loaded()
            await self._ensure_user_memory_context_injected()
            self.last_used_at = datetime.now(UTC)
            response_text = await self.agent.query(message)
            await self._append_round(message, response_text)
            return response_text

    async def query_stream(self, message: str):
        """Execute a streaming query on this session's agent."""
        async with self._lock:
            await self._ensure_history_loaded()
            await self._ensure_user_memory_context_injected()
            self.last_used_at = datetime.now(UTC)
            async for event in self.agent.query_stream(message):
                if isinstance(event, AgentFinalResponseEvent):
                    await self._append_round(message, event.content)
                yield event

    async def query_stream_delta(self, message: str):
        """Execute a token-level streaming query on this session's agent."""
        async with self._lock:
            await self._ensure_history_loaded()
            await self._ensure_user_memory_context_injected()
            self.last_used_at = datetime.now(UTC)
            async for event in self.agent.query_stream_delta(message):
                if isinstance(event, AgentFinalResponseEvent):
                    await self._append_round(message, event.content)
                yield event

    async def get_usage(self):
        """Get usage statistics for this session."""
        async with self._lock:
            return await self.agent.get_usage()

    async def clear_history(self):
        """Clear the conversation history for this session."""
        async with self._lock:
            self.agent.clear_history()
            self.history_loaded = False
            self.memory_context_injected = False

    @property
    def message_count(self) -> int:
        """Get the number of messages in this session."""
        return len(self.agent.messages)


AgentFactory = Callable[[], Agent]
"""A function that creates a new Agent instance."""


class SessionManager:
    """
    Manages agent sessions with automatic cleanup.

    Sessions are identified by a unique session_id and maintain
    their own conversation state. The manager can optionally
    clean up inactive sessions.
    """

    def __init__(
        self,
        agent_factory: AgentFactory,
        session_timeout_minutes: int = 60,
        max_sessions: int = 1000,
        memory_service: object | None = None,
    ):
        """
        Initialize the session manager.

        Args:
            agent_factory: A callable that creates new Agent instances
            session_timeout_minutes: Minutes of inactivity before a session is eligible for cleanup
            max_sessions: Maximum number of active sessions to maintain
        """
        self._agent_factory = agent_factory
        self._session_timeout = timedelta(minutes=session_timeout_minutes)
        self._max_sessions = max_sessions
        self._memory_service = memory_service
        self._sessions: dict[str, AgentSession] = {}
        self._lock = asyncio.Lock()

    async def get_or_create_session(
        self,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> AgentSession:
        """
        Get an existing session or create a new one.

        Args:
            session_id: Optional session ID. If None, a new session is created.
            user_id: Optional user ID to bind to the session.

        Returns:
            The agent session.
        """
        async with self._lock:
            if session_id and session_id in self._sessions:
                session = self._sessions[session_id]
                session.bind_user(user_id)
                logger.debug(f"Reusing existing session: {session_id}")
                return session

            new_session_id = session_id or str(uuid4())
            agent = self._agent_factory()
            session = AgentSession(
                session_id=new_session_id,
                agent=agent,
                user_id=user_id,
                memory_service=self._memory_service,
            )

            if len(self._sessions) >= self._max_sessions:
                await self._cleanup_sessions()

            self._sessions[new_session_id] = session
            logger.info(f"Created new session: {new_session_id}")
            return session

    async def get_session(self, session_id: str) -> AgentSession | None:
        """
        Get an existing session by ID.

        Args:
            session_id: The session ID to retrieve.

        Returns:
            The agent session, or None if not found.
        """
        async with self._lock:
            return self._sessions.get(session_id)

    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.

        Args:
            session_id: The session ID to delete.

        Returns:
            True if the session was deleted, False if it didn't exist.
        """
        async with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                logger.info(f"Deleted session: {session_id}")
                return True
            return False

    async def _cleanup_sessions(self) -> int:
        """
        Clean up inactive sessions.

        Returns:
            The number of sessions cleaned up.
        """
        now = datetime.now(UTC)
        to_delete = []

        for session_id, session in self._sessions.items():
            # Clean up sessions that haven't been used recently
            if now - session.last_used_at > self._session_timeout:
                to_delete.append(session_id)

        for session_id in to_delete:
            del self._sessions[session_id]

        if to_delete:
            logger.info(f"Cleaned up {len(to_delete)} inactive sessions")

        return len(to_delete)

    async def cleanup_task(self, interval_seconds: int = 300):
        """
        Background task that periodically cleans up inactive sessions.

        Args:
            interval_seconds: How often to run cleanup (default: 5 minutes)
        """
        while True:
            try:
                await asyncio.sleep(interval_seconds)
                await self._cleanup_sessions()
            except asyncio.CancelledError:
                logger.info("Session cleanup task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in session cleanup task: {e}")

    @property
    def session_count(self) -> int:
        """Get the current number of active sessions."""
        return len(self._sessions)

    def list_sessions(self) -> list[dict]:
        """List all active sessions with metadata."""
        return [
            {
                "session_id": s.session_id,
                "user_id": s.user_id,
                "created_at": s.created_at.isoformat(),
                "last_used_at": s.last_used_at.isoformat(),
                "message_count": s.message_count,
            }
            for s in self._sessions.values()
        ]
