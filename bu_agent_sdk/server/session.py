"""
Session management for agent instances.

This module provides a session manager that creates and manages
agent instances with unique session IDs.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Awaitable, Callable
from uuid import uuid4

from bu_agent_sdk.agent import Agent

logger = logging.getLogger("bu_agent_sdk.server.session")


class AgentSession:
    """A single agent session with its own agent instance."""

    def __init__(self, session_id: str, agent: Agent, created_at: datetime | None = None):
        self.session_id = session_id
        self.agent = agent
        self.created_at = created_at or datetime.utcnow()
        self.last_used_at = self.created_at
        self._lock = asyncio.Lock()

    async def query(self, message: str) -> str:
        """Execute a query on this session's agent."""
        async with self._lock:
            self.last_used_at = datetime.utcnow()
            return await self.agent.query(message)

    async def query_stream(self, message: str):
        """Execute a streaming query on this session's agent."""
        async with self._lock:
            self.last_used_at = datetime.utcnow()
            async for event in self.agent.query_stream(message):
                yield event

    async def query_stream_delta(self, message: str):
        """Execute a token-level streaming query on this session's agent."""
        async with self._lock:
            self.last_used_at = datetime.utcnow()
            async for event in self.agent.query_stream_delta(message):
                yield event

    async def get_usage(self):
        """Get usage statistics for this session."""
        async with self._lock:
            return await self.agent.get_usage()

    async def clear_history(self):
        """Clear the conversation history for this session."""
        async with self._lock:
            self.agent.clear_history()

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
        self._sessions: dict[str, AgentSession] = {}
        self._lock = asyncio.Lock()

    async def get_or_create_session(self, session_id: str | None = None) -> AgentSession:
        """
        Get an existing session or create a new one.

        Args:
            session_id: Optional session ID. If None, a new session is created.

        Returns:
            The agent session.
        """
        async with self._lock:
            # If session_id is provided and exists, return it
            if session_id and session_id in self._sessions:
                session = self._sessions[session_id]
                logger.debug(f"Reusing existing session: {session_id}")
                return session

            # Create new session
            new_session_id = session_id or str(uuid4())
            agent = self._agent_factory()
            session = AgentSession(session_id=new_session_id, agent=agent)

            # Check if we need to clean up old sessions
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
        now = datetime.utcnow()
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
                "created_at": s.created_at.isoformat(),
                "last_used_at": s.last_used_at.isoformat(),
                "message_count": s.message_count,
            }
            for s in self._sessions.values()
        ]
