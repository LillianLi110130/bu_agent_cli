"""
HTTP Client for the agent_core server.

This module provides a Python client for interacting with the HTTP API,
making it easy to integrate with external applications.
"""

import logging
from typing import AsyncIterator, Literal

import httpx

from agent_core.server.models import (
    QueryRequest,
    QueryResponse,
    StreamEvent,
    UsageInfo,
    SessionInfoResponse,
)


logger = logging.getLogger("agent_core.server.client")


class AgentClient:
    """
    Python client for the agent_core HTTP API.

    Example:
        client = AgentClient(base_url="http://localhost:8000")

        # Non-streaming query
        response = await client.query("What is 2+2?")
        print(response.response)

        # Streaming query
        async for event in client.query_stream("Tell me a joke"):
            if event.type == "text":
                print(event.content)
            elif event.type == "final":
                break
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        session_id: str | None = None,
        user_id: str | None = None,
        timeout: float = 300.0,
    ):
        """
        Initialize the client.

        Args:
            base_url: Base URL of the server (e.g., "http://localhost:8000")
            session_id: Optional session ID to reuse. If None, a new one is created.
            timeout: Request timeout in seconds (default: 5 minutes for long-running queries)
        """
        self.base_url = base_url.rstrip("/")
        self._session_id = session_id
        self._user_id = user_id
        self._client = httpx.AsyncClient(timeout=timeout)
        self._initialized = False

    async def __aenter__(self):
        await self._client.__aenter__()
        return self

    async def __aexit__(self, *args):
        await self._client.__aexit__(*args)

    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()

    @property
    def session_id(self) -> str:
        """Get the current session ID, creating one if needed."""
        if self._session_id is None:
            raise RuntimeError("Session not initialized. Call create_session() first.")
        return self._session_id

    async def create_session(self, user_id: str | None = None) -> str:
        """
        Create a new session.

        Returns:
            The new session ID.
        """
        resolved_user_id = user_id or self._user_id
        payload = {}
        if resolved_user_id is not None:
            payload["user_id"] = resolved_user_id

        response = await self._client.post(f"{self.base_url}/sessions", json=payload)
        response.raise_for_status()
        data = response.json()
        self._session_id = data["session_id"]
        self._initialized = True
        logger.info(f"Created session: {self._session_id}")
        return self._session_id

    async def query(
        self,
        message: str,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> QueryResponse:
        """
        Send a non-streaming query to the agent.

        Args:
            message: The user message.
            session_id: Optional session ID. If None, uses the current session.

        Returns:
            QueryResponse with the agent's response and usage info.
        """
        sid = session_id or self._session_id
        resolved_user_id = user_id or self._user_id
        if sid is None and resolved_user_id is None:
            raise RuntimeError("user_id is required before creating a new session")

        response = await self._client.post(
            f"{self.base_url}/agent/query",
            json=QueryRequest(
                message=message,
                session_id=sid,
                user_id=resolved_user_id,
            ).model_dump(),
        )
        response.raise_for_status()
        result = QueryResponse(**response.json())
        self._session_id = result.session_id
        if resolved_user_id is not None:
            self._user_id = resolved_user_id
        return result

    async def query_stream(
        self,
        message: str,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """
        Send a streaming query to the agent.

        Args:
            message: The user message.
            session_id: Optional session ID. If None, uses the current session.

        Yields:
            StreamEvent objects as they arrive from the server.
        """
        sid = session_id or self._session_id
        resolved_user_id = user_id or self._user_id
        if sid is None and resolved_user_id is None:
            raise RuntimeError("user_id is required before creating a new session")

        async with self._client.stream(
            "POST",
            f"{self.base_url}/agent/query-stream",
            json={"message": message, "session_id": sid, "user_id": resolved_user_id},
            timeout=None,
        ) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]  # Remove "data: " prefix
                    if data.strip() == ": done":
                        break

                    import json

                    event_data = json.loads(data)

                    # Parse event based on type
                    event_type = event_data.get("type")
                    if event_type == "text":
                        from agent_core.server.models import TextEvent

                        yield TextEvent(**event_data)
                    elif event_type == "thinking":
                        from agent_core.server.models import ThinkingEvent

                        yield ThinkingEvent(**event_data)
                    elif event_type == "tool_call":
                        from agent_core.server.models import ToolCallEvent

                        yield ToolCallEvent(**event_data)
                    elif event_type == "tool_result":
                        from agent_core.server.models import ToolResultEvent

                        yield ToolResultEvent(**event_data)
                    elif event_type == "step_start":
                        from agent_core.server.models import StepStartEvent

                        yield StepStartEvent(**event_data)
                    elif event_type == "step_complete":
                        from agent_core.server.models import StepCompleteEvent

                        yield StepCompleteEvent(**event_data)
                    elif event_type == "final":
                        from agent_core.server.models import FinalResponseEvent

                        yield FinalResponseEvent(**event_data)
                    elif event_type == "hidden":
                        from agent_core.server.models import HiddenMessageEvent

                        yield HiddenMessageEvent(**event_data)
                    elif event_type == "usage":
                        # Usage info, update session_id if present
                        if "session_id" in event_data:
                            self._session_id = event_data["session_id"]
                    elif event_type == "error":
                        logger.error(f"Server error: {event_data.get('error')}")
                    else:
                        logger.warning(f"Unknown event type: {event_type}")

    async def get_usage(self, session_id: str | None = None) -> dict:
        """
        Get usage statistics for a session.

        Args:
            session_id: Optional session ID. If None, uses the current session.

        Returns:
            Dictionary with usage information.
        """
        sid = session_id or self._ensure_session()

        response = await self._client.get(f"{self.base_url}/agent/usage/{sid}")
        response.raise_for_status()
        return response.json()

    async def get_session_info(self, session_id: str | None = None) -> SessionInfoResponse:
        """
        Get information about a session.

        Args:
            session_id: Optional session ID. If None, uses the current session.

        Returns:
            SessionInfoResponse with session details.
        """
        sid = session_id or self._ensure_session()

        response = await self._client.get(f"{self.base_url}/sessions/{sid}")
        response.raise_for_status()
        return SessionInfoResponse(**response.json())

    async def clear_history(self, session_id: str | None = None) -> bool:
        """
        Clear the conversation history for a session.

        Args:
            session_id: Optional session ID. If None, uses the current session.

        Returns:
            True if successful.
        """
        sid = session_id or self._ensure_session()

        response = await self._client.post(f"{self.base_url}/sessions/{sid}/clear", json={})
        response.raise_for_status()
        return True

    async def delete_session(self, session_id: str | None = None) -> bool:
        """
        Delete a session.

        Args:
            session_id: Optional session ID. If None, uses the current session.

        Returns:
            True if successful.
        """
        sid = session_id or self._ensure_session()

        response = await self._client.delete(f"{self.base_url}/sessions/{sid}")
        response.raise_for_status()

        if sid == self._session_id:
            self._session_id = None

        return True

    async def health_check(self) -> dict:
        """
        Check the server health.

        Returns:
            Health status information.
        """
        response = await self._client.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def _ensure_session(self) -> str:
        """Ensure a session exists before calling session-scoped endpoints."""
        if self._session_id is None:
            raise RuntimeError(
                "Session not initialized. Call query() with user_id or create_session() first."
            )
        return self._session_id


class SimpleAgentClient:
    """
    Simplified synchronous-style client that auto-manages sessions.

    This is a convenience wrapper that handles session creation automatically
    and provides a simpler interface for basic use cases.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        auto_create_session: bool = True,
        user_id: str | None = None,
    ):
        """
        Initialize the simple client.

        Args:
            base_url: Base URL of the server.
            auto_create_session: Whether to auto-create a session on first query.
        """
        self._client = AgentClient(base_url=base_url, user_id=user_id)
        self._auto_create = auto_create_session

    async def __aenter__(self):
        await self._client.__aenter__()
        if self._auto_create:
            await self._client.create_session()
        return self

    async def __aexit__(self, *args):
        await self._client.__aexit__(*args)

    async def query(self, message: str) -> str:
        """Send a query and return just the response text."""
        response = await self._client.query(message)
        return response.response

    async def query_stream(self, message: str):
        """Send a streaming query."""
        return self._client.query_stream(message)

    async def close(self):
        """Close the client."""
        await self._client.close()

    @property
    def session_id(self) -> str:
        """Get the current session ID."""
        return self._client.session_id
