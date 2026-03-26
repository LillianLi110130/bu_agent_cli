"""Runtime registry for session-scoped gateway agents."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from agent_core.runtime.session import AgentRuntime

if TYPE_CHECKING:
    from agent_core import Agent
    from tools.sandbox import SandboxContext

RuntimeFactory = Callable[[], tuple["Agent", "SandboxContext"]]


class RuntimeManager:
    """Manage session-scoped runtimes for gateway conversations."""

    def __init__(self, runtime_factory: RuntimeFactory):
        self._runtime_factory = runtime_factory
        self._runtimes: dict[str, AgentRuntime] = {}
        self._lock = asyncio.Lock()

    async def get_or_create_runtime(self, session_key: str) -> AgentRuntime:
        """Return an existing runtime or create a new one for *session_key*."""
        async with self._lock:
            runtime = self._runtimes.get(session_key)
            if runtime is not None:
                runtime.touch()
                return runtime

            agent, context = self._runtime_factory()
            runtime = AgentRuntime(agent=agent, context=context)
            self._runtimes[session_key] = runtime
            return runtime

    async def get_runtime(self, session_key: str) -> AgentRuntime | None:
        """Return the runtime for *session_key* if it already exists."""
        async with self._lock:
            return self._runtimes.get(session_key)

    async def drop_runtime(self, session_key: str) -> bool:
        """Remove a runtime entirely."""
        async with self._lock:
            return self._runtimes.pop(session_key, None) is not None

    async def clear_runtime(self, session_key: str) -> bool:
        """Clear a runtime so the next message gets a fresh agent session."""
        return await self.drop_runtime(session_key)

    async def cleanup_expired_runtimes(self, timeout_minutes: int) -> int:
        """Remove runtimes that have been inactive longer than *timeout_minutes*."""
        cutoff = datetime.now(UTC) - timedelta(minutes=timeout_minutes)
        async with self._lock:
            stale_keys = [
                session_key
                for session_key, runtime in self._runtimes.items()
                if runtime.last_used_at < cutoff
            ]
            for session_key in stale_keys:
                self._runtimes.pop(session_key, None)
            return len(stale_keys)

    @property
    def runtime_count(self) -> int:
        """Return the number of active runtimes."""
        return len(self._runtimes)
