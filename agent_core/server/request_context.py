"""Request-scoped context helpers for agent server executions."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class RequestContext:
    session_id: str | None = None
    user_id: str | None = None


_CURRENT_REQUEST_CONTEXT: ContextVar[RequestContext | None] = ContextVar(
    "agent_server_request_context",
    default=None,
)


def get_request_context() -> RequestContext:
    """Return the current request context, or an empty one when unbound."""
    context = _CURRENT_REQUEST_CONTEXT.get()
    if context is None:
        return RequestContext()
    return context


def get_current_user_id() -> str | None:
    """Return the current request-bound user ID, if any."""
    return get_request_context().user_id


@contextmanager
def bind_request_context(*, session_id: str | None, user_id: str | None) -> Iterator[RequestContext]:
    """Bind request metadata to the current async execution context."""
    context = RequestContext(session_id=session_id, user_id=user_id)
    token: Token[RequestContext | None] = _CURRENT_REQUEST_CONTEXT.set(context)
    try:
        yield context
    finally:
        _CURRENT_REQUEST_CONTEXT.reset(token)
