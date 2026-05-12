"""Outbound message tool for proactive text and attachment delivery."""

from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Annotated, Literal

from agent_core.tools import Depends, tool
from cli.im_bridge import get_bridge_store
from cli.im_bridge.store import FileBridgeStore
from tools.sandbox import SandboxContext, get_sandbox_context

_MAX_ATTACHMENT_BYTES = 5 * 1024 * 1024
_ALLOWED_ATTACHMENT_SUFFIXES = {
    ".csv",
    ".doc",
    ".docx",
    ".htm",
    ".html",
    ".jpeg",
    ".jpg",
    ".json",
    ".md",
    ".pdf",
    ".png",
    ".ppt",
    ".pptx",
    ".txt",
    ".xls",
    ".xlsx",
}
_MIME_OVERRIDES = {
    ".csv": "text/csv",
    ".doc": "application/msword",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".htm": "text/html",
    ".html": "text/html",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".json": "application/json",
    ".md": "text/markdown",
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".ppt": "application/vnd.ms-powerpoint",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".txt": "text/plain",
    ".xls": "application/vnd.ms-excel",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}


@tool("Send a proactive text or attachment message to the connected IM channel")
async def message(
    action: Literal["text", "attachment"],
    ctx: Annotated[SandboxContext, Depends(get_sandbox_context)],
    bridge_store: Annotated[FileBridgeStore | None, Depends(get_bridge_store)],
    text: str = "",
    file_path: str = "",
) -> str:
    """Queue one outbound text or attachment event for the worker."""
    if bridge_store is None:
        return "Error: outbound message bridge is not configured."

    if action == "text":
        normalized = text.strip()
        if not normalized:
            return "Error: text is required when action='text'."
        bridge_store.enqueue_outbound_text(normalized)
        return f"Queued outbound text message: {normalized}"

    normalized_path = file_path.strip()
    if not normalized_path:
        return "Error: file_path is required when action='attachment'."

    try:
        resolved = ctx.resolve_path(normalized_path)
    except Exception as exc:
        return f"Security error: {exc}"

    if not resolved.exists() or not resolved.is_file():
        return f"Error: Attachment file not found: {file_path}"

    suffix = resolved.suffix.lower()
    if suffix not in _ALLOWED_ATTACHMENT_SUFFIXES:
        supported = ", ".join(sorted(_ALLOWED_ATTACHMENT_SUFFIXES))
        return (
            f"Error: Unsupported attachment format '{suffix}'. "
            f"Supported formats: {supported}"
        )

    file_size = resolved.stat().st_size
    if file_size > _MAX_ATTACHMENT_BYTES:
        return (
            f"Error: File too large ({file_size} bytes). "
            f"Maximum allowed size is 5 MB."
        )

    mime_type = _detect_mime_type(resolved)
    bridge_store.enqueue_outbound_attachment(
        file_path=str(resolved),
        file_name=resolved.name,
        mime_type=mime_type,
        file_size=file_size,
    )
    return f"Queued outbound attachment: {resolved.name} ({file_size} bytes)"


def _detect_mime_type(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(path.name)
    if guessed:
        return guessed
    return _MIME_OVERRIDES.get(path.suffix.lower(), "application/octet-stream")
