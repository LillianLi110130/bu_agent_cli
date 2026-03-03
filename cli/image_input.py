"""Utilities for parsing quoted @ image commands into multimodal user content."""

from __future__ import annotations

import base64
import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from bu_agent_sdk.llm.messages import (
    ContentPartImageParam,
    ContentPartTextParam,
    ImageURL,
)
from tools import SandboxContext

SUPPORTED_IMAGE_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
}
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10MB
DEFAULT_IMAGE_PROMPT = "请分析这张图片。"
_IMAGE_COMMAND_HEAD_RE = re.compile(r"^@\s*(?:img\b|['\"])", flags=re.IGNORECASE)
_LEGACY_IMG_RE = re.compile(r"^@\s*img\b", flags=re.IGNORECASE)
_QUOTED_IMAGE_RE = re.compile(
    r"^@\s*(?P<quote>['\"])(?P<path>.+?)(?P=quote)(?P<message>.*)$",
    flags=re.DOTALL,
)
IMAGE_USAGE = "Usage: @\"<image_path>\"<message> or @'<image_path>'<message>"


class ImageInputError(ValueError):
    """Raised when quoted @ image input is invalid."""


@dataclass
class ParsedImageInput:
    """Parsed result for quoted @ image command."""

    source_path: Path
    user_text: str
    mime_type: str
    content_parts: list[ContentPartTextParam | ContentPartImageParam]


def is_image_command(text: str) -> bool:
    """Return True when the input uses image @ syntax (new or legacy)."""
    return bool(_IMAGE_COMMAND_HEAD_RE.match(text.strip()))


def parse_image_command(
    text: str,
    ctx: SandboxContext,
    default_prompt: str = DEFAULT_IMAGE_PROMPT,
) -> ParsedImageInput:
    """Parse quoted @ image command text into multimodal message content."""
    stripped = text.strip()
    if _LEGACY_IMG_RE.match(stripped):
        raise ImageInputError(
            "Legacy @img syntax was removed. "
            "Use @\"<image_path>\"<message> or @'<image_path>'<message>."
        )

    match = _QUOTED_IMAGE_RE.match(stripped)
    if not match:
        raise ImageInputError(IMAGE_USAGE)

    path_part = match.group("path").strip()
    user_text = match.group("message")
    if not path_part:
        raise ImageInputError("Image path is empty.")

    try:
        resolved = ctx.resolve_path(path_part)
    except Exception as e:
        raise ImageInputError(f"Invalid path: {e}") from e

    if not resolved.exists():
        raise ImageInputError(f"Image not found: {resolved}")
    if not resolved.is_file():
        raise ImageInputError(f"Path is not a file: {resolved}")

    mime_type = _detect_supported_mime_type(resolved)
    if mime_type is None:
        supported = ", ".join(sorted(SUPPORTED_IMAGE_MIME_TYPES))
        raise ImageInputError(
            f"Unsupported image type for '{resolved.name}'. Supported: {supported}"
        )

    try:
        raw_bytes = resolved.read_bytes()
    except OSError as e:
        raise ImageInputError(f"Failed to read image: {resolved} ({e})") from e
    if len(raw_bytes) > MAX_IMAGE_BYTES:
        raise ImageInputError(
            f"Image is too large: {len(raw_bytes)} bytes (max {MAX_IMAGE_BYTES} bytes)"
        )

    prompt = user_text.strip() or default_prompt
    data_b64 = base64.b64encode(raw_bytes).decode("ascii")
    image_url = f"data:{mime_type};base64,{data_b64}"

    image_part = ContentPartImageParam(
        image_url=ImageURL(
            url=image_url,
            detail="auto",
            media_type=_to_supported_media_type(mime_type),
        )
    )
    text_part = ContentPartTextParam(text=prompt)

    return ParsedImageInput(
        source_path=resolved,
        user_text=prompt,
        mime_type=mime_type,
        content_parts=[text_part, image_part],
    )


def _detect_supported_mime_type(path: Path) -> str | None:
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = (mime_type or "").lower().strip()

    if mime_type == "image/jpg":
        mime_type = "image/jpeg"

    if mime_type in SUPPORTED_IMAGE_MIME_TYPES:
        return mime_type

    suffix_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    return suffix_map.get(path.suffix.lower())


def _to_supported_media_type(
    mime_type: str,
) -> Literal["image/jpeg", "image/png", "image/gif", "image/webp"]:
    if mime_type == "image/jpeg":
        return "image/jpeg"
    if mime_type == "image/png":
        return "image/png"
    if mime_type == "image/gif":
        return "image/gif"
    return "image/webp"