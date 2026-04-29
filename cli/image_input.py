"""Utilities for parsing quoted @ image commands into multimodal user content."""

from __future__ import annotations

import base64
import binascii
import json
import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from agent_core.llm.messages import (
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
IMAGE_USAGE = "用法：@\"<图片路径>\"<消息> 或 @'<图片路径>'<消息>"
REMOTE_IMAGE_MSG_TYPE = "image"


class ImageInputError(ValueError):
    """Raised when quoted @ image input is invalid."""


@dataclass
class ParsedImageInput:
    """Parsed result for quoted @ image command."""

    source_path: Path
    user_text: str
    mime_type: str
    content_parts: list[ContentPartTextParam | ContentPartImageParam]
    invalid_images: list[str] | None = None


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
            "旧版 @img 语法已移除。" "请使用 @\"<图片路径>\"<消息> 或 @'<图片路径>'<消息>。"
        )

    match = _QUOTED_IMAGE_RE.match(stripped)
    if not match:
        raise ImageInputError(IMAGE_USAGE)

    path_part = match.group("path").strip()
    user_text = match.group("message")
    if not path_part:
        raise ImageInputError("图片路径不能为空。")

    try:
        resolved = ctx.resolve_path(path_part)
    except Exception as e:
        raise ImageInputError(f"路径无效：{e}") from e

    if not resolved.exists():
        raise ImageInputError(f"未找到图片：{resolved}")
    if not resolved.is_file():
        raise ImageInputError(f"路径不是文件：{resolved}")

    mime_type = _detect_supported_mime_type(resolved)
    if mime_type is None:
        supported = ", ".join(sorted(SUPPORTED_IMAGE_MIME_TYPES))
        raise ImageInputError(f"不支持图片类型“{resolved.name}”。支持的类型：{supported}")

    try:
        raw_bytes = resolved.read_bytes()
    except OSError as e:
        raise ImageInputError(f"读取图片失败：{resolved}（{e}）") from e
    if len(raw_bytes) > MAX_IMAGE_BYTES:
        raise ImageInputError(f"图片过大：{len(raw_bytes)} 字节（最大 {MAX_IMAGE_BYTES} 字节）")

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
        invalid_images=[],
    )


def parse_remote_image_message(
    text: str,
    default_prompt: str = DEFAULT_IMAGE_PROMPT,
) -> ParsedImageInput | None:
    """Parse one remote JSON image message into multimodal content.

    Returns ``None`` when *text* is not a JSON image payload.
    Raises ``ImageInputError`` when it looks like an image payload but is invalid.
    """
    stripped = text.strip()
    if not stripped or not stripped.startswith("{"):
        return None

    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return None

    if not isinstance(payload, dict):
        return None

    msg_type = str(payload.get("msgType", "")).strip().lower()
    if msg_type != REMOTE_IMAGE_MSG_TYPE:
        return None

    image_data_base64 = payload.get("imageDataBase64")
    if not isinstance(image_data_base64, list):
        raise ImageInputError("远程图片消息缺少 imageDataBase64。")

    if not image_data_base64:
        raise ImageInputError("远程图片消息中的 imageDataBase64 列表不能为空。")

    user_input = payload.get("userInput", "")
    if user_input is None:
        user_input = ""
    if not isinstance(user_input, str):
        user_input = str(user_input)

    prompt = user_input.strip() or default_prompt
    text_part = ContentPartTextParam(text=prompt)
    image_parts: list[ContentPartImageParam] = []
    mime_types: list[str] = []
    invalid_images: list[str] = []

    for index, image_data_item in enumerate(image_data_base64, start=1):
        try:
            if not isinstance(image_data_item, str) or not image_data_item.strip():
                raise ImageInputError(f"第 {index} 张图片数据为空或不是字符串。")

            mime_type, data_b64 = _normalize_remote_image_data(image_data_item)
            mime_types.append(mime_type)
            image_url = f"data:{mime_type};base64,{data_b64}"
            image_parts.append(
                ContentPartImageParam(
                    image_url=ImageURL(
                        url=image_url,
                        detail="auto",
                        media_type=_to_supported_media_type(mime_type),
                    )
                )
            )
        except ImageInputError as exc:
            invalid_images.append(str(exc))

    if not image_parts:
        summary = "；".join(invalid_images) if invalid_images else "未提供可用图片。"
        raise ImageInputError(f"远程图片消息中没有可用图片。{summary}")

    return ParsedImageInput(
        source_path=Path("<remote-image>"),
        user_text=prompt,
        mime_type=mime_types[0],
        content_parts=[text_part, *image_parts],
        invalid_images=invalid_images,
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


def _normalize_remote_image_data(image_data_base64: str) -> tuple[str, str]:
    raw_value = image_data_base64.strip()
    if not raw_value:
        raise ImageInputError("远程图片消息中的 imageDataBase64 不能为空。")

    mime_type: str | None = None
    data_b64 = raw_value

    if raw_value.startswith("data:"):
        match = re.match(
            r"^data:(?P<mime>[^;,]+);base64,(?P<data>.+)$",
            raw_value,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not match:
            raise ImageInputError("远程图片消息中的 data URL 格式无效。")
        mime_type = match.group("mime").lower().strip()
        data_b64 = match.group("data").strip()
        if mime_type == "image/jpg":
            mime_type = "image/jpeg"
        if mime_type not in SUPPORTED_IMAGE_MIME_TYPES:
            supported = ", ".join(sorted(SUPPORTED_IMAGE_MIME_TYPES))
            raise ImageInputError(f"远程图片 MIME 不支持：{mime_type}。支持的类型：{supported}")

    normalized_b64 = "".join(data_b64.split())
    try:
        raw_bytes = base64.b64decode(normalized_b64, validate=True)
    except (ValueError, binascii.Error) as e:
        raise ImageInputError("远程图片消息中的 imageDataBase64 不是合法的 base64。") from e

    if not raw_bytes:
        raise ImageInputError("远程图片消息解码后内容为空。")
    if len(raw_bytes) > MAX_IMAGE_BYTES:
        raise ImageInputError(
            f"远程图片过大：{len(raw_bytes)} 字节（最大 {MAX_IMAGE_BYTES} 字节）"
        )

    detected_mime_type = _detect_mime_type_from_bytes(raw_bytes)
    resolved_mime_type = mime_type or detected_mime_type
    if resolved_mime_type not in SUPPORTED_IMAGE_MIME_TYPES:
        supported = ", ".join(sorted(SUPPORTED_IMAGE_MIME_TYPES))
        raise ImageInputError(f"远程图片类型不支持。支持的类型：{supported}")

    return resolved_mime_type, base64.b64encode(raw_bytes).decode("ascii")


def _detect_mime_type_from_bytes(raw_bytes: bytes) -> str | None:
    if raw_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if raw_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if raw_bytes.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if raw_bytes.startswith(b"RIFF") and raw_bytes[8:12] == b"WEBP":
        return "image/webp"
    return None


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
