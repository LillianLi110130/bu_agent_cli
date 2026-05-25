"""Image analysis tool for local screenshots and other image files."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Annotated, Any, Literal

from agent_core.llm.factory import create_chat_model
from agent_core.llm.messages import (
    ContentPartImageParam,
    ContentPartTextParam,
    ImageURL,
    UserMessage,
)
from agent_core.tools import Depends, tool
from config.model_config import (
    ModelPreset,
    get_auto_vision_preset,
    get_image_summary_preset,
    load_model_presets,
)
from tools.sandbox import SandboxContext, get_current_agent, get_sandbox_context

SUPPORTED_IMAGE_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
}
MAX_IMAGE_BYTES = 10 * 1024 * 1024
DEFAULT_IMAGE_ANALYSIS_PROMPT = (
    "请分析这张浏览器截图。识别页面主要区域、可见文本、按钮、输入框、弹窗、"
    "错误信息和可点击目标。如果能判断坐标，请给出大致 CSS 像素坐标；"
    "不确定的地方明确说明。"
)


def _detect_supported_mime_type(path: Path, raw_bytes: bytes | None = None) -> str | None:
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
    suffix_type = suffix_map.get(path.suffix.lower())
    if suffix_type:
        return suffix_type

    if raw_bytes is None:
        return None
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


def _resolve_vision_preset(presets: dict[str, ModelPreset], current_agent: Any) -> str | None:
    candidates = [
        get_image_summary_preset(presets),
        get_auto_vision_preset(presets),
    ]
    current_model = str(getattr(getattr(current_agent, "llm", None), "model", "") or "")
    for name, preset in presets.items():
        if preset.get("model") == current_model:
            candidates.append(name)
            break
    candidates.extend(presets.keys())

    seen: set[str] = set()
    for name in candidates:
        if not name or name in seen:
            continue
        seen.add(name)
        preset = presets.get(name)
        if preset and bool(preset.get("vision", False)):
            return name
    return None


def _read_image_as_data_url(path: Path) -> tuple[str, str]:
    if not path.exists():
        raise ValueError(f"未找到图片：{path}")
    if not path.is_file():
        raise ValueError(f"路径不是文件：{path}")

    try:
        raw_bytes = path.read_bytes()
    except OSError as e:
        raise ValueError(f"读取图片失败：{path}（{e}）") from e

    if not raw_bytes:
        raise ValueError(f"图片为空：{path}")
    if len(raw_bytes) > MAX_IMAGE_BYTES:
        raise ValueError(f"图片过大：{len(raw_bytes)} 字节（最大 {MAX_IMAGE_BYTES} 字节）")

    mime_type = _detect_supported_mime_type(path, raw_bytes)
    if mime_type is None:
        supported = ", ".join(sorted(SUPPORTED_IMAGE_MIME_TYPES))
        raise ValueError(f"不支持图片类型：{path.name}。支持的类型：{supported}")

    data_b64 = base64.b64encode(raw_bytes).decode("ascii")
    return mime_type, f"data:{mime_type};base64,{data_b64}"


@tool(
    "Analyze a local image file with a vision-capable model and return a concise text report.",
    name="analyze_image",
)
async def analyze_image(
    path: str,
    ctx: Annotated[SandboxContext, Depends(get_sandbox_context)],
    current_agent: Annotated[Any, Depends(get_current_agent)] = None,
    prompt: str = DEFAULT_IMAGE_ANALYSIS_PROMPT,
) -> str:
    """Analyze a local image path with the configured vision model.

    Args:
        path: Local image file path. Relative paths are resolved from the workspace.
        prompt: Analysis instructions for the vision model.
    """
    try:
        resolved = ctx.resolve_path(path)
        mime_type, data_url = _read_image_as_data_url(resolved)
    except Exception as e:
        return f"Error: {e}"

    presets = load_model_presets()
    vision_preset = _resolve_vision_preset(presets, current_agent)
    if not vision_preset:
        return (
            "Error: 未配置可用视觉预设。请在 config/model_presets.json 中配置 "
            "`vision: true` 和 `auto_vision_preset` 或 `image_summary_preset`。"
        )

    try:
        llm = create_chat_model(
            vision_preset,
            presets=presets,
            fallback_llm=getattr(current_agent, "llm", None),
        )
        response = await llm.ainvoke(
            messages=[
                UserMessage(
                    content=[
                        ContentPartTextParam(
                            text=(prompt or DEFAULT_IMAGE_ANALYSIS_PROMPT).strip()
                        ),
                        ContentPartImageParam(
                            image_url=ImageURL(
                                url=data_url,
                                detail="high",
                                media_type=_to_supported_media_type(mime_type),
                            )
                        ),
                    ]
                )
            ],
            tools=None,
            tool_choice=None,
        )
    except Exception as e:
        return f"Error: 视觉模型分析失败：{e}"

    content = (response.content or "").strip()
    return content or "Error: 视觉模型未返回可用文本。"
