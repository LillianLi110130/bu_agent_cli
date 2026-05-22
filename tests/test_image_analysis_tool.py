from __future__ import annotations

from pathlib import Path

import pytest

from tools import ALL_TOOLS
from tools.image_analysis import analyze_image
from tools.sandbox import SandboxContext, get_current_agent, get_sandbox_context

PNG_1X1 = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01"
    b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


class _FakeCompletion:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeLLM:
    model = "text-model"
    base_url = None
    api_key = None

    def __init__(self) -> None:
        self.calls = []

    async def ainvoke(self, messages, tools=None, tool_choice=None, **kwargs):
        self.calls.append((messages, tools, tool_choice, kwargs))
        return _FakeCompletion("页面包含一个登录按钮，坐标约为 x=100 y=200。")


class _FakeAgent:
    def __init__(self) -> None:
        self.llm = _FakeLLM()


def _ctx(tmp_path: Path) -> SandboxContext:
    return SandboxContext.create(tmp_path)


def test_analyze_image_is_registered() -> None:
    assert analyze_image in ALL_TOOLS
    assert analyze_image.name == "analyze_image"


@pytest.mark.asyncio
async def test_analyze_image_invokes_vision_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "shot.png"
    image_path.write_bytes(PNG_1X1)
    ctx = _ctx(tmp_path)
    agent = _FakeAgent()
    fake_vision = _FakeLLM()

    monkeypatch.setattr(
        "tools.image_analysis.load_model_presets",
        lambda: {
            "text": {"model": "text-model", "vision": False},
            "vision": {"model": "vision-model", "vision": True},
        },
    )
    monkeypatch.setattr("tools.image_analysis.get_image_summary_preset", lambda presets: "vision")
    monkeypatch.setattr("tools.image_analysis.get_auto_vision_preset", lambda presets: "vision")
    monkeypatch.setattr("tools.image_analysis.create_chat_model", lambda *_, **__: fake_vision)

    result = await analyze_image.execute(
        _overrides={get_sandbox_context: lambda: ctx, get_current_agent: lambda: agent},
        path=str(image_path),
        prompt="找按钮",
    )

    assert "登录按钮" in result
    assert fake_vision.calls
    message = fake_vision.calls[0][0][0]
    assert message.content[0].text == "找按钮"
    assert message.content[1].image_url.url.startswith("data:image/png;base64,")
    assert message.content[1].image_url.detail == "high"


@pytest.mark.asyncio
async def test_analyze_image_reports_missing_file(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    agent = _FakeAgent()

    result = await analyze_image.execute(
        _overrides={get_sandbox_context: lambda: ctx, get_current_agent: lambda: agent},
        path=str(tmp_path / "missing.png"),
    )

    assert result.startswith("Error:")
    assert "未找到图片" in result


@pytest.mark.asyncio
async def test_analyze_image_rejects_non_image_file(tmp_path: Path) -> None:
    text_path = tmp_path / "note.txt"
    text_path.write_text("not an image", encoding="utf-8")
    ctx = _ctx(tmp_path)
    agent = _FakeAgent()

    result = await analyze_image.execute(
        _overrides={get_sandbox_context: lambda: ctx, get_current_agent: lambda: agent},
        path=str(text_path),
    )

    assert result.startswith("Error:")
    assert "不支持图片类型" in result


@pytest.mark.asyncio
async def test_analyze_image_rejects_large_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "large.png"
    image_path.write_bytes(PNG_1X1)
    ctx = _ctx(tmp_path)
    agent = _FakeAgent()

    monkeypatch.setattr("tools.image_analysis.MAX_IMAGE_BYTES", 1)

    result = await analyze_image.execute(
        _overrides={get_sandbox_context: lambda: ctx, get_current_agent: lambda: agent},
        path=str(image_path),
    )

    assert result.startswith("Error:")
    assert "图片过大" in result


@pytest.mark.asyncio
async def test_analyze_image_reports_missing_vision_preset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "shot.png"
    image_path.write_bytes(PNG_1X1)
    ctx = _ctx(tmp_path)
    agent = _FakeAgent()

    monkeypatch.setattr(
        "tools.image_analysis.load_model_presets",
        lambda: {"text": {"model": "text-model", "vision": False}},
    )
    monkeypatch.setattr("tools.image_analysis.get_image_summary_preset", lambda presets: None)
    monkeypatch.setattr("tools.image_analysis.get_auto_vision_preset", lambda presets: None)

    result = await analyze_image.execute(
        _overrides={get_sandbox_context: lambda: ctx, get_current_agent: lambda: agent},
        path=str(image_path),
    )

    assert result.startswith("Error:")
    assert "未配置可用视觉预设" in result
