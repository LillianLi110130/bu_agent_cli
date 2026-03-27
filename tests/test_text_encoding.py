from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pytest

from tools.files import edit, read
from tools.sandbox import SandboxContext
from tools.search import grep
from tools.text_encoding import read_text_with_fallback


def _make_workspace() -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    temp_root = repo_root / ".pytest_tmp"
    temp_root.mkdir(exist_ok=True)
    workspace = temp_root / f"text-encoding-{uuid.uuid4().hex[:8]}"
    workspace.mkdir()
    return workspace


def test_read_text_with_fallback_reads_gbk_content():
    workspace = _make_workspace()
    path = workspace / "中文目录" / "示例.txt"
    path.parent.mkdir(parents=True)
    path.write_bytes("你好，GBK".encode("gbk"))

    try:
        content, encoding = read_text_with_fallback(path)

        assert content == "你好，GBK"
        assert encoding.lower() in {"gb18030", "gbk", "cp936"}
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


@pytest.mark.asyncio
async def test_read_tool_reads_gbk_file_with_chinese_path():
    workspace = _make_workspace()
    path = workspace / "中文目录" / "示例.txt"
    path.parent.mkdir(parents=True)
    path.write_bytes("第一行\n第二行".encode("gbk"))
    ctx = SandboxContext.create(workspace)

    try:
        result = await read.func("中文目录/示例.txt", ctx=ctx)

        assert "[Lines 1-2 of 2]" in result
        assert "第一行" in result
        assert "第二行" in result
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


@pytest.mark.asyncio
async def test_edit_tool_preserves_original_gbk_encoding():
    workspace = _make_workspace()
    path = workspace / "中文目录" / "示例.txt"
    path.parent.mkdir(parents=True)
    path.write_bytes("你好，世界".encode("gbk"))
    ctx = SandboxContext.create(workspace)

    try:
        result = await edit.func("中文目录/示例.txt", old_string="世界", new_string="朋友", ctx=ctx)

        assert "Replaced 1 occurrence" in result
        assert path.read_bytes().decode("gbk") == "你好，朋友"
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


@pytest.mark.asyncio
async def test_grep_tool_searches_gbk_content():
    workspace = _make_workspace()
    path = workspace / "中文目录" / "示例.txt"
    path.parent.mkdir(parents=True)
    path.write_bytes("这里有中文关键字".encode("gbk"))
    ctx = SandboxContext.create(workspace)

    try:
        result = await grep.func("关键字", ctx=ctx, path="中文目录")

        assert "中文目录" in result
        assert "示例.txt:1: 这里有中文关键字" in result
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
