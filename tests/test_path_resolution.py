from __future__ import annotations

import shutil
import uuid
from pathlib import Path
import importlib

import pytest

from tools.files import read
from tools.path_resolution import AmbiguousPathError, resolve_target_path
from tools.sandbox import SandboxContext


def _make_workspace() -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    temp_root = repo_root / ".pytest_tmp"
    temp_root.mkdir(exist_ok=True)
    workspace = temp_root / f"path-resolution-{uuid.uuid4().hex[:8]}"
    workspace.mkdir()
    return workspace


def test_resolve_target_path_matches_real_file_from_path_with_inserted_spaces():
    workspace = _make_workspace()
    real_dir = workspace / "AI价值链探索-特色存款案例"
    real_dir.mkdir()
    real_file = real_dir / "特色存款专区_价值链分析.md"
    real_file.write_text("content", encoding="utf-8")
    ctx = SandboxContext.create(workspace)

    try:
        query = str(workspace / "AI 价值链探索 - 特色存款案例" / "特色存款专区_价值链分析.md")

        resolved = resolve_target_path(query, ctx, kind="file")

        assert resolved == real_file.resolve()
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_resolve_target_path_from_partial_filename():
    workspace = _make_workspace()
    real_dir = workspace / "资料"
    real_dir.mkdir()
    real_file = real_dir / "特色存款专区_价值链分析.md"
    real_file.write_text("content", encoding="utf-8")
    ctx = SandboxContext.create(workspace)

    try:
        resolved = resolve_target_path("价值链分析", ctx, kind="file")

        assert resolved == real_file.resolve()
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_resolve_target_path_raises_for_ambiguous_filename():
    workspace = _make_workspace()
    first = workspace / "甲目录"
    second = workspace / "乙目录"
    first.mkdir()
    second.mkdir()
    (first / "报告.md").write_text("a", encoding="utf-8")
    (second / "报告.md").write_text("b", encoding="utf-8")
    ctx = SandboxContext.create(workspace)

    try:
        with pytest.raises(AmbiguousPathError):
            resolve_target_path("报告.md", ctx, kind="file")
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


@pytest.mark.anyio
async def test_read_tool_uses_resolved_real_path():
    workspace = _make_workspace()
    real_dir = workspace / "AI价值链探索-特色存款案例"
    real_dir.mkdir()
    real_file = real_dir / "特色存款专区_价值链分析.md"
    real_file.write_text("第一行\n第二行", encoding="utf-8")
    ctx = SandboxContext.create(workspace)

    try:
        query = str(workspace / "AI 价值链探索 - 特色存款案例" / "特色存款专区_价值链分析.md")

        result = await read.func(query, ctx=ctx)

        assert "第一行" in result
        assert "第二行" in result
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_decode_process_stream_handles_gbk_output(monkeypatch: pytest.MonkeyPatch):
    bash_module = importlib.import_module("tools.bash")
    monkeypatch.setattr(bash_module, "_shell_output_encodings", lambda: ["utf-8", "gbk"])

    decoded = bash_module._decode_process_stream("中文目录".encode("gbk"))

    assert decoded == "中文目录"
