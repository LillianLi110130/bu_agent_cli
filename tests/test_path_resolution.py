from __future__ import annotations

import shutil
import uuid
from pathlib import Path
import importlib
import json

import pytest

from tools.files import read
from tools.resolve_path import resolve_path
from tools.search import glob_search
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


def test_resolve_target_path_matches_chinese_punctuation_variants():
    workspace = _make_workspace()
    real_dir = workspace / "《AI价值链探索》-特色存款案例"
    real_dir.mkdir()
    real_file = real_dir / "数据表解释-市场维度数据集0324.xlsx"
    real_file.write_text("content", encoding="utf-8")
    ctx = SandboxContext.create(workspace)

    try:
        query = "AI价值链探索（特色存款案例）/数据表解释（市场维度数据集）0324.xlsx"

        resolved = resolve_target_path(query, ctx, kind="file")

        assert resolved == real_file.resolve()
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


@pytest.mark.anyio
async def test_glob_search_uses_resolved_directory_path():
    workspace = _make_workspace()
    real_dir = workspace / "AI价值链探索-特色存款案例"
    real_dir.mkdir()
    (real_dir / "特色存款专区_价值链分析.md").write_text("content", encoding="utf-8")
    ctx = SandboxContext.create(workspace)

    try:
        result = await glob_search.func("*.md", ctx=ctx, path="AI 价值链探索 - 特色存款案例")

        assert "特色存款专区_价值链分析.md" in result
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


@pytest.mark.anyio
async def test_glob_search_reports_missing_path_instead_of_empty_match():
    workspace = _make_workspace()
    ctx = SandboxContext.create(workspace)

    try:
        result = await glob_search.func("*.md", ctx=ctx, path="不存在的目录")

        assert result == "Path not found: 不存在的目录"
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_format_bash_result_includes_return_code_and_streams():
    bash_module = importlib.import_module("tools.bash")

    payload = json.loads(
        bash_module._format_bash_result(
            command="echo hello",
            cwd="D:/workspace",
            returncode=3,
            stdout="out\n",
            stderr="err\n",
            timed_out=False,
        )
    )

    assert payload["ok"] is False
    assert payload["returncode"] == 3
    assert payload["stdout"] == "out\n"
    assert payload["stderr"] == "err\n"


@pytest.mark.anyio
async def test_resolve_path_tool_returns_real_path_for_fuzzy_query():
    workspace = _make_workspace()
    real_dir = workspace / "AI价值链探索-特色存款案例"
    real_dir.mkdir()
    real_file = real_dir / "数据表解释-市场维度数据集0324.xlsx"
    real_file.write_text("content", encoding="utf-8")
    ctx = SandboxContext.create(workspace)

    try:
        result = await resolve_path.func("市场维度数据集0324.xlsx", ctx=ctx, kind="file")
        payload = json.loads(result)

        assert payload["ok"] is True
        assert payload["kind"] == "file"
        assert payload["resolved_path"] == str(real_file.resolve())
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


@pytest.mark.anyio
async def test_resolve_path_tool_reports_ambiguous_candidates():
    workspace = _make_workspace()
    first = workspace / "甲目录"
    second = workspace / "乙目录"
    first.mkdir()
    second.mkdir()
    (first / "报告.md").write_text("a", encoding="utf-8")
    (second / "报告.md").write_text("b", encoding="utf-8")
    ctx = SandboxContext.create(workspace)

    try:
        result = await resolve_path.func("报告.md", ctx=ctx, kind="file")
        payload = json.loads(result)

        assert payload["ok"] is False
        assert payload["reason"] == "ambiguous"
        assert len(payload["candidates"]) == 2
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
