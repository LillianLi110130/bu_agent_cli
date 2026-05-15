from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pytest

from tools.files import write
from tools.path_resolution import resolve_target_path
from tools.sandbox import SandboxContext, SecurityError
from tools.search import glob_search, grep


def _make_workspace() -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    temp_root = repo_root / ".pytest_tmp"
    temp_root.mkdir(exist_ok=True)
    workspace = temp_root / f"path-filters-{uuid.uuid4().hex[:8]}"
    workspace.mkdir()
    return workspace


def _make_temp_dir(prefix: str) -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    temp_root = repo_root / ".pytest_tmp"
    temp_root.mkdir(exist_ok=True)
    path = temp_root / f"{prefix}-{uuid.uuid4().hex[:8]}"
    path.mkdir()
    return path


def _write_ignore_file(workspace: Path, lines: list[str]) -> None:
    (workspace / ".tgagentignore").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _set_fake_home(monkeypatch: pytest.MonkeyPatch, home_dir: Path) -> Path:
    tg_agent_dir = home_dir / ".tg_agent"
    monkeypatch.setenv("HOME", str(home_dir))
    monkeypatch.setenv("USERPROFILE", str(home_dir))
    monkeypatch.delenv("HOMEDRIVE", raising=False)
    monkeypatch.delenv("HOMEPATH", raising=False)
    return tg_agent_dir


def test_sandbox_context_loads_ignore_rules_from_workspace_file() -> None:
    workspace = _make_workspace()
    secret_dir = workspace / "target"
    secret_dir.mkdir()
    _write_ignore_file(workspace, ["# build output", "target", "*.log"])

    try:
        ctx = SandboxContext.create(workspace)

        assert ctx.is_ignored(secret_dir)
        assert ctx.is_ignored(workspace / "app.log")
        assert ctx.ignore_file == workspace / ".tgagentignore"
        assert ctx.ignored_patterns == ["target", "*.log"]
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_resolve_path_blocks_paths_matched_by_ignore_pattern() -> None:
    workspace = _make_workspace()
    secret_file = workspace / "target" / "note.txt"
    secret_file.parent.mkdir()
    secret_file.write_text("top secret", encoding="utf-8")
    _write_ignore_file(workspace, ["target"])

    try:
        ctx = SandboxContext.create(workspace)

        with pytest.raises(SecurityError):
            ctx.resolve_path("target/note.txt")
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_sandbox_context_allows_user_tg_agent_directory_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _make_workspace()
    home_dir = _make_temp_dir("home")
    skill_file = _set_fake_home(monkeypatch, home_dir) / "skills" / "demo" / "SKILL.md"
    skill_file.parent.mkdir(parents=True)
    skill_file.write_text("demo", encoding="utf-8")

    try:
        ctx = SandboxContext.create(workspace)

        resolved = ctx.resolve_path("~/.tg_agent/skills/demo/SKILL.md")

        assert resolved == skill_file.resolve()
        assert any(path == skill_file.parents[2].resolve() for path in ctx.allowed_dirs)
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
        shutil.rmtree(home_dir, ignore_errors=True)


def test_resolve_target_path_skips_ignored_candidates() -> None:
    workspace = _make_workspace()
    ignored_file = workspace / "target" / "report.md"
    visible_file = workspace / "visible" / "report.md"
    ignored_file.parent.mkdir()
    visible_file.parent.mkdir()
    ignored_file.write_text("hidden", encoding="utf-8")
    visible_file.write_text("shown", encoding="utf-8")
    _write_ignore_file(workspace, ["target"])

    try:
        ctx = SandboxContext.create(workspace)

        resolved = resolve_target_path("report.md", ctx, kind="file")

        assert resolved == visible_file.resolve()
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_root_anchored_pattern_only_matches_workspace_root() -> None:
    workspace = _make_workspace()
    root_match = workspace / "context" / "template"
    nested_non_match = workspace / "nested" / "context" / "template"
    root_match.mkdir(parents=True)
    nested_non_match.mkdir(parents=True)
    _write_ignore_file(workspace, ["/context/template"])

    try:
        ctx = SandboxContext.create(workspace)

        assert ctx.is_ignored(root_match)
        assert not ctx.is_ignored(nested_non_match)
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


@pytest.mark.anyio
async def test_write_tool_rejects_paths_inside_ignored_directory() -> None:
    workspace = _make_workspace()
    (workspace / "target").mkdir()
    _write_ignore_file(workspace, ["target"])

    try:
        ctx = SandboxContext.create(workspace)

        result = await write.func("target/new.txt", "blocked", ctx=ctx)

        assert result.startswith("Security error:")
        assert ".tgagentignore" in result
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


@pytest.mark.anyio
async def test_glob_search_skips_ignored_patterns() -> None:
    workspace = _make_workspace()
    visible_file = workspace / "src" / "main.py"
    hidden_file = workspace / "target" / "hidden.py"
    visible_file.parent.mkdir()
    hidden_file.parent.mkdir()
    visible_file.write_text("print('visible')", encoding="utf-8")
    hidden_file.write_text("print('hidden')", encoding="utf-8")
    _write_ignore_file(workspace, ["target"])

    try:
        ctx = SandboxContext.create(workspace)

        result = await glob_search.func("**/*.py", ctx=ctx)

        assert "src\\main.py" in result or "src/main.py" in result
        assert "hidden.py" not in result
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


@pytest.mark.anyio
async def test_glob_search_can_match_directories() -> None:
    workspace = _make_workspace()
    directory = workspace / "src" / "nested"
    directory.mkdir(parents=True)
    (directory / "module.py").write_text("content", encoding="utf-8")

    try:
        ctx = SandboxContext.create(workspace)

        result = await glob_search.func("**/nested", ctx=ctx, kind="dir")

        assert "src\\nested" in result or "src/nested" in result
        assert "module.py" not in result
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


@pytest.mark.anyio
async def test_glob_search_uses_pathlib_glob_semantics() -> None:
    workspace = _make_workspace()
    root_file = workspace / "src" / "a.py"
    nested_file = workspace / "src" / "nested" / "b.py"
    unrelated_file = workspace / "other" / "src" / "c.py"
    root_file.parent.mkdir()
    nested_file.parent.mkdir()
    unrelated_file.parent.mkdir(parents=True)
    root_file.write_text("root", encoding="utf-8")
    nested_file.write_text("nested", encoding="utf-8")
    unrelated_file.write_text("unrelated", encoding="utf-8")

    try:
        ctx = SandboxContext.create(workspace)

        result = await glob_search.func("src/**/*.py", ctx=ctx)

        assert "src\\a.py" in result or "src/a.py" in result
        assert "src\\nested\\b.py" in result or "src/nested/b.py" in result
        assert "other" not in result
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


@pytest.mark.anyio
async def test_glob_search_reports_truncated_results() -> None:
    workspace = _make_workspace()
    for index in range(55):
        (workspace / f"file_{index:02d}.txt").write_text("content", encoding="utf-8")

    try:
        ctx = SandboxContext.create(workspace)

        result = await glob_search.func("*.txt", ctx=ctx)

        assert result.startswith("Found 50 file(s), showing first 50:")
        assert "... (truncated; refine pattern or path)" in result
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


@pytest.mark.anyio
async def test_glob_search_allows_user_tg_agent_skills_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _make_workspace()
    home_dir = _make_temp_dir("home")
    skill_file = _set_fake_home(monkeypatch, home_dir) / "skills" / "demo" / "SKILL.md"
    skill_file.parent.mkdir(parents=True)
    skill_file.write_text("demo", encoding="utf-8")

    try:
        ctx = SandboxContext.create(workspace)

        result = await glob_search.func("**/SKILL.md", ctx=ctx, path="~/.tg_agent/skills")

        assert "SKILL.md" in result
        assert "Security error:" not in result
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
        shutil.rmtree(home_dir, ignore_errors=True)


@pytest.mark.anyio
async def test_grep_skips_ignored_patterns() -> None:
    workspace = _make_workspace()
    visible_file = workspace / "docs" / "public.txt"
    hidden_file = workspace / "target" / "private.txt"
    visible_file.parent.mkdir()
    hidden_file.parent.mkdir()
    visible_file.write_text("no match here", encoding="utf-8")
    hidden_file.write_text("needle", encoding="utf-8")
    _write_ignore_file(workspace, ["target"])

    try:
        ctx = SandboxContext.create(workspace)

        result = await grep.func("needle", ctx=ctx)

        assert result == "No matches for: needle"
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


@pytest.mark.anyio
async def test_grep_skips_binary_files() -> None:
    workspace = _make_workspace()
    binary_file = workspace / "blob.bin"
    binary_file.write_bytes(b"\x00needle\x01\x02")

    try:
        ctx = SandboxContext.create(workspace)

        result = await grep.func("needle", ctx=ctx)

        assert result == "No matches for: needle"
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


@pytest.mark.anyio
async def test_grep_reports_truncated_results() -> None:
    workspace = _make_workspace()
    target = workspace / "matches.txt"
    target.write_text("\n".join(f"needle {index}" for index in range(55)), encoding="utf-8")

    try:
        ctx = SandboxContext.create(workspace)

        result = await grep.func("needle", ctx=ctx)

        assert result.endswith("... (truncated)")
        assert "matches.txt:1: needle 0" in result
        assert "matches.txt:50: needle 49" in result
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_sandbox_context_ignores_common_large_directories_by_default() -> None:
    workspace = _make_workspace()
    git_dir = workspace / ".git"
    node_modules_dir = workspace / "node_modules"
    git_dir.mkdir()
    node_modules_dir.mkdir()

    try:
        ctx = SandboxContext.create(workspace)

        assert ctx.is_ignored(git_dir)
        assert ctx.is_ignored(node_modules_dir)
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_sandbox_context_allows_user_tgagent_even_before_directory_exists(
    monkeypatch,
) -> None:
    workspace = _make_workspace()
    home_dir = workspace / "home"
    monkeypatch.setenv("HOME", str(home_dir))

    try:
        ctx = SandboxContext.create(workspace)

        assert ctx.is_allowed(home_dir / ".tg_agent" / "skills" / "demo" / "SKILL.md")
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
