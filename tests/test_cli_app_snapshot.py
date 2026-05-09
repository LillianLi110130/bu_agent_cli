from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cli.app import TGAgentCLI
from tools import SandboxContext


def test_build_project_snapshot_skips_ignored_and_deep_paths(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / ".tgagentignore").write_text("ignored_dir\n", encoding="utf-8")
    (workspace / "README.md").write_text("Project overview", encoding="utf-8")
    (workspace / "requirements.txt").write_text("pytest\n", encoding="utf-8")
    (workspace / "src").mkdir()
    (workspace / "src" / "main.py").write_text("print('visible')\n", encoding="utf-8")
    (workspace / "ignored_dir").mkdir()
    (workspace / "ignored_dir" / "secret.txt").write_text("hidden\n", encoding="utf-8")
    deep_dir = workspace / "a" / "b" / "c" / "d" / "e"
    deep_dir.mkdir(parents=True)
    (deep_dir / "too_deep.txt").write_text("skip me\n", encoding="utf-8")

    cli = object.__new__(TGAgentCLI)
    cli._ctx = SandboxContext.create(workspace)

    snapshot = cli._build_project_snapshot()

    assert f"Project root: {workspace}" in snapshot
    assert "Tree (depth 4):" in snapshot
    assert "Key files:" in snapshot
    assert "Files (samples):" in snapshot
    assert "README.md" in snapshot
    assert "Project overview" in snapshot
    assert "src/main.py" in snapshot or "src\\main.py" in snapshot
    assert "ignored_dir/" not in snapshot
    assert "secret.txt" not in snapshot
    assert "too_deep.txt" not in snapshot
