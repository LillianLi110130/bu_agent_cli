from __future__ import annotations

import json
import re
from pathlib import Path

import agent_core.release_notes as release_notes_module
from agent_core.release_notes import load_release_notes


ROOT = Path(__file__).resolve().parent.parent


def _project_version() -> str:
    pyproject_text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'(?m)^version\s*=\s*"([^"]+)"\s*$', pyproject_text)
    if match is None:
        raise AssertionError("Could not locate project version in pyproject.toml")
    return match.group(1)


def test_packaged_release_notes_match_project_version():
    version = _project_version()
    payload = json.loads((ROOT / "config" / "release_notes.json").read_text(encoding="utf-8"))

    assert payload["version"] == version
    assert isinstance(payload["published_at"], str)
    assert payload["published_at"]
    assert isinstance(payload["notes"], list)
    assert payload["notes"]
    assert all(isinstance(item, str) and item.strip() for item in payload["notes"])


def test_load_release_notes_requires_matching_version(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "release_notes.json").write_text(
        json.dumps(
            {
                "version": "1.2.3",
                "published_at": "2026-05-26",
                "notes": ["第一条", "第二条"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(release_notes_module, "application_root", lambda: tmp_path)

    notes = load_release_notes(expected_version="1.2.3")

    assert notes is not None
    assert notes.version == "1.2.3"
    assert notes.published_at == "2026-05-26"
    assert notes.notes == ["第一条", "第二条"]
    assert load_release_notes(expected_version="1.2.4") is None
