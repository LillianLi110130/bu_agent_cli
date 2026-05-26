from __future__ import annotations

import json
from dataclasses import dataclass

from agent_core.runtime_paths import application_root


@dataclass(frozen=True)
class ReleaseNotes:
    version: str
    published_at: str | None
    notes: list[str]


def load_release_notes(expected_version: str | None = None) -> ReleaseNotes | None:
    """Load packaged release notes for the current CLI build."""
    notes_path = application_root() / "config" / "release_notes.json"
    if not notes_path.exists():
        return None

    try:
        payload = json.loads(notes_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(payload, dict):
        return None

    version = payload.get("version")
    if not isinstance(version, str) or not version.strip():
        return None
    version = version.strip()
    if expected_version is not None and version != expected_version:
        return None

    published_at = payload.get("published_at")
    if not isinstance(published_at, str) or not published_at.strip():
        published_at = None
    else:
        published_at = published_at.strip()

    raw_notes = payload.get("notes")
    if not isinstance(raw_notes, list):
        return None

    notes = [item.strip() for item in raw_notes if isinstance(item, str) and item.strip()]
    return ReleaseNotes(version=version, published_at=published_at, notes=notes)
