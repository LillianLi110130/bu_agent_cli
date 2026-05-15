"""Shared helpers for reading text files with conservative encoding fallback."""

from __future__ import annotations

import locale
from pathlib import Path


def read_text_with_fallback(path: Path) -> tuple[str, str]:
    """Read text using UTF-8 first, then a small set of locale-aware fallbacks."""
    errors: list[str] = []
    for encoding in candidate_encodings():
        try:
            return path.read_text(encoding=encoding), encoding
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")

    joined = "; ".join(errors) if errors else "no candidate encodings"
    raise UnicodeDecodeError(
        "text-decoder",
        b"",
        0,
        1,
        f"Unable to decode '{path}' with fallback encodings ({joined})",
    )


def candidate_encodings() -> list[str]:
    encodings = ["utf-8", "utf-8-sig"]

    preferred = locale.getpreferredencoding(False)
    if preferred:
        encodings.append(preferred)

    encodings.extend(["gb18030", "gbk", "cp936"])

    unique: list[str] = []
    seen: set[str] = set()
    for encoding in encodings:
        normalized = encoding.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(encoding)
    return unique
