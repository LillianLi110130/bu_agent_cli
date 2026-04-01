"""Deterministic file target resolution for tool inputs."""

from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from tools.sandbox import SandboxContext, SecurityError

TargetKind = Literal["file", "dir", "any"]


class PathResolutionError(ValueError):
    """Base error for file target resolution failures."""


class PathNotFoundError(PathResolutionError):
    """Raised when no candidate path can be found."""


class AmbiguousPathError(PathResolutionError):
    """Raised when multiple candidate paths match the same query."""

    def __init__(self, query: str, candidates: list[Path]):
        self.query = query
        self.candidates = candidates
        preview = "\n".join(f"- {path}" for path in candidates[:5])
        suffix = "\n..." if len(candidates) > 5 else ""
        super().__init__(f"Ambiguous path for '{query}'. Candidates:\n{preview}{suffix}")


@dataclass(frozen=True)
class _Match:
    path: Path
    score: int


@dataclass(frozen=True)
class _QuerySignature:
    strict_path: str
    relaxed_path: str
    strict_name: str
    relaxed_name: str
    suffix: str
    parts: tuple[str, ...]


@dataclass(frozen=True)
class _CandidateSignature:
    strict_path: str
    relaxed_path: str
    strict_name: str
    relaxed_name: str


_SEPARATOR_TRANSLATION = str.maketrans(
    {
        "\uff08": "(",
        "\uff09": ")",
        "\u3010": "[",
        "\u3011": "]",
        "\u3008": "<",
        "\u3009": ">",
        "\uff0c": ",",
        "\u3002": ".",
        "\u3001": ",",
        "\uff1b": ";",
        "\uff1a": ":",
        "\u201c": '"',
        "\u201d": '"',
        "\u2018": "'",
        "\u2019": "'",
        "\u2014": "-",
        "\u2013": "-",
        "\uff0d": "-",
        "\u2015": "-",
        "\u2212": "-",
        "\u2500": "-",
        "\u2010": "-",
        "\u00b7": "-",
        "\u30fb": "-",
        "\u3000": " ",
    }
)


def resolve_target_path(
    query: str,
    ctx: SandboxContext,
    *,
    kind: TargetKind = "file",
) -> Path:
    """Resolve a user-provided file target to a real path in the sandbox."""
    cleaned = query.strip()
    if not cleaned:
        raise PathNotFoundError("Empty path query.")

    direct = _resolve_direct_path(cleaned, ctx, kind=kind)
    if direct is not None:
        return direct

    matches = _search_matches(cleaned, ctx, kind=kind)
    if not matches:
        raise PathNotFoundError(f"Path not found: {query}")

    top = matches[0]
    second = matches[1] if len(matches) > 1 else None
    if len(matches) == 1:
        return top.path
    if top.score >= 190 and (second is None or top.score >= second.score + 20):
        return top.path
    if top.score >= 140 and second is not None and top.score >= second.score + 45:
        return top.path

    raise AmbiguousPathError(cleaned, [match.path for match in matches[:5]])


def _resolve_direct_path(query: str, ctx: SandboxContext, *, kind: TargetKind) -> Path | None:
    last_security_error: SecurityError | None = None
    for candidate_text in _candidate_query_paths(query):
        try:
            resolved = ctx.resolve_path(candidate_text)
        except SecurityError as exc:
            last_security_error = exc
            continue
        except Exception:
            continue
        if _matches_kind(resolved, kind):
            return resolved

    if last_security_error is not None and _looks_like_path(query):
        raise last_security_error
    return None


def _candidate_query_paths(query: str) -> list[str]:
    candidates = [query]
    expanded = str(Path(query).expanduser())
    if expanded != query:
        candidates.append(expanded)
    return candidates


def _build_query_signature(query: str) -> _QuerySignature:
    query_name = Path(query).name or query
    return _QuerySignature(
        strict_path=_normalize_strict(query),
        relaxed_path=_normalize_relaxed(query),
        strict_name=_normalize_strict(query_name),
        relaxed_name=_normalize_relaxed(query_name),
        suffix=Path(query_name).suffix.lower(),
        parts=tuple(part for part in _split_query_parts(query) if part),
    )


def _build_candidate_signature(candidate: Path) -> _CandidateSignature:
    candidate_text = str(candidate)
    return _CandidateSignature(
        strict_path=_normalize_strict(candidate_text),
        relaxed_path=_normalize_relaxed(candidate_text),
        strict_name=_normalize_strict(candidate.name),
        relaxed_name=_normalize_relaxed(candidate.name),
    )


def _resolve_search_root(root: Path, ctx: SandboxContext) -> Path | None:
    try:
        resolved_root = root.resolve()
    except Exception:
        return None
    if not resolved_root.exists() or ctx.is_ignored(resolved_root):
        return None
    return resolved_root


def _append_candidate_if_unique(
    candidate: Path,
    candidates: list[Path],
    seen: set[Path],
    *,
    kind: TargetKind,
) -> None:
    if candidate in seen or not _matches_kind(candidate, kind):
        return
    seen.add(candidate)
    candidates.append(candidate)


def _collect_file_candidates(
    current_path: Path,
    files: list[str],
    ctx: SandboxContext,
    candidates: list[Path],
    seen: set[Path],
    *,
    kind: TargetKind,
) -> None:
    for file_name in files:
        candidate = current_path / file_name
        if ctx.is_ignored(candidate):
            continue
        _append_candidate_if_unique(candidate, candidates, seen, kind=kind)


def _collect_walk_candidates(
    resolved_root: Path,
    ctx: SandboxContext,
    candidates: list[Path],
    seen: set[Path],
    *,
    kind: TargetKind,
) -> None:
    for current, dirs, files in os.walk(resolved_root):
        current_path = Path(current)
        dirs[:] = [name for name in dirs if not ctx.is_ignored(current_path / name)]

        if current_path != resolved_root:
            _append_candidate_if_unique(current_path, candidates, seen, kind=kind)

        if kind == "dir":
            continue

        _collect_file_candidates(current_path, files, ctx, candidates, seen, kind=kind)


def _iter_unique_search_candidates(
    roots: list[Path],
    ctx: SandboxContext,
    *,
    kind: TargetKind,
) -> list[Path]:
    candidates: list[Path] = []
    seen: set[Path] = set()

    for root in roots:
        resolved_root = _resolve_search_root(root, ctx)
        if resolved_root is None:
            continue

        _append_candidate_if_unique(resolved_root, candidates, seen, kind=kind)
        if not resolved_root.is_dir():
            continue

        _collect_walk_candidates(resolved_root, ctx, candidates, seen, kind=kind)

    return candidates


def _score_candidate_name_match(
    query_signature: _QuerySignature,
    candidate_signature: _CandidateSignature,
) -> int:
    if candidate_signature.strict_name == query_signature.strict_name:
        return 240
    if candidate_signature.relaxed_name == query_signature.relaxed_name:
        return 220
    if query_signature.relaxed_name and query_signature.relaxed_name in candidate_signature.relaxed_name:
        return 170
    if candidate_signature.relaxed_name and candidate_signature.relaxed_name in query_signature.relaxed_name:
        return 150
    return 0


def _score_candidate_path_match(
    query_signature: _QuerySignature,
    candidate_signature: _CandidateSignature,
) -> int:
    if candidate_signature.strict_path == query_signature.strict_path:
        return 260
    if query_signature.relaxed_path and query_signature.relaxed_path == candidate_signature.relaxed_path:
        return 240
    if query_signature.relaxed_path and query_signature.relaxed_path in candidate_signature.relaxed_path:
        return 120
    return 0


def _score_candidate_suffix_match(query_signature: _QuerySignature, candidate: Path) -> int:
    if query_signature.suffix and candidate.suffix.lower() == query_signature.suffix:
        return 25
    return 0


def _score_candidate_part_matches(
    query_signature: _QuerySignature,
    candidate_signature: _CandidateSignature,
) -> int:
    return sum(15 for part in query_signature.parts if part in candidate_signature.relaxed_path)


def _score_candidate(query_signature: _QuerySignature, candidate: Path) -> int:
    candidate_signature = _build_candidate_signature(candidate)
    return (
        _score_candidate_name_match(query_signature, candidate_signature)
        + _score_candidate_path_match(query_signature, candidate_signature)
        + _score_candidate_suffix_match(query_signature, candidate)
        + _score_candidate_part_matches(query_signature, candidate_signature)
    )


def _search_matches(query: str, ctx: SandboxContext, *, kind: TargetKind) -> list[_Match]:
    roots = _search_roots(query, ctx)
    query_signature = _build_query_signature(query)
    results: list[_Match] = []
    for candidate in _iter_unique_search_candidates(roots, ctx, kind=kind):
        score = _score_candidate(query_signature, candidate)
        if score >= 80:
            results.append(_Match(path=candidate, score=score))

    results.sort(key=lambda item: (-item.score, len(str(item.path)), str(item.path).lower()))
    return results


def _search_roots(query: str, ctx: SandboxContext) -> list[Path]:
    roots: list[Path] = []
    if _looks_like_path(query):
        candidate_path = Path(query).expanduser()
        if not candidate_path.is_absolute():
            candidate_path = ctx.working_dir / candidate_path
        for ancestor in (candidate_path, *candidate_path.parents):
            try:
                resolved = ancestor.resolve()
            except Exception:
                continue
            if not resolved.exists() or not resolved.is_dir():
                continue
            if not ctx.is_allowed(resolved) or ctx.is_ignored(resolved):
                continue
            roots.append(resolved)
            break

    for path in ctx.allowed_dirs:
        try:
            resolved = path.resolve()
        except Exception:
            continue
        if resolved.exists() and not ctx.is_ignored(resolved):
            roots.append(resolved)

    deduped: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if root in seen:
            continue
        seen.add(root)
        deduped.append(root)
    return deduped


def _looks_like_path(text: str) -> bool:
    return (
        Path(text).is_absolute()
        or "\\" in text
        or "/" in text
        or text.startswith((".", "~"))
        or bool(re.search(r"\.[A-Za-z0-9]{1,8}$", text.strip()))
    )


def _matches_kind(path: Path, kind: TargetKind) -> bool:
    if not path.exists():
        return False
    if kind == "file":
        return path.is_file()
    if kind == "dir":
        return path.is_dir()
    return True


def _normalize_strict(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).translate(_SEPARATOR_TRANSLATION).strip().lower()
    normalized = normalized.replace("\\", "/")
    normalized = re.sub(r"/+", "/", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"[-_.:,;]+", "-", normalized)
    normalized = re.sub(r"\(\s*", "(", normalized)
    normalized = re.sub(r"\s*\)", ")", normalized)
    return normalized


def _normalize_relaxed(text: str) -> str:
    normalized = _normalize_strict(text)
    normalized = re.sub(r"[\s\-_.,:;()\[\]{}<>\"'`]+", "", normalized)
    return normalized


def _split_query_parts(query: str) -> list[str]:
    strict = _normalize_strict(query)
    parts = [part for part in re.split(r"[\\/]+", strict) if part]
    tokens: list[str] = []
    for part in parts:
        tokens.append(_normalize_relaxed(part))
        tokens.extend(token for token in re.split(r"[\s._,:;()<>\[\]{}\"'`-]+", part) if token)
    return [_normalize_relaxed(token) for token in tokens if token]
