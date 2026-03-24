"""Deterministic file target resolution for tool inputs."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from tools.sandbox import SandboxContext

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
    for candidate_text in _candidate_query_paths(query):
        try:
            resolved = ctx.resolve_path(candidate_text)
        except Exception:
            continue
        if _matches_kind(resolved, kind):
            return resolved
    return None


def _candidate_query_paths(query: str) -> list[str]:
    candidates = [query]
    expanded = str(Path(query).expanduser())
    if expanded != query:
        candidates.append(expanded)
    return candidates


def _search_matches(query: str, ctx: SandboxContext, *, kind: TargetKind) -> list[_Match]:
    roots = _search_roots(query, ctx)
    results: list[_Match] = []
    seen: set[Path] = set()
    strict_query = _normalize_strict(query)
    relaxed_query = _normalize_relaxed(query)
    query_name = Path(query).name or query
    strict_name = _normalize_strict(query_name)
    relaxed_name = _normalize_relaxed(query_name)
    query_suffix = Path(query_name).suffix.lower()
    query_parts = [part for part in _split_query_parts(query) if part]

    for root in roots:
        try:
            iterator = root.rglob("*")
        except Exception:
            continue
        for candidate in iterator:
            if candidate in seen:
                continue
            seen.add(candidate)
            if not _matches_kind(candidate, kind):
                continue

            candidate_text = str(candidate)
            strict_candidate = _normalize_strict(candidate_text)
            relaxed_candidate = _normalize_relaxed(candidate_text)
            strict_candidate_name = _normalize_strict(candidate.name)
            relaxed_candidate_name = _normalize_relaxed(candidate.name)

            score = 0
            if strict_candidate_name == strict_name:
                score += 240
            elif relaxed_candidate_name == relaxed_name:
                score += 220
            elif relaxed_name and relaxed_name in relaxed_candidate_name:
                score += 170
            elif relaxed_candidate_name and relaxed_candidate_name in relaxed_name:
                score += 150

            if strict_candidate == strict_query:
                score += 260
            elif relaxed_query and relaxed_query == relaxed_candidate:
                score += 240
            elif relaxed_query and relaxed_query in relaxed_candidate:
                score += 120

            if query_suffix and candidate.suffix.lower() == query_suffix:
                score += 25

            for part in query_parts:
                if part and part in relaxed_candidate:
                    score += 15

            if score >= 80:
                results.append(_Match(path=candidate, score=score))

    results.sort(key=lambda item: (-item.score, len(str(item.path)), str(item.path).lower()))
    return results


def _search_roots(query: str, ctx: SandboxContext) -> list[Path]:
    roots: list[Path] = []
    path_like = _looks_like_path(query)
    if path_like:
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
            if not ctx.is_allowed(resolved):
                continue
            roots.append(resolved)
            break

    roots.extend(path.resolve() for path in ctx.allowed_dirs if path.exists())

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
    normalized = unicodedata.normalize("NFKC", text).strip().lower()
    normalized = normalized.replace("\\", "/")
    normalized = re.sub(r"/+", "/", normalized)
    return normalized


def _normalize_relaxed(text: str) -> str:
    normalized = _normalize_strict(text)
    normalized = re.sub(r"\s+", "", normalized)
    return normalized


def _split_query_parts(query: str) -> list[str]:
    strict = _normalize_strict(query)
    parts = [part for part in re.split(r"[\\/]+", strict) if part]
    tokens: list[str] = []
    for part in parts:
        tokens.append(_normalize_relaxed(part))
        tokens.extend(token for token in re.split(r"[\s._-]+", part) if token)
    return [_normalize_relaxed(token) for token in tokens if token]
