"""Format LSP responses into compact JSON-friendly payloads."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from agent_core.lsp.types import LSPDiagnostic


def success_payload(tool: str, **fields: Any) -> str:
    payload = {"ok": True, "tool": tool, **fields}
    return json.dumps(payload, ensure_ascii=False, indent=2)


def error_payload(tool: str, error: str) -> str:
    return json.dumps({"ok": False, "tool": tool, "error": error}, ensure_ascii=False, indent=2)


def format_status(status: dict[str, Any]) -> str:
    return success_payload("lsp_status", **status)


def format_diagnostics(diagnostics: list[LSPDiagnostic], *, max_results: int = 100) -> str:
    results = []
    for diagnostic in diagnostics[:max_results]:
        start = diagnostic.range.get("start", {}) if isinstance(diagnostic.range, dict) else {}
        path = uri_to_path(diagnostic.uri)
        results.append(
            {
                "file": str(path) if path is not None else diagnostic.uri,
                "line": _one_based(start.get("line")),
                "character": _one_based(start.get("character")),
                "severity": diagnostic.severity,
                "code": diagnostic.code,
                "source": diagnostic.source,
                "message": diagnostic.message,
            }
        )
    return success_payload(
        "lsp_diagnostics",
        results=results,
        truncated=len(diagnostics) > max_results,
    )


def format_locations(
    tool: str,
    raw: Any,
    *,
    workspace_root: Path,
    max_results: int = 50,
) -> str:
    locations = _extract_locations(raw)
    results = [_format_location(location, workspace_root) for location in locations[:max_results]]
    return success_payload(tool, results=results, truncated=len(locations) > max_results)


def format_hover(raw: Any) -> str:
    contents = raw.get("contents") if isinstance(raw, dict) else raw
    return success_payload("lsp_hover", contents=_stringify_hover(contents))


def format_symbols(tool: str, raw: Any, *, workspace_root: Path, max_results: int = 100) -> str:
    symbols = _flatten_symbols(raw)
    results = []
    for symbol in symbols[:max_results]:
        location = symbol.get("location")
        item: dict[str, Any] = {
            "name": symbol.get("name"),
            "kind": symbol.get("kind"),
            "containerName": symbol.get("containerName"),
        }
        if isinstance(location, dict):
            item.update(_format_location(location, workspace_root))
        elif "selectionRange" in symbol:
            item["range"] = _format_range(symbol.get("selectionRange") or symbol.get("range"))
        results.append(item)
    return success_payload(tool, results=results, truncated=len(symbols) > max_results)


def uri_to_path(uri: str) -> Path | None:
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        return None
    return Path(unquote(parsed.path)).resolve()


def _extract_locations(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        if "targetUri" in raw:
            return [
                {
                    "uri": raw.get("targetUri"),
                    "range": raw.get("targetSelectionRange") or raw.get("targetRange"),
                }
            ]
        if "uri" in raw:
            return [raw]
        return []
    if isinstance(raw, list):
        locations: list[dict[str, Any]] = []
        for item in raw:
            locations.extend(_extract_locations(item))
        return locations
    return []


def _format_location(location: dict[str, Any], workspace_root: Path) -> dict[str, Any]:
    uri = location.get("uri")
    path = uri_to_path(uri) if isinstance(uri, str) else None
    range_payload = _format_range(location.get("range"))
    result: dict[str, Any] = {
        "uri": uri,
        **range_payload,
    }
    if path is not None:
        result["file"] = str(path)
        preview = _read_preview(path, range_payload.get("line"), workspace_root)
        if preview is not None:
            result["preview"] = preview
    return result


def _format_range(raw_range: Any) -> dict[str, Any]:
    if not isinstance(raw_range, dict):
        return {}
    start = raw_range.get("start", {})
    end = raw_range.get("end", {})
    if not isinstance(start, dict):
        start = {}
    if not isinstance(end, dict):
        end = {}
    return {
        "line": _one_based(start.get("line")),
        "character": _one_based(start.get("character")),
        "endLine": _one_based(end.get("line")),
        "endCharacter": _one_based(end.get("character")),
    }


def _one_based(value: Any) -> int | None:
    if not isinstance(value, int):
        return None
    return value + 1


def _read_preview(path: Path, one_based_line: Any, workspace_root: Path) -> str | None:
    if not isinstance(one_based_line, int):
        return None
    if not _is_same_or_parent(workspace_root, path):
        return None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return None
    index = one_based_line - 1
    if index < 0 or index >= len(lines):
        return None
    return lines[index].strip()


def _stringify_hover(contents: Any) -> str:
    if contents is None:
        return ""
    if isinstance(contents, str):
        return contents
    if isinstance(contents, dict):
        value = contents.get("value")
        if isinstance(value, str):
            return value
    if isinstance(contents, list):
        return "\n".join(_stringify_hover(item) for item in contents if item is not None)
    return str(contents)


def _flatten_symbols(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    flattened: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        children = item.get("children")
        shallow = dict(item)
        shallow.pop("children", None)
        flattened.append(shallow)
        if isinstance(children, list):
            flattened.extend(_flatten_symbols(children))
    return flattened


def _is_same_or_parent(parent: Path, child: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False
