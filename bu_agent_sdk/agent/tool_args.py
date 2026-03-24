"""Utilities for parsing tool arguments returned by the LLM."""

from __future__ import annotations

import json
import re
from typing import Any


_JSON_ESCAPE_STARTERS = {'"', "\\", "/", "b", "f", "n", "r", "t", "u"}
_STRING_FIELD_RE = re.compile(
    r'"(?P<key>(?:[^"\\]|\\.)*)"\s*:\s*"(?P<value>(?:[^"\\]|\\.)*)"'
)


class ToolArgumentsError(ValueError):
    """Raised when a tool call payload cannot be coerced into arguments."""


def parse_tool_arguments_for_display(arguments: str) -> dict[str, Any]:
    """Parse tool arguments for UI display and approval flows."""
    try:
        parsed = _load_tool_arguments(arguments)
    except ToolArgumentsError:
        return {"_raw": arguments}

    if isinstance(parsed, dict):
        return parsed
    return {"_value": parsed}


def parse_tool_arguments_for_execution(arguments: str) -> dict[str, Any]:
    """Parse tool arguments for actual tool execution."""
    parsed = _load_tool_arguments(arguments)
    if not isinstance(parsed, dict):
        raise ToolArgumentsError("Tool arguments must decode to a JSON object.")
    return parsed


def _load_tool_arguments(arguments: str) -> Any:
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError as exc:
        repaired = _escape_invalid_backslashes_in_json_strings(arguments)
        if repaired != arguments:
            try:
                parsed = json.loads(repaired)
            except json.JSONDecodeError:
                pass
            else:
                return _normalize_path_like_values(parsed, arguments)
        raise ToolArgumentsError(str(exc)) from exc
    return _normalize_path_like_values(parsed, arguments)


def _escape_invalid_backslashes_in_json_strings(text: str) -> str:
    """Repair lone backslashes inside JSON strings without disturbing valid escapes."""
    chars: list[str] = []
    in_string = False
    i = 0

    while i < len(text):
        char = text[i]

        if not in_string:
            chars.append(char)
            if char == '"':
                in_string = True
            i += 1
            continue

        if char == '"':
            chars.append(char)
            in_string = False
            i += 1
            continue

        if char != "\\":
            chars.append(char)
            i += 1
            continue

        next_index = i + 1
        if next_index >= len(text):
            chars.append("\\\\")
            i += 1
            continue

        next_char = text[next_index]
        if next_char == "u":
            unicode_escape = text[next_index + 1:next_index + 5]
            if len(unicode_escape) == 4 and all(c in "0123456789abcdefABCDEF" for c in unicode_escape):
                chars.append("\\")
                chars.append("u")
                chars.extend(unicode_escape)
                i += 6
                continue

        if next_char in _JSON_ESCAPE_STARTERS - {"u"}:
            chars.append("\\")
            chars.append(next_char)
            i += 2
            continue

        chars.append("\\\\")
        i += 1

    return "".join(chars)


def _normalize_path_like_values(parsed: Any, original_arguments: str) -> Any:
    if not isinstance(parsed, dict):
        return parsed

    raw_string_fields = _extract_top_level_string_fields(original_arguments)
    if not raw_string_fields:
        return parsed

    normalized = dict(parsed)
    for key, value in normalized.items():
        if not isinstance(value, str) or not _looks_like_path_key(key):
            continue
        raw_value = raw_string_fields.get(key)
        if raw_value is None:
            continue
        normalized[key] = _decode_lenient_path_string(raw_value)
    return normalized


def _extract_top_level_string_fields(arguments: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for match in _STRING_FIELD_RE.finditer(arguments):
        key = _decode_json_string(match.group("key"))
        fields[key] = match.group("value")
    return fields


def _decode_json_string(value: str) -> str:
    return json.loads(f'"{value}"')


def _looks_like_path_key(key: str) -> bool:
    lowered = key.lower()
    return (
        lowered == "path"
        or lowered == "cwd"
        or lowered.endswith("_path")
        or lowered.endswith("_dir")
        or lowered.endswith("_root")
        or lowered.endswith("_file")
    )


def _decode_lenient_path_string(raw_value: str) -> str:
    chars: list[str] = []
    i = 0

    while i < len(raw_value):
        char = raw_value[i]
        if char != "\\":
            chars.append(char)
            i += 1
            continue

        next_index = i + 1
        if next_index >= len(raw_value):
            chars.append("\\")
            break

        next_char = raw_value[next_index]
        if next_char in {'"', "\\", "/"}:
            chars.append({'"': '"', "\\": "\\", "/": "/"}[next_char])
            i += 2
            continue

        if next_char == "u":
            unicode_escape = raw_value[next_index + 1:next_index + 5]
            if len(unicode_escape) == 4 and all(c in "0123456789abcdefABCDEF" for c in unicode_escape):
                chars.append(chr(int(unicode_escape, 16)))
                i += 6
                continue

        chars.append("\\")
        chars.append(next_char)
        i += 2

    return "".join(chars)
