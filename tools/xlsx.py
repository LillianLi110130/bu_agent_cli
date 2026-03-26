"""Excel workbook inspection tool for OOXML files."""

from __future__ import annotations

import json
import posixpath
import re
import unicodedata
import zipfile
from pathlib import Path
from typing import Annotated
from xml.etree import ElementTree as ET

from pydantic import BaseModel, Field, ValidationInfo, field_validator

from agent_core.tools import Depends, tool

from tools.path_resolution import AmbiguousPathError, PathNotFoundError, resolve_target_path
from tools.sandbox import SandboxContext, get_sandbox_context

_MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_DOC_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_NS = {"main": _MAIN_NS, "docrel": _DOC_REL_NS, "pkgrel": _PKG_REL_NS}
_SUPPORTED_SUFFIXES = {".xlsx", ".xlsm", ".xltx", ".xltm"}
_CELL_REF_RE = re.compile(r"([A-Z]+)")


class ReadExcelParams(BaseModel):
    file_path: str = Field(description="Excel workbook path, supports fuzzy path resolution.")
    sheet_name: str | None = Field(default=None, description="Optional sheet name to inspect.")
    find_text: str | None = Field(
        default=None,
        description="Optional text to search for within the selected sheet(s).",
    )
    offset_row: int = Field(default=1, description="1-based row number to start previewing from.")
    context_rows: int = Field(default=2, description="Context rows to include around each match.")
    max_matches: int = Field(default=10, description="Maximum number of search matches to return.")
    max_rows: int = Field(default=10, description="Maximum preview rows per sheet.")
    max_cols: int = Field(default=20, description="Maximum preview columns per row.")

    @field_validator("sheet_name", "find_text", mode="before")
    @classmethod
    def _normalize_optional_text(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped or stripped.lower() in {"none", "null"}:
                return None
            return stripped
        return value

    @field_validator(
        "offset_row",
        "context_rows",
        "max_matches",
        "max_rows",
        "max_cols",
        mode="before",
    )
    @classmethod
    def _coerce_integer_like(cls, value: object, info: ValidationInfo) -> int:
        defaults = {
            "offset_row": 1,
            "context_rows": 2,
            "max_matches": 10,
            "max_rows": 10,
            "max_cols": 20,
        }
        default = defaults[info.field_name]

        if value is None or isinstance(value, bool):
            return default
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value) if value.is_integer() else default
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped or stripped.lower() in {"none", "null"}:
                return default
            try:
                return int(stripped)
            except ValueError:
                try:
                    float_value = float(stripped)
                except ValueError:
                    return default
                return int(float_value) if float_value.is_integer() else default
        return default

    @field_validator("offset_row", "max_matches", "max_rows", "max_cols")
    @classmethod
    def _clamp_positive(cls, value: int) -> int:
        return max(1, value)

    @field_validator("context_rows")
    @classmethod
    def _clamp_nonnegative(cls, value: int) -> int:
        return max(0, value)


def _dump_payload(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _normalize_sheet_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip().lower()
    return re.sub(r"\s+", "", normalized)


def _normalize_search_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip().lower()
    return re.sub(r"\s+", " ", normalized)


def _normalize_archive_path(base_dir: str, target: str) -> str:
    return posixpath.normpath(posixpath.join(base_dir, target)).lstrip("./")


def _column_index_from_ref(cell_ref: str) -> int:
    match = _CELL_REF_RE.match(cell_ref.upper())
    if not match:
        return 0

    index = 0
    for char in match.group(1):
        index = index * 26 + (ord(char) - ord("A") + 1)
    return index


def _collect_text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return "".join(node.text or "" for node in element.findall(".//main:t", _NS))


def _load_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    try:
        raw = archive.read("xl/sharedStrings.xml")
    except KeyError:
        return []

    root = ET.fromstring(raw)
    shared_strings: list[str] = []
    for item in root.findall("main:si", _NS):
        shared_strings.append(_collect_text(item))
    return shared_strings


def _load_sheet_targets(archive: zipfile.ZipFile) -> list[tuple[str, str]]:
    workbook_root = ET.fromstring(archive.read("xl/workbook.xml"))
    rels_root = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))

    rel_map: dict[str, str] = {}
    for rel in rels_root.findall("pkgrel:Relationship", _NS):
        rel_id = rel.attrib.get("Id")
        rel_type = rel.attrib.get("Type", "")
        target = rel.attrib.get("Target")
        if not rel_id or not target or not rel_type.endswith("/worksheet"):
            continue
        rel_map[rel_id] = _normalize_archive_path("xl", target)

    sheets: list[tuple[str, str]] = []
    rel_attr = f"{{{_DOC_REL_NS}}}id"
    for sheet in workbook_root.findall("main:sheets/main:sheet", _NS):
        name = sheet.attrib.get("name")
        rel_id = sheet.attrib.get(rel_attr)
        target = rel_map.get(rel_id or "")
        if name and target:
            sheets.append((name, target))
    return sheets


def _read_cell_value(cell: ET.Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    value = cell.findtext("main:v", default="", namespaces=_NS)

    if cell_type == "s":
        if not value:
            return ""
        try:
            return shared_strings[int(value)]
        except (ValueError, IndexError):
            return value

    if cell_type == "inlineStr":
        return _collect_text(cell.find("main:is", _NS))

    if cell_type == "b":
        return "TRUE" if value == "1" else "FALSE"

    if cell_type == "str":
        return value

    if value:
        return value

    formula = cell.findtext("main:f", default="", namespaces=_NS)
    return f"={formula}" if formula else ""


def _materialize_row(values_by_col: dict[int, str], max_cols: int) -> list[str]:
    row_width = min(max(values_by_col.keys(), default=0), max_cols)
    return [values_by_col.get(index, "") for index in range(1, row_width + 1)]


def _parse_sheet(
    archive: zipfile.ZipFile,
    sheet_path: str,
    shared_strings: list[str],
) -> tuple[int, int, list[tuple[int, dict[int, str]]]]:
    root = ET.fromstring(archive.read(sheet_path))
    sheet_data = root.find("main:sheetData", _NS)

    row_count = 0
    column_count = 0
    rows: list[tuple[int, dict[int, str]]] = []

    if sheet_data is None:
        return row_count, column_count, rows

    for row_position, row in enumerate(sheet_data.findall("main:row", _NS), start=1):
        row_number = int(row.attrib.get("r", row_position))
        row_count = max(row_count, row_number)

        values_by_col: dict[int, str] = {}
        for cell_position, cell in enumerate(row.findall("main:c", _NS), start=1):
            ref = cell.attrib.get("r", "")
            col_index = _column_index_from_ref(ref) or cell_position
            values_by_col[col_index] = _read_cell_value(cell, shared_strings)
            column_count = max(column_count, col_index)

        rows.append((row_number, values_by_col))

    return row_count, column_count, rows


def _inspect_sheet(
    sheet_name: str,
    parsed_rows: list[tuple[int, dict[int, str]]],
    *,
    row_count: int,
    column_count: int,
    offset_row: int,
    max_rows: int,
    max_cols: int,
) -> dict:
    preview_rows: list[dict[str, object]] = []
    for row_number, values_by_col in parsed_rows:
        if row_number < offset_row:
            continue
        if len(preview_rows) >= max_rows:
            continue
        preview_rows.append(
            {"row": row_number, "values": _materialize_row(values_by_col, max_cols)}
        )

    return {
        "name": sheet_name,
        "row_count": row_count,
        "column_count": column_count,
        "preview_rows": preview_rows,
    }


def _find_matches(
    *,
    sheet_name: str,
    parsed_rows: list[tuple[int, dict[int, str]]],
    find_text: str,
    offset_row: int,
    max_cols: int,
    context_rows: int,
    remaining_matches: int,
) -> list[dict[str, object]]:
    normalized_query = _normalize_search_text(find_text)
    compact_query = normalized_query.replace(" ", "")
    if not normalized_query:
        return []

    matches: list[dict[str, object]] = []
    filtered_rows = [
        (row_number, values_by_col)
        for row_number, values_by_col in parsed_rows
        if row_number >= offset_row
    ]
    for index, (row_number, values_by_col) in enumerate(filtered_rows):
        matched_columns = [
            column_index
            for column_index, value in sorted(values_by_col.items())
            if normalized_query in _normalize_search_text(value)
            or compact_query in _normalize_search_text(value).replace(" ", "")
        ]
        if not matched_columns:
            continue

        start = max(0, index - context_rows)
        end = min(len(filtered_rows), index + context_rows + 1)
        preview_rows = [
            {"row": ctx_row_number, "values": _materialize_row(ctx_values, max_cols)}
            for ctx_row_number, ctx_values in filtered_rows[start:end]
        ]
        matches.append(
            {
                "sheet": sheet_name,
                "row": row_number,
                "matched_columns": matched_columns,
                "preview_rows": preview_rows,
            }
        )
        if len(matches) >= remaining_matches:
            break

    return matches


def _resolve_sheet_selection(
    requested_name: str | None,
    sheets: list[tuple[str, str]],
) -> tuple[str | None, list[tuple[str, str]] | str]:
    if requested_name is None:
        return None, sheets

    exact = [(name, sheet_path) for name, sheet_path in sheets if name == requested_name]
    if exact:
        return exact[0][0], exact

    normalized_query = _normalize_sheet_name(requested_name)
    normalized_matches = [
        (name, sheet_path)
        for name, sheet_path in sheets
        if _normalize_sheet_name(name) == normalized_query
    ]
    if len(normalized_matches) == 1:
        return normalized_matches[0][0], normalized_matches
    if len(normalized_matches) > 1:
        candidates = ", ".join(name for name, _ in normalized_matches)
        return (
            f"Error: Sheet '{requested_name}' is ambiguous after normalization. "
            f"Candidates: {candidates}"
        )

    available_sheet_names = ", ".join(name for name, _ in sheets)
    return f"Error: Sheet '{requested_name}' not found. Available sheets: {available_sheet_names}"


def _resolve_excel_path(
    file_path: str,
    ctx: SandboxContext,
) -> tuple[str | None, Path | None]:
    try:
        path = resolve_target_path(file_path, ctx, kind="file")
    except Exception as e:
        if isinstance(e, (PathNotFoundError, AmbiguousPathError)):
            return str(e), None
        return f"Security error: {e}", None

    if path.suffix.lower() not in _SUPPORTED_SUFFIXES:
        supported = ", ".join(sorted(_SUPPORTED_SUFFIXES))
        return (
            f"Error: Unsupported Excel format '{path.suffix}'. Supported formats: {supported}",
            None,
        )

    return None, path


def _validate_preview_limits(
    *,
    offset_row: int,
    context_rows: int,
    max_matches: int,
    max_rows: int,
    max_cols: int,
) -> str | None:
    if offset_row < 1 or context_rows < 0 or max_matches < 1 or max_rows < 1 or max_cols < 1:
        return (
            "Error: offset_row, context_rows, max_matches, max_rows, and max_cols "
            "must be valid positive integers (context_rows may be zero)."
        )
    return None


def _normalize_optional_request_text(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None

    stripped = value.strip()
    return stripped or None


def _collect_sheet_payloads(
    archive: zipfile.ZipFile,
    selected_sheets: list[tuple[str, str]],
    shared_strings: list[str],
    *,
    requested_find_text: str | None,
    offset_row: int,
    context_rows: int,
    max_matches: int,
    max_rows: int,
    max_cols: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    remaining_matches = max_matches
    matches: list[dict[str, object]] = []
    sheets_payload: list[dict[str, object]] = []

    for name, sheet_path in selected_sheets:
        row_count, column_count, parsed_rows = _parse_sheet(archive, sheet_path, shared_strings)
        sheets_payload.append(
            _inspect_sheet(
                name,
                parsed_rows,
                row_count=row_count,
                column_count=column_count,
                offset_row=offset_row,
                max_rows=max_rows,
                max_cols=max_cols,
            )
        )

        if requested_find_text is None or remaining_matches <= 0:
            continue

        sheet_matches = _find_matches(
            sheet_name=name,
            parsed_rows=parsed_rows,
            find_text=requested_find_text,
            offset_row=offset_row,
            max_cols=max_cols,
            context_rows=context_rows,
            remaining_matches=remaining_matches,
        )
        matches.extend(sheet_matches)
        remaining_matches -= len(sheet_matches)

    return matches, sheets_payload


def _build_preview_limits(
    *,
    requested_find_text: str | None,
    offset_row: int,
    context_rows: int,
    max_matches: int,
    max_rows: int,
    max_cols: int,
) -> dict[str, object]:
    return {
        "find_text": requested_find_text,
        "offset_row": offset_row,
        "context_rows": context_rows,
        "max_matches": max_matches,
        "max_rows": max_rows,
        "max_cols": max_cols,
    }


def _inspect_workbook(
    archive: zipfile.ZipFile,
    *,
    resolved_path: str,
    requested_sheet_name: str | None,
    requested_find_text: str | None,
    offset_row: int,
    context_rows: int,
    max_matches: int,
    max_rows: int,
    max_cols: int,
) -> tuple[str | None, dict[str, object] | None]:
    shared_strings = _load_shared_strings(archive)
    sheets = _load_sheet_targets(archive)
    if not sheets:
        return "Error: Workbook does not contain readable worksheet definitions.", None

    available_sheet_names = [name for name, _ in sheets]
    selection_result = _resolve_sheet_selection(requested_sheet_name, sheets)
    if isinstance(selection_result, str):
        return selection_result, None

    resolved_sheet_name, selected_sheets = selection_result
    matches, sheets_payload = _collect_sheet_payloads(
        archive,
        selected_sheets,
        shared_strings,
        requested_find_text=requested_find_text,
        offset_row=offset_row,
        context_rows=context_rows,
        max_matches=max_matches,
        max_rows=max_rows,
        max_cols=max_cols,
    )
    payload = {
        "resolved_path": resolved_path,
        "sheet_names": available_sheet_names,
        "selected_sheet": resolved_sheet_name,
        "preview_limits": _build_preview_limits(
            requested_find_text=requested_find_text,
            offset_row=offset_row,
            context_rows=context_rows,
            max_matches=max_matches,
            max_rows=max_rows,
            max_cols=max_cols,
        ),
        "matches": matches,
        "sheets": sheets_payload,
    }
    return None, payload


def _format_workbook_error(path_name: str, error: Exception) -> str:
    if isinstance(error, zipfile.BadZipFile):
        return f"Error: '{path_name}' is not a valid OOXML Excel file."
    if isinstance(error, KeyError):
        return f"Error: Workbook is missing required OOXML part: {error}"
    if isinstance(error, ET.ParseError):
        return f"Error: Failed to parse workbook XML: {error}"
    return f"Error reading Excel file: {error}"


@tool(
    "Read an Excel workbook from a resolved path and return sheet names, preview rows, "
    "and optional text matches.",
    args_schema=ReadExcelParams,
)
async def read_excel(
    file_path: str,
    ctx: Annotated[SandboxContext, Depends(get_sandbox_context)],
    sheet_name: str | None = None,
    find_text: str | None = None,
    offset_row: int = 1,
    context_rows: int = 2,
    max_matches: int = 10,
    max_rows: int = 10,
    max_cols: int = 20,
) -> str:
    """Inspect an OOXML Excel workbook.

    Args:
        file_path: Excel workbook path, supports fuzzy path resolution.
        sheet_name: Optional exact sheet name to inspect. Defaults to all sheets.
        find_text: Optional text to search for within the selected sheet(s).
        offset_row: 1-based row number to start previewing from. Defaults to 1.
        context_rows: Number of surrounding rows to include around each match.
        max_matches: Maximum number of search matches to return.
        max_rows: Maximum preview rows per sheet.
        max_cols: Maximum preview columns per row.
    """
    path_error, path = _resolve_excel_path(file_path, ctx)
    if path_error is not None:
        return path_error

    preview_limit_error = _validate_preview_limits(
        offset_row=offset_row,
        context_rows=context_rows,
        max_matches=max_matches,
        max_rows=max_rows,
        max_cols=max_cols,
    )
    if preview_limit_error is not None:
        return preview_limit_error

    requested_sheet_name = _normalize_optional_request_text(sheet_name)
    requested_find_text = _normalize_optional_request_text(find_text)

    try:
        with zipfile.ZipFile(path) as archive:
            workbook_error, payload = _inspect_workbook(
                archive,
                resolved_path=str(path),
                requested_sheet_name=requested_sheet_name,
                requested_find_text=requested_find_text,
                offset_row=offset_row,
                context_rows=context_rows,
                max_matches=max_matches,
                max_rows=max_rows,
                max_cols=max_cols,
            )
    except Exception as e:
        return _format_workbook_error(path.name, e)

    if workbook_error is not None:
        return workbook_error
    return _dump_payload(payload)
