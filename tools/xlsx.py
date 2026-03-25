"""Excel workbook inspection tool for OOXML files."""

from __future__ import annotations

import json
import posixpath
import re
import unicodedata
import zipfile
from typing import Annotated
from xml.etree import ElementTree as ET

from bu_agent_sdk.tools import Depends, tool

from tools.path_resolution import AmbiguousPathError, PathNotFoundError, resolve_target_path
from tools.sandbox import SandboxContext, get_sandbox_context

_MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_DOC_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_NS = {"main": _MAIN_NS, "docrel": _DOC_REL_NS, "pkgrel": _PKG_REL_NS}
_SUPPORTED_SUFFIXES = {".xlsx", ".xlsm", ".xltx", ".xltm"}
_CELL_REF_RE = re.compile(r"([A-Z]+)")


def _dump_payload(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _normalize_sheet_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip().lower()
    return re.sub(r"\s+", "", normalized)


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


def _inspect_sheet(
    archive: zipfile.ZipFile,
    sheet_name: str,
    sheet_path: str,
    shared_strings: list[str],
    *,
    max_rows: int,
    max_cols: int,
) -> dict:
    root = ET.fromstring(archive.read(sheet_path))
    sheet_data = root.find("main:sheetData", _NS)

    row_count = 0
    column_count = 0
    preview_rows: list[dict[str, object]] = []

    if sheet_data is None:
        return {
            "name": sheet_name,
            "row_count": row_count,
            "column_count": column_count,
            "preview_rows": preview_rows,
        }

    for row_position, row in enumerate(sheet_data.findall("main:row", _NS), start=1):
        row_number = int(row.attrib.get("r", row_position))
        row_count = max(row_count, row_number)

        values_by_col: dict[int, str] = {}
        for cell_position, cell in enumerate(row.findall("main:c", _NS), start=1):
            ref = cell.attrib.get("r", "")
            col_index = _column_index_from_ref(ref) or cell_position
            values_by_col[col_index] = _read_cell_value(cell, shared_strings)
            column_count = max(column_count, col_index)

        if len(preview_rows) >= max_rows:
            continue

        row_width = min(max(values_by_col.keys(), default=0), max_cols)
        preview_rows.append(
            {
                "row": row_number,
                "values": [values_by_col.get(index, "") for index in range(1, row_width + 1)],
            }
        )

    return {
        "name": sheet_name,
        "row_count": row_count,
        "column_count": column_count,
        "preview_rows": preview_rows,
    }


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


@tool("Read an Excel workbook from a resolved path and return sheet names plus preview rows")
async def read_excel(
    file_path: str,
    ctx: Annotated[SandboxContext, Depends(get_sandbox_context)],
    sheet_name: str | None = None,
    max_rows: int = 10,
    max_cols: int = 20,
) -> str:
    """Inspect an OOXML Excel workbook.

    Args:
        file_path: Excel workbook path, supports fuzzy path resolution.
        sheet_name: Optional exact sheet name to inspect. Defaults to all sheets.
        max_rows: Maximum preview rows per sheet.
        max_cols: Maximum preview columns per row.
    """
    try:
        path = resolve_target_path(file_path, ctx, kind="file")
    except Exception as e:
        if isinstance(e, (PathNotFoundError, AmbiguousPathError)):
            return str(e)
        return f"Security error: {e}"

    if path.suffix.lower() not in _SUPPORTED_SUFFIXES:
        supported = ", ".join(sorted(_SUPPORTED_SUFFIXES))
        return f"Error: Unsupported Excel format '{path.suffix}'. Supported formats: {supported}"

    if max_rows < 1 or max_cols < 1:
        return "Error: max_rows and max_cols must be positive integers."

    requested_sheet_name = sheet_name.strip() if isinstance(sheet_name, str) else None
    if requested_sheet_name == "":
        requested_sheet_name = None

    try:
        with zipfile.ZipFile(path) as archive:
            shared_strings = _load_shared_strings(archive)
            sheets = _load_sheet_targets(archive)
            if not sheets:
                return "Error: Workbook does not contain readable worksheet definitions."

            available_sheet_names = [name for name, _ in sheets]
            selection_result = _resolve_sheet_selection(
                requested_sheet_name,
                sheets,
            )
            if isinstance(selection_result, str):
                return selection_result

            resolved_sheet_name, selected_or_error = selection_result
            selected = selected_or_error

            payload = {
                "resolved_path": str(path),
                "sheet_names": available_sheet_names,
                "selected_sheet": resolved_sheet_name,
                "preview_limits": {"max_rows": max_rows, "max_cols": max_cols},
                "sheets": [
                    _inspect_sheet(
                        archive,
                        name,
                        sheet_path,
                        shared_strings,
                        max_rows=max_rows,
                        max_cols=max_cols,
                    )
                    for name, sheet_path in selected
                ],
            }
            return _dump_payload(payload)
    except zipfile.BadZipFile:
        return f"Error: '{path.name}' is not a valid OOXML Excel file."
    except KeyError as e:
        return f"Error: Workbook is missing required OOXML part: {e}"
    except ET.ParseError as e:
        return f"Error: Failed to parse workbook XML: {e}"
    except Exception as e:
        return f"Error reading Excel file: {e}"
