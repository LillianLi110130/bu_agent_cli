from __future__ import annotations

import json
import shutil
import uuid
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import pytest

from tools.sandbox import SandboxContext
from tools.xlsx import read_excel


def _make_workspace() -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    temp_root = repo_root / ".pytest_tmp"
    temp_root.mkdir(exist_ok=True)
    workspace = temp_root / f"excel-tool-{uuid.uuid4().hex[:8]}"
    workspace.mkdir()
    return workspace


def _column_label(index: int) -> str:
    label = ""
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        label = chr(ord("A") + remainder) + label
    return label


def _sheet_xml(rows: list[list[str]], shared_index: dict[str, int]) -> str:
    row_xml: list[str] = []
    for row_number, row in enumerate(rows, start=1):
        cells: list[str] = []
        for col_number, value in enumerate(row, start=1):
            cell_ref = f"{_column_label(col_number)}{row_number}"
            shared_id = shared_index[value]
            cells.append(f'<c r="{cell_ref}" t="s"><v>{shared_id}</v></c>')
        row_xml.append(f'<row r="{row_number}">{"".join(cells)}</row>')

    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f"<sheetData>{''.join(row_xml)}</sheetData>"
        "</worksheet>"
    )


def _shared_strings_xml(strings: list[str]) -> str:
    items = "".join(f"<si><t>{escape(value)}</t></si>" for value in strings)
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        f'count="{len(strings)}" uniqueCount="{len(strings)}">'
        f"{items}</sst>"
    )


def _write_test_workbook(path: Path, sheets: list[tuple[str, list[list[str]]]]) -> None:
    strings: list[str] = []
    seen: set[str] = set()
    for _, rows in sheets:
        for row in rows:
            for value in row:
                if value in seen:
                    continue
                seen.add(value)
                strings.append(value)

    shared_index = {value: index for index, value in enumerate(strings)}
    workbook_sheets = "".join(
        f'<sheet name="{escape(name)}" sheetId="{index}" r:id="rId{index}"/>'
        for index, (name, _) in enumerate(sheets, start=1)
    )
    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f"<sheets>{workbook_sheets}</sheets>"
        "</workbook>"
    )
    workbook_rels = [
        (
            index,
            "http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet",
            f"worksheets/sheet{index}.xml",
        )
        for index in range(1, len(sheets) + 1)
    ]
    workbook_rels.append(
        (
            len(sheets) + 1,
            "http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings",
            "sharedStrings.xml",
        )
    )
    workbook_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + "".join(
            f'<Relationship Id="rId{rel_id}" Type="{rel_type}" Target="{target}"/>'
            for rel_id, rel_type, target in workbook_rels
        )
        + "</Relationships>"
    )
    content_types_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        + "".join(
            f'<Override PartName="/xl/worksheets/sheet{index}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            for index in range(1, len(sheets) + 1)
        )
        + '<Override PartName="/xl/sharedStrings.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>'
        "</Types>"
    )
    root_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        "</Relationships>"
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types_xml)
        archive.writestr("_rels/.rels", root_rels_xml)
        archive.writestr("xl/workbook.xml", workbook_xml)
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml)
        archive.writestr("xl/sharedStrings.xml", _shared_strings_xml(strings))
        for index, (_, rows) in enumerate(sheets, start=1):
            archive.writestr(f"xl/worksheets/sheet{index}.xml", _sheet_xml(rows, shared_index))


@pytest.mark.anyio
async def test_read_excel_reads_workbook_via_fuzzy_path():
    workspace = _make_workspace()
    workbook_path = workspace / "资料" / "数据表解释-市场维度数据集0324.xlsx"
    _write_test_workbook(
        workbook_path,
        [
            ("市场维度", [["城市", "GMV"], ["上海", "100"], ["北京", "200"]]),
            ("说明", [["字段", "解释"], ["GMV", "成交额"]]),
        ],
    )
    ctx = SandboxContext.create(workspace)

    try:
        result = await read_excel.func("市场维度数据集0324.xlsx", ctx=ctx, max_rows=2)
        payload = json.loads(result)

        assert payload["resolved_path"] == str(workbook_path.resolve())
        assert payload["sheet_names"] == ["市场维度", "说明"]
        assert payload["sheets"][0]["preview_rows"][0]["values"] == ["城市", "GMV"]
        assert payload["sheets"][0]["preview_rows"][1]["values"] == ["上海", "100"]
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


@pytest.mark.anyio
async def test_read_excel_reads_selected_sheet_with_chinese_path():
    workspace = _make_workspace()
    workbook_path = workspace / "中文目录" / "样例.xlsx"
    _write_test_workbook(
        workbook_path,
        [
            ("市场维度", [["渠道", "订单量"], ["App", "18"]]),
            ("备注", [["状态"], ["已核对"]]),
        ],
    )
    ctx = SandboxContext.create(workspace)

    try:
        result = await read_excel.func("中文目录/样例.xlsx", ctx=ctx, sheet_name="备注", max_rows=5)
        payload = json.loads(result)

        assert payload["resolved_path"] == str(workbook_path.resolve())
        assert payload["selected_sheet"] == "备注"
        assert len(payload["sheets"]) == 1
        assert payload["sheets"][0]["name"] == "备注"
        assert payload["sheets"][0]["preview_rows"][1]["values"] == ["已核对"]
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


@pytest.mark.anyio
async def test_read_excel_reports_missing_sheet_name():
    workspace = _make_workspace()
    workbook_path = workspace / "demo.xlsx"
    _write_test_workbook(workbook_path, [("Sheet1", [["hello"]])])
    ctx = SandboxContext.create(workspace)

    try:
        result = await read_excel.func("demo.xlsx", ctx=ctx, sheet_name="不存在")

        assert result == "Error: Sheet '不存在' not found. Available sheets: Sheet1"
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
