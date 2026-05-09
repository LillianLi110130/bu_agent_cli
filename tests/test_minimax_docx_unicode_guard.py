from __future__ import annotations

import subprocess
import sys
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


SKILL_DIR = Path(r"C:\Users\Summer\.codex\skills\minimax-docx")
CHECK_SCRIPT = SKILL_DIR / "scripts" / "check_docx_text_integrity.py"
FALLBACK_SCRIPT = SKILL_DIR / "scripts" / "create_basic_docx.py"
TMP_ROOT = Path(__file__).resolve().parents[1] / "generated_docs" / "test_unicode_guard"


def _write_docx(docx_path: Path, text: str) -> None:
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body>"
        f"<w:p><w:r><w:t>{text}</w:t></w:r></w:p>"
        "</w:body>"
        "</w:document>"
    )
    with ZipFile(docx_path, "w", ZIP_DEFLATED) as zip_file:
        zip_file.writestr("word/document.xml", document_xml)


def _make_temp_dir() -> Path:
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    test_dir = TMP_ROOT / "case_dir"
    test_dir.mkdir(parents=True, exist_ok=True)
    return test_dir


def test_unicode_guard_script_exists() -> None:
    assert CHECK_SCRIPT.exists()


def test_python_fallback_script_exists() -> None:
    assert FALLBACK_SCRIPT.exists()


def test_unicode_guard_passes_for_expected_chinese() -> None:
    docx_path = _make_temp_dir() / "ok.docx"
    _write_docx(docx_path, "2026 Q1 项目进度报告")

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), str(docx_path), "--expect", "项目进度报告"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_unicode_guard_fails_on_question_marks() -> None:
    docx_path = _make_temp_dir() / "bad.docx"
    _write_docx(docx_path, "2026 Q1 ??????")

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), str(docx_path), "--expect", "项目进度报告"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    combined_output = result.stdout + result.stderr
    assert "expected text missing" in combined_output.lower()
    assert "question mark" in combined_output.lower()


def test_python_fallback_preserves_chinese(tmp_path: Path | None = None) -> None:
    work_dir = _make_temp_dir()
    spec_path = work_dir / "spec.json"
    output_path = work_dir / "fallback.docx"
    spec = {
        "title": "2026 Q1 项目进度报告",
        "blocks": [
            {"type": "heading", "text": "项目概述"},
            {"type": "paragraph", "text": "本季度聚焦中文文档稳定生成。"},
        ],
    }
    spec_path.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(FALLBACK_SCRIPT), "--spec", str(spec_path), "--output", str(output_path)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    with ZipFile(output_path) as zip_file:
        document_xml = zip_file.read("word/document.xml").decode("utf-8")
    assert "2026 Q1 项目进度报告" in document_xml
    assert "本季度聚焦中文文档稳定生成。" in document_xml
