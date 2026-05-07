from __future__ import annotations

import pytest

from tools import ALL_TOOLS
from tools.files import edit, read, write
from tools.sandbox import SandboxContext


@pytest.mark.anyio
async def test_write_overwrites_by_default(tmp_path):
    ctx = SandboxContext.create(tmp_path)
    target = tmp_path / "note.txt"
    target.write_text("old", encoding="utf-8")

    result = await write.func("note.txt", "new", ctx=ctx)

    assert result == "Wrote 3 chars to note.txt"
    assert target.read_text(encoding="utf-8") == "new"


@pytest.mark.anyio
async def test_write_replaces_entire_existing_file(tmp_path):
    ctx = SandboxContext.create(tmp_path)
    target = tmp_path / "note.txt"
    target.write_text("first\n", encoding="utf-8")

    result = await write.func("note.txt", "second\n", ctx=ctx)

    assert result == "Wrote 7 chars to note.txt"
    assert target.read_text(encoding="utf-8") == "second\n"


@pytest.mark.anyio
async def test_write_creates_missing_file(tmp_path):
    ctx = SandboxContext.create(tmp_path)
    target = tmp_path / "nested" / "note.txt"

    result = await write.func("nested/note.txt", "created", ctx=ctx)

    assert result == "Wrote 7 chars to nested/note.txt"
    assert target.read_text(encoding="utf-8") == "created"


@pytest.mark.anyio
async def test_read_uses_pipe_separated_line_numbers(tmp_path):
    ctx = SandboxContext.create(tmp_path)
    target = tmp_path / "code.py"
    target.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")

    result = await read.func("code.py", ctx=ctx)

    assert "[Lines 1-2 of 2]" in result
    assert "     1|def add(a, b):" in result
    assert "     2|    return a + b" in result


def test_write_tool_schema_only_exposes_content_and_file_path():
    properties = write.definition.parameters["properties"]

    assert list(properties) == ["content", "file_path"]
    assert write.definition.parameters["required"] == ["content", "file_path"]
    assert write.definition.parameters["additionalProperties"] is False


def test_write_tool_schema_describes_full_replacement():
    content_schema = write.definition.parameters["properties"]["content"]

    assert "Prefer edit" in write.definition.description
    assert "localized changes" in write.definition.description
    assert "replacing the entire file" in write.definition.description
    assert "Content to write" in content_schema["description"]


def test_edit_tool_schema_guides_localized_code_changes():
    assert "existing file" in edit.definition.description
    assert "exact text span" in edit.definition.description
    assert "localized changes" in edit.definition.description
    assert "code files" in edit.definition.description


def test_edit_tool_schema_warns_old_string_must_exclude_read_line_numbers():
    old_string_schema = edit.definition.parameters["properties"]["old_string"]

    assert "exact file text" in old_string_schema["description"]
    assert "line numbers" in old_string_schema["description"]
    assert '"[Lines ...]" header' in old_string_schema["description"]


def test_file_tool_order_prefers_edit_before_write():
    tool_names = [tool.name for tool in ALL_TOOLS]

    assert tool_names.index("edit") < tool_names.index("write")
