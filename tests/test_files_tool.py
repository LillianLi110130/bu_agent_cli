from __future__ import annotations

import pytest

from tools.files import write
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
async def test_write_appends_to_existing_file(tmp_path):
    ctx = SandboxContext.create(tmp_path)
    target = tmp_path / "note.txt"
    target.write_text("first\n", encoding="utf-8")

    result = await write.func("note.txt", "second\n", ctx=ctx, mode="append")

    assert result == "Appended 7 chars to note.txt"
    assert target.read_text(encoding="utf-8") == "first\nsecond\n"


@pytest.mark.anyio
async def test_write_append_creates_missing_file(tmp_path):
    ctx = SandboxContext.create(tmp_path)
    target = tmp_path / "nested" / "note.txt"

    result = await write.func("nested/note.txt", "created", ctx=ctx, mode="append")

    assert result == "Appended 7 chars to nested/note.txt"
    assert target.read_text(encoding="utf-8") == "created"


@pytest.mark.anyio
async def test_write_append_line_starts_on_new_line_and_ends_with_newline(tmp_path):
    ctx = SandboxContext.create(tmp_path)
    target = tmp_path / "note.txt"
    target.write_text("first", encoding="utf-8")

    result = await write.func("note.txt", "second", ctx=ctx, mode="append_line")

    assert result == "Appended 6 chars as lines to note.txt"
    assert target.read_text(encoding="utf-8") == "first\nsecond\n"


@pytest.mark.anyio
async def test_write_append_line_does_not_duplicate_existing_newline(tmp_path):
    ctx = SandboxContext.create(tmp_path)
    target = tmp_path / "note.txt"
    target.write_text("first\n", encoding="utf-8")

    result = await write.func("note.txt", "second\nthird", ctx=ctx, mode="append_line")

    assert result == "Appended 12 chars as lines to note.txt"
    assert target.read_text(encoding="utf-8") == "first\nsecond\nthird\n"


@pytest.mark.anyio
async def test_write_append_line_creates_missing_file_with_trailing_newline(tmp_path):
    ctx = SandboxContext.create(tmp_path)
    target = tmp_path / "note.txt"

    result = await write.func("note.txt", "created", ctx=ctx, mode="append_line")

    assert result == "Appended 7 chars as lines to note.txt"
    assert target.read_text(encoding="utf-8") == "created\n"


def test_write_tool_schema_exposes_write_mode():
    mode_schema = write.definition.parameters["properties"]["mode"]

    assert mode_schema["enum"] == ["overwrite", "append", "append_line"]
    assert "replaces the whole file" in mode_schema["description"]
    assert "starts on a new line" in mode_schema["description"]


def test_write_tool_schema_describes_chunked_long_writes():
    content_schema = write.definition.parameters["properties"]["content"]

    assert "write in chunks" in write.definition.description
    assert "around 4000 characters" in write.definition.description
    assert "truncated tool arguments" in write.definition.description
    assert "one chunk at a time" in content_schema["description"]
    assert "around 4000 characters" in content_schema["description"]
