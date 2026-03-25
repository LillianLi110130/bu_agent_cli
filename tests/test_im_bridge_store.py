from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import pytest

from cli.im_bridge import FileBridgeStore


@pytest.fixture
def workspace_root() -> Path:
    root = Path(".pytest_tmp") / f"im-bridge-store-{uuid.uuid4().hex}"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    try:
        yield root.resolve()
    finally:
        if root.exists():
            shutil.rmtree(root)


def _create_store(workspace_root: Path) -> FileBridgeStore:
    store = FileBridgeStore(workspace_root, session_binding_id="test-session")
    store.initialize()
    return store


def test_enqueue_claim_and_complete_round_trip(workspace_root: Path) -> None:
    store = _create_store(workspace_root)

    request = store.enqueue_text("hello world")
    claimed = store.claim_next_pending()

    assert claimed is not None
    assert claimed.request_id == request.request_id
    assert claimed.seq == 1
    assert claimed.status == "running"
    assert claimed.input_kind == "text"
    assert store.pending_count() == 0

    result = store.complete_request(claimed, final_content="done")

    assert result.final_status == "completed"
    assert result.input_content == "hello world"
    assert result.input_kind == "text"
    assert result.final_content == "done"
    assert store.find_result(request.request_id) is not None
    assert not (store.inbox_processing_dir / "00000000000000000001.json").exists()


def test_claim_returns_lowest_sequence_first(workspace_root: Path) -> None:
    store = _create_store(workspace_root)

    first = store.enqueue_text("first")
    second = store.enqueue_text("/model")

    claimed_first = store.claim_next_pending()
    claimed_second = store.claim_next_pending()

    assert claimed_first is not None
    assert claimed_second is not None
    assert claimed_first.request_id == first.request_id
    assert claimed_second.request_id == second.request_id
    assert claimed_first.seq == 1
    assert claimed_second.seq == 2
    assert claimed_second.input_kind == "slash"


def test_fail_request_writes_failed_result(workspace_root: Path) -> None:
    store = _create_store(workspace_root)

    request = store.enqueue_text("@demo hello")
    claimed = store.claim_next_pending()

    assert claimed is not None

    result = store.fail_request(
        claimed,
        final_content="failed",
        error_code="TEST_ERROR",
        error_message="synthetic failure",
    )

    assert result.final_status == "failed"
    assert result.input_content == "@demo hello"
    assert result.input_kind == "skill"
    assert result.error_code == "TEST_ERROR"
    stored = store.find_result(request.request_id)
    assert stored is not None
    assert stored.final_status == "failed"
    assert stored.error_message == "synthetic failure"


def test_result_file_aggregates_entries_with_utf8_text(workspace_root: Path) -> None:
    store = _create_store(workspace_root)

    first = store.enqueue_text("你好，世界")
    first_claimed = store.claim_next_pending()
    assert first_claimed is not None
    store.complete_request(first_claimed, final_content="第一条结果")

    second = store.enqueue_text("/model")
    second_claimed = store.claim_next_pending()
    assert second_claimed is not None
    store.complete_request(second_claimed, final_content="第二条结果")

    raw = store.results_index_path.read_text(encoding="utf-8")
    assert "你好，世界" in raw
    assert "第一条结果" in raw
    assert "\\u4f60\\u597d" not in raw

    payload = json.loads(raw)
    assert payload["version"] == store.VERSION
    assert len(payload["results"]) == 2
    assert payload["results"][0]["request_id"] == first.request_id
    assert payload["results"][1]["request_id"] == second.request_id

