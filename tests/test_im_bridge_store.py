from __future__ import annotations

import shutil
import sqlite3
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

import pytest

from cli.im_bridge import SqliteBridgeStore


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


def _create_store(
    workspace_root: Path,
    *,
    worker_no: str = "test-session",
    processing_timeout_seconds: float = 24 * 60 * 60,
) -> SqliteBridgeStore:
    store = SqliteBridgeStore(
        workspace_root,
        session_binding_id=worker_no,
        processing_timeout_seconds=processing_timeout_seconds,
    )
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
    with closing(sqlite3.connect(store.db_path)) as conn:
        status = conn.execute(
            "SELECT status FROM bridge_requests WHERE request_id = ?",
            (request.request_id,),
        ).fetchone()[0]
    assert status == "completed"


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


def test_results_are_stored_in_sqlite_with_utf8_text(workspace_root: Path) -> None:
    store = _create_store(workspace_root)

    first = store.enqueue_text("你好，世界")
    first_claimed = store.claim_next_pending()
    assert first_claimed is not None
    store.complete_request(first_claimed, final_content="第一条结果")

    second = store.enqueue_text("/model")
    second_claimed = store.claim_next_pending()
    assert second_claimed is not None
    store.complete_request(second_claimed, final_content="第二条结果")

    first_result = store.find_result(first.request_id)
    second_result = store.find_result(second.request_id)
    assert first_result is not None
    assert second_result is not None
    assert first_result.input_content == "你好，世界"
    assert first_result.final_content == "第一条结果"
    assert second_result.final_content == "第二条结果"


def test_workers_share_database_but_keep_queues_isolated(workspace_root: Path) -> None:
    first_store = _create_store(workspace_root, worker_no="worker-1")
    second_store = _create_store(workspace_root, worker_no="worker-2")

    first_request = first_store.enqueue_text("first worker")
    second_request = second_store.enqueue_text("second worker")

    assert first_store.db_path == second_store.db_path
    assert first_store.pending_count() == 1
    assert second_store.pending_count() == 1
    assert first_store.claim_next_pending().request_id == first_request.request_id
    assert second_store.claim_next_pending().request_id == second_request.request_id


def test_initialize_removes_only_recognized_legacy_worker_dirs(workspace_root: Path) -> None:
    bridge_dir = workspace_root / ".tg_agent" / "im_bridge"
    legacy_worker_dir = bridge_dir / "worker-old"
    legacy_local_dir = bridge_dir / "local-cli"
    preserved_dir = bridge_dir / "debug-files"
    legacy_worker_dir.mkdir(parents=True)
    legacy_local_dir.mkdir()
    preserved_dir.mkdir()
    (legacy_worker_dir / "request.json").write_text("{}", encoding="utf-8")
    (preserved_dir / "keep.txt").write_text("keep", encoding="utf-8")

    store = _create_store(workspace_root)

    assert not legacy_worker_dir.exists()
    assert not legacy_local_dir.exists()
    assert preserved_dir.exists()
    assert store.db_path.exists()
    assert store.migration_path.exists()


def test_expired_running_request_is_requeued(workspace_root: Path) -> None:
    store = _create_store(workspace_root, processing_timeout_seconds=-1)
    request = store.enqueue_text("retry me")

    first_claim = store.claim_next_pending()
    store.recover_expired_leases()
    second_claim = store.claim_next_pending()

    assert first_claim is not None
    assert second_claim is not None
    assert second_claim.request_id == request.request_id


def test_concurrent_enqueue_assigns_unique_sequences(workspace_root: Path) -> None:
    store = _create_store(workspace_root)

    with ThreadPoolExecutor(max_workers=4) as executor:
        requests = list(executor.map(store.enqueue_text, [f"message-{index}" for index in range(20)]))

    assert sorted(request.seq for request in requests) == list(range(1, 21))
