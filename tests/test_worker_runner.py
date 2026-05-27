from __future__ import annotations

import asyncio
import json
import shutil
import uuid
from pathlib import Path

import httpx
import pytest

from agent_core import Agent
from agent_core.llm.messages import BaseMessage
from agent_core.llm.views import ChatInvokeCompletion, ChatInvokeCompletionChunk
from agent_core.tools import tool
from cli.im_bridge import FileBridgeStore, resolve_session_binding_id
from cli.worker.gateway_client import WorkerGatewayClient, WorkerMessage, WorkerStreamEvent
from cli.worker.mock_server import MockGatewayState, create_mock_gateway_app
from cli.worker.runner import WorkerRunner
from cron.jobs import CronJobStore


class _RecordingCronScheduler:
    def __init__(self) -> None:
        self.contexts = []

    async def tick(self, *, host_context):
        self.contexts.append(host_context)


class _FailingCronScheduler:
    async def tick(self, *, host_context):
        del host_context
        raise RuntimeError("synthetic cron failure")


class _CapturingLLM:
    def __init__(self) -> None:
        self.model = "capture-model"
        self.tool_names: list[str] = []
        self.last_user_content = ""

    async def ainvoke(
        self,
        messages: list[BaseMessage],
        tools=None,
        tool_choice=None,
        **kwargs,
    ) -> ChatInvokeCompletion:
        del tool_choice, kwargs
        self.tool_names = [tool_def.name for tool_def in tools or []]
        for message in reversed(messages):
            if getattr(message, "role", None) == "user":
                self.last_user_content = str(getattr(message, "content", ""))
                break
        return ChatInvokeCompletion(content="cron final")

    async def astream(
        self,
        messages: list[BaseMessage],
        tools=None,
        tool_choice=None,
        **kwargs,
    ):
        del messages, tools, tool_choice, kwargs
        yield ChatInvokeCompletionChunk(delta="stream")


@tool("Cron management", name="cronjob")
async def _fake_cronjob_tool() -> str:
    return "cron"


@tool("Message delivery", name="message")
async def _fake_message_tool() -> str:
    return "message"


@tool("Delegate", name="delegate")
async def _fake_delegate_tool() -> str:
    return "delegate"


@tool("Parallel delegate", name="delegate_parallel")
async def _fake_delegate_parallel_tool() -> str:
    return "delegate_parallel"


@tool("Allowed tool", name="allowed_tool")
async def _fake_allowed_tool() -> str:
    return "allowed"


async def _wait_until(predicate, *, timeout: float = 2.0, interval: float = 0.01) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(interval)
    raise AssertionError("condition not reached before timeout")


@pytest.fixture
def workspace_root() -> Path:
    root = Path(".pytest_tmp") / f"worker-runner-{uuid.uuid4().hex}"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    try:
        yield root.resolve()
    finally:
        if root.exists():
            shutil.rmtree(root)


@pytest.mark.asyncio
async def test_worker_runner_bridges_remote_message_and_completes(workspace_root: Path):
    state = MockGatewayState()
    app = create_mock_gateway_app(state)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as http_client:
        gateway_client = WorkerGatewayClient(base_url="http://testserver", client=http_client)
        runner = WorkerRunner(
            worker_id="worker-1",
            gateway_client=gateway_client,
            model=None,
            root_dir=workspace_root,
            gateway_transport="poll",
            result_poll_interval_seconds=0.01,
        )
        task = asyncio.create_task(runner.run_forever())

        store = FileBridgeStore(
            workspace_root,
            session_binding_id=resolve_session_binding_id("worker-1"),
        )

        await _wait_until(lambda: state.is_online("worker-1"))
        state.enqueue_message(worker_id="worker-1", content="remote hello", source="web")
        await _wait_until(lambda: store.pending_count() == 1)
        claimed = store.claim_next_pending()
        assert claimed is not None
        assert claimed.source == "web"
        assert claimed.content == "remote hello"
        store.complete_request(claimed, final_content="remote done")

        await _wait_until(lambda: len(state.completions) == 1)
        runner.stop()
        await asyncio.wait_for(task, timeout=1.0)

        assert state.is_online("worker-1") is False
        assert state.completions[0]["worker_id"] == "worker-1"
        assert state.completions[0]["final_content"] == "remote done"
        assert state.completions[0]["source"] == "web"


@pytest.mark.asyncio
async def test_worker_runner_forwards_bridge_progress_before_completion(workspace_root: Path):
    state = MockGatewayState()
    app = create_mock_gateway_app(state)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as http_client:
        gateway_client = WorkerGatewayClient(base_url="http://testserver", client=http_client)
        runner = WorkerRunner(
            worker_id="worker-1",
            gateway_client=gateway_client,
            model=None,
            root_dir=workspace_root,
            gateway_transport="poll",
            result_poll_interval_seconds=0.01,
        )
        task = asyncio.create_task(runner.run_forever())

        store = FileBridgeStore(
            workspace_root,
            session_binding_id=resolve_session_binding_id("worker-1"),
        )

        await _wait_until(lambda: state.is_online("worker-1"))
        state.enqueue_message(worker_id="worker-1", content="remote hello", source="web")
        await _wait_until(lambda: store.pending_count() == 1)
        claimed = store.claim_next_pending()
        assert claimed is not None

        store.record_progress(claimed, "partial remote")
        await _wait_until(lambda: len(state.progress_updates) == 1)
        assert state.progress_updates[0]["content"] == "partial remote"
        assert state.progress_updates[0]["source"] == "web"
        assert state.completions == []

        store.complete_request(claimed, final_content="remote done")
        await _wait_until(lambda: len(state.completions) == 1)
        runner.stop()
        await asyncio.wait_for(task, timeout=1.0)

        assert state.completions[0]["final_content"] == "remote done"
        assert state.completions[0]["source"] == "web"


@pytest.mark.asyncio
async def test_worker_runner_bridges_remote_message_and_completes_via_sse(workspace_root: Path):
    class FakeSSEGatewayClient:
        def __init__(self) -> None:
            self.queue: asyncio.Queue[WorkerStreamEvent] = asyncio.Queue()
            self.online_calls: list[str] = []
            self.offline_calls: list[str] = []
            self.completions: list[dict[str, str]] = []

        async def online(self, worker_id: str) -> bool:
            self.online_calls.append(worker_id)
            return True

        async def offline(self, worker_id: str) -> bool:
            self.offline_calls.append(worker_id)
            return True

        async def complete(
            self,
            worker_id: str,
            final_content: str,
            source: str | None = None,
        ) -> bool:
            self.completions.append(
                {"worker_id": worker_id, "final_content": final_content, "source": source}
            )
            return True

        async def stream_events(self, worker_id: str):
            yield WorkerStreamEvent(event="ready", payload={"worker_id": worker_id})
            while True:
                event = await self.queue.get()
                yield event

    gateway_client = FakeSSEGatewayClient()
    runner = WorkerRunner(
        worker_id="worker-1",
        gateway_client=gateway_client,
        model=None,
        root_dir=workspace_root,
        gateway_transport="sse",
        result_poll_interval_seconds=0.01,
    )
    task = asyncio.create_task(runner.run_forever())

    store = FileBridgeStore(
        workspace_root,
        session_binding_id=resolve_session_binding_id("worker-1"),
    )

    await _wait_until(lambda: gateway_client.online_calls == ["worker-1"])
    await gateway_client.queue.put(
        WorkerStreamEvent(
            event="message",
            message=WorkerMessage(content="remote hello sse", source="web"),
            payload={"content": "remote hello sse", "source": "web"},
        )
    )
    await _wait_until(lambda: store.pending_count() == 1)
    claimed = store.claim_next_pending()
    assert claimed is not None
    assert claimed.source == "web"
    assert claimed.content == "remote hello sse"
    store.complete_request(claimed, final_content="remote done sse")

    await _wait_until(lambda: len(gateway_client.completions) == 1)
    runner.stop()
    await gateway_client.queue.put(WorkerStreamEvent(event="heartbeat", payload={"ts": 1}))
    await asyncio.wait_for(task, timeout=1.0)

    assert gateway_client.offline_calls == ["worker-1"]
    assert gateway_client.completions == [
        {"worker_id": "worker-1", "final_content": "remote done sse", "source": "web"}
    ]


@pytest.mark.asyncio
async def test_worker_runner_keeps_streaming_while_previous_remote_request_is_processing(
    workspace_root: Path,
):
    class FakeSSEGatewayClient:
        def __init__(self) -> None:
            self.queue: asyncio.Queue[WorkerStreamEvent] = asyncio.Queue()
            self.online_calls: list[str] = []
            self.offline_calls: list[str] = []
            self.completions: list[dict[str, str]] = []

        async def online(self, worker_id: str) -> bool:
            self.online_calls.append(worker_id)
            return True

        async def offline(self, worker_id: str) -> bool:
            self.offline_calls.append(worker_id)
            return True

        async def complete(
            self,
            worker_id: str,
            final_content: str,
            source: str | None = None,
        ) -> bool:
            self.completions.append(
                {"worker_id": worker_id, "final_content": final_content, "source": source}
            )
            return True

        async def stream_events(self, worker_id: str):
            yield WorkerStreamEvent(event="ready", payload={"worker_id": worker_id})
            while True:
                event = await self.queue.get()
                yield event

    gateway_client = FakeSSEGatewayClient()
    runner = WorkerRunner(
        worker_id="worker-1",
        gateway_client=gateway_client,
        model=None,
        root_dir=workspace_root,
        gateway_transport="sse",
        result_poll_interval_seconds=0.01,
    )
    task = asyncio.create_task(runner.run_forever())

    store = FileBridgeStore(
        workspace_root,
        session_binding_id=resolve_session_binding_id("worker-1"),
    )

    await _wait_until(lambda: gateway_client.online_calls == ["worker-1"])
    await gateway_client.queue.put(
        WorkerStreamEvent(
            event="message",
            message=WorkerMessage(content="remote first sse", source="web"),
            payload={"content": "remote first sse", "source": "web"},
        )
    )
    await _wait_until(lambda: store.pending_count() == 1)

    first = store.claim_next_pending()
    assert first is not None
    assert first.content == "remote first sse"

    await gateway_client.queue.put(
        WorkerStreamEvent(
            event="message",
            message=WorkerMessage(content="remote second sse", source="web"),
            payload={"content": "remote second sse", "source": "web"},
        )
    )
    await _wait_until(lambda: store.pending_count() == 1)

    second = store.claim_next_pending()
    assert second is not None
    assert second.content == "remote second sse"

    store.complete_request(first, final_content="done first sse")
    await _wait_until(lambda: len(gateway_client.completions) == 1)
    assert gateway_client.completions[0] == {
        "worker_id": "worker-1",
        "final_content": "done first sse",
        "source": "web",
    }

    store.complete_request(second, final_content="done second sse")
    await _wait_until(lambda: len(gateway_client.completions) == 2)
    runner.stop()
    await gateway_client.queue.put(WorkerStreamEvent(event="heartbeat", payload={"ts": 1}))
    await asyncio.wait_for(task, timeout=1.0)

    assert gateway_client.offline_calls == ["worker-1"]
    assert gateway_client.completions[1] == {
        "worker_id": "worker-1",
        "final_content": "done second sse",
        "source": "web",
    }


@pytest.mark.asyncio
async def test_worker_runner_uses_local_request_id_for_remote_message(workspace_root: Path):
    state = MockGatewayState()
    app = create_mock_gateway_app(state)
    transport = httpx.ASGITransport(app=app)

    store = FileBridgeStore(
        workspace_root,
        session_binding_id=resolve_session_binding_id("worker-1"),
    )

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as http_client:
        gateway_client = WorkerGatewayClient(base_url="http://testserver", client=http_client)
        runner = WorkerRunner(
            worker_id="worker-1",
            gateway_client=gateway_client,
            model=None,
            root_dir=workspace_root,
            gateway_transport="poll",
            result_poll_interval_seconds=0.01,
        )
        task = asyncio.create_task(runner.run_forever())

        await _wait_until(lambda: state.is_online("worker-1"))
        state.enqueue_message(worker_id="worker-1", content="remote cached", source="web")

        await _wait_until(lambda: store.pending_count() == 1)
        pending_path = next(store.inbox_pending_dir.glob("*.json"))
        payload = json.loads(pending_path.read_text(encoding="utf-8"))
        request_id = str(payload["request_id"])
        assert payload["source"] == "web"
        claimed = store.claim_next_pending()
        assert claimed is not None
        store.complete_request(claimed, final_content="cached done")

        await _wait_until(lambda: len(state.completions) == 1)
        runner.stop()
        await asyncio.wait_for(task, timeout=1.0)

        assert request_id.startswith("req_")
        assert state.completions[0]["worker_id"] == "worker-1"
        assert state.completions[0]["final_content"] == "cached done"
        assert state.completions[0]["source"] == "web"
        assert store.pending_count() == 0


@pytest.mark.asyncio
async def test_worker_runner_keeps_polling_while_previous_remote_request_is_processing(
    workspace_root: Path,
):
    state = MockGatewayState()
    app = create_mock_gateway_app(state)
    transport = httpx.ASGITransport(app=app)

    store = FileBridgeStore(
        workspace_root,
        session_binding_id=resolve_session_binding_id("worker-1"),
    )

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as http_client:
        gateway_client = WorkerGatewayClient(base_url="http://testserver", client=http_client)
        runner = WorkerRunner(
            worker_id="worker-1",
            gateway_client=gateway_client,
            model=None,
            root_dir=workspace_root,
            gateway_transport="poll",
            result_poll_interval_seconds=0.01,
        )
        task = asyncio.create_task(runner.run_forever())

        await _wait_until(lambda: state.is_online("worker-1"))
        state.enqueue_message(worker_id="worker-1", content="remote first", source="web")
        await _wait_until(lambda: store.pending_count() == 1)

        first = store.claim_next_pending()
        assert first is not None
        assert first.content == "remote first"

        state.enqueue_message(worker_id="worker-1", content="remote second", source="web")
        await _wait_until(lambda: store.pending_count() == 1)

        second = store.claim_next_pending()
        assert second is not None
        assert second.content == "remote second"

        store.complete_request(first, final_content="done first")
        await _wait_until(lambda: len(state.completions) == 1)
        assert state.completions[0]["final_content"] == "done first"
        assert state.completions[0]["source"] == "web"

        store.complete_request(second, final_content="done second")
        await _wait_until(lambda: len(state.completions) == 2)
        runner.stop()
        await asyncio.wait_for(task, timeout=1.0)

        assert state.completions[1]["final_content"] == "done second"
        assert state.completions[1]["source"] == "web"


@pytest.mark.asyncio
async def test_mock_gateway_rejects_messages_when_no_worker_is_online():
    state = MockGatewayState()
    app = create_mock_gateway_app(state)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as http_client:
        response = await http_client.post(
            "/mock/messages",
            json={
                "worker_id": "worker-1",
                "content": "hello",
                "source": "web",
            },
        )

    assert response.status_code == 409
    payload = response.json()
    assert payload["ok"] is False
    assert payload["error"] == "no_online_worker"


@pytest.mark.asyncio
async def test_mock_gateway_assigns_message_to_worker_queue():
    state = MockGatewayState()
    state.mark_online(worker_id="worker-1")
    app = create_mock_gateway_app(state)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as http_client:
        response = await http_client.post(
            "/mock/messages",
            json={
                "worker_id": "worker-1",
                "content": "hello",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert state.queued_messages[0].worker_id == "worker-1"
    assert state.queued_messages[0].source == "im"


@pytest.mark.asyncio
async def test_mock_gateway_routes_messages_by_worker_no():
    state = MockGatewayState()
    state.mark_online(worker_id="worker-terminal-a")
    state.mark_online(worker_id="worker-terminal-b")
    app = create_mock_gateway_app(state)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as http_client:
        response = await http_client.post(
            "/mock/messages",
            json={
                "worker_id": "user-1",
                "worker_no": "worker-terminal-b",
                "content": "hello terminal b",
            },
        )
        assert response.status_code == 200

        poll_a = await http_client.post(
            "/api/worker/poll",
            json={"worker_id": "user-1", "worker_no": "worker-terminal-a"},
        )
        poll_b = await http_client.post(
            "/api/worker/poll",
            json={"worker_id": "user-1", "worker_no": "worker-terminal-b"},
        )

    assert poll_a.json() == {"messages": []}
    assert poll_b.json()["messages"][0]["content"] == "hello terminal b"
    assert poll_b.json()["messages"][0]["worker_no"] == "worker-terminal-b"


@pytest.mark.asyncio
async def test_mock_gateway_new_and_llm_stream_return_session_event():
    state = MockGatewayState()
    app = create_mock_gateway_app(state)

    class FakeLLMGateway:
        async def query_stream(self, request):  # noqa: ANN001
            assert request.session_id.startswith("session-")
            yield type("TextEvent", (), {"model_dump": lambda self: {"type": "text", "content": "hello"}})()
            yield type("DoneEvent", (), {"model_dump": lambda self: {"type": "done", "stop_reason": "stop"}})()

    app.state.llm_gateway = FakeLLMGateway()
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as http_client:
        new_response = await http_client.post(
            "/new",
            json={"worker_id": "user-1", "worker_no": "worker-terminal-1"},
        )
        stream_response = await http_client.post(
            "/llm/query-stream",
            json={
                "model": "coding-default",
                "worker_no": "worker-terminal-1",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    new_payload = new_response.json()
    assert new_payload["session_id"].startswith("session-")
    assert '"type": "session_created"' in stream_response.text
    assert '"session_no": "session-' in stream_response.text
    assert '"is_new": true' in stream_response.text
    assert '"type": "text"' in stream_response.text
    assert '"content": "hello"' in stream_response.text
    assert '"type": "done"' in stream_response.text


def test_mock_gateway_register_stream_replaces_previous_connection():
    state = MockGatewayState()
    first_version = state.register_stream(worker_id="worker-1")
    second_version = state.register_stream(worker_id="worker-1")

    assert state.is_current_stream(worker_id="worker-1", version=first_version) is False
    assert state.is_current_stream(worker_id="worker-1", version=second_version) is True


@pytest.mark.asyncio
async def test_worker_runner_builds_remote_cron_host_context(workspace_root: Path):
    class FakeGatewayClient:
        pass

    scheduler = _RecordingCronScheduler()
    runner = WorkerRunner(
        worker_id="worker-1",
        gateway_client=FakeGatewayClient(),
        model=None,
        root_dir=workspace_root,
        cron_tick_interval_seconds=0.01,
    )
    runner.cron_scheduler = scheduler
    runner._cron_next_tick_at = 0

    await runner._maybe_tick_cron()

    assert len(scheduler.contexts) == 1
    context = scheduler.contexts[0]
    assert context.source == "remote"
    assert context.workspace_root == workspace_root
    assert context.session_binding_id == resolve_session_binding_id("worker-1")
    assert context.worker_id == "worker-1"
    assert context.gateway_client is runner.gateway_client
    assert context.default_delivery == "remote"


@pytest.mark.asyncio
async def test_worker_runner_cron_tick_error_does_not_raise(workspace_root: Path):
    class FakeGatewayClient:
        pass

    runner = WorkerRunner(
        worker_id="worker-1",
        gateway_client=FakeGatewayClient(),
        model=None,
        root_dir=workspace_root,
        cron_tick_interval_seconds=0.01,
    )
    runner.cron_scheduler = _FailingCronScheduler()
    runner._cron_next_tick_at = 0

    await runner._maybe_tick_cron()


@pytest.mark.asyncio
async def test_worker_runner_fresh_cron_agent_reuses_main_agent_llm_and_query(
    workspace_root: Path,
):
    class FakeGatewayClient:
        pass

    llm = _CapturingLLM()
    main_agent = Agent(
        llm=llm,
        tools=[
            _fake_cronjob_tool,
            _fake_message_tool,
            _fake_delegate_tool,
            _fake_delegate_parallel_tool,
            _fake_allowed_tool,
        ],
        system_prompt="main system",
        max_iterations=37,
        hooks=[],
        use_streaming=False,
    )
    store = CronJobStore(base_dir=workspace_root / "cron")
    job = store.create_job(
        name="Fresh",
        prompt="do scheduled work",
        schedule_text="every 30m",
        workspace_root=workspace_root,
        session_binding_id="worker-1",
        execution_mode="fresh_agent_background",
    )
    runner = WorkerRunner(
        worker_id="worker-1",
        gateway_client=FakeGatewayClient(),
        model=None,
        root_dir=workspace_root,
        agent=main_agent,
    )

    cron_agent = runner._build_cron_background_agent(main_agent, job)
    result = await runner._run_cron_fresh_agent(job)

    assert result == "cron final"
    assert cron_agent.llm is main_agent.llm
    assert cron_agent.system_prompt == "main system"
    assert cron_agent.max_iterations == 37
    assert cron_agent.runtime_role == "primary"
    assert cron_agent.hooks == []
    assert "message" in llm.tool_names
    assert "allowed_tool" in llm.tool_names
    assert "cronjob" not in llm.tool_names
    assert "delegate" not in llm.tool_names
    assert "delegate_parallel" not in llm.tool_names
    assert "do scheduled work" in llm.last_user_content
