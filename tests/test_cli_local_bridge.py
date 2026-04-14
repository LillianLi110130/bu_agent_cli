from __future__ import annotations

import asyncio
import io
import shutil
import sys
import uuid
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import prompt_toolkit
import prompt_toolkit.patch_stdout as prompt_patch_stdout
import pytest

import cli.app as app_module
from agent_core import Agent
from agent_core.agent.events import FinalResponseEvent
from cli.app import TGAgentCLI, _SafeLoadingIndicator
from cli.im_bridge import FileBridgeStore
from cli.slash_commands import SlashCommandRegistry
from cli.worker.runtime_factory import EchoLLM
from tools import SandboxContext


class _DummyPrompter:
    def __init__(self, console):
        self.console = console


@pytest.fixture
def workspace_root() -> Path:
    root = Path(".pytest_tmp") / f"cli-local-bridge-{uuid.uuid4().hex}"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    try:
        yield root.resolve()
    finally:
        if root.exists():
            shutil.rmtree(root)


def _create_cli(workspace_root: Path, monkeypatch):
    monkeypatch.setattr(app_module, "InteractivePrompter", _DummyPrompter)
    agent = Agent(
        llm=EchoLLM(prefix="echo:"),
        tools=[],
        system_prompt="test",
    )
    context = SandboxContext.create(workspace_root)
    store = FileBridgeStore(workspace_root, session_binding_id="local-cli-test")
    cli = TGAgentCLI(
        agent=agent,
        context=context,
        slash_registry=SlashCommandRegistry(),
        bridge_store=store,
    )
    return cli, store


@pytest.mark.asyncio
async def test_local_bridge_drains_enqueued_input_through_execution(workspace_root, monkeypatch):
    cli, store = _create_cli(workspace_root, monkeypatch)
    seen_calls: list[tuple[str, bool]] = []

    async def fake_run_agent(user_input, has_image=False):
        seen_calls.append((user_input, has_image))
        return f"processed:{user_input}"

    monkeypatch.setattr(cli, "_run_agent", fake_run_agent)

    request = cli._enqueue_local_bridge_input("hello bridge")

    should_continue = await cli._drain_bridge_queue()

    assert should_continue is True
    assert seen_calls == [("hello bridge", False)]
    result = store.find_result(request.request_id)
    assert result is not None
    assert result.final_status == "completed"
    assert result.input_content == "hello bridge"
    assert result.input_kind == "text"
    assert result.final_content == "processed:hello bridge"


@pytest.mark.asyncio
async def test_init_command_uses_dedicated_agent_and_injects_immediately(
    workspace_root,
    monkeypatch,
):
    monkeypatch.setenv("HOME", str(workspace_root))
    cli, _store = _create_cli(workspace_root, monkeypatch)
    init_agent = SimpleNamespace(name="init-agent", bind_session_runtime=lambda runtime: None)

    monkeypatch.setattr(app_module, "build_init_agent", lambda **kwargs: init_agent)
    monkeypatch.setattr(
        app_module,
        "build_init_user_prompt",
        lambda workspace: f"init prompt for {workspace}",
    )
    monkeypatch.setattr(app_module, "validate_init_output", lambda workspace: (True, None))

    async def fake_run_agent(user_input, has_image=False, agent=None):
        assert has_image is False
        assert agent is init_agent
        assert str(workspace_root) in user_input
        (workspace_root / "TGAGENTS.md").write_text("repo rules", encoding="utf-8")
        return "init completed"

    monkeypatch.setattr(cli, "_run_agent", fake_run_agent)

    handled = await cli._handle_slash_command("/init")

    assert handled is True
    assert (workspace_root / "TGAGENTS.md").read_text(encoding="utf-8") == "repo rules"
    system_contents = [
        getattr(message, "content", "")
        for message in cli._agent.messages
        if message.role == "system"
    ]
    assert "test" in system_contents
    assert "repo rules" in system_contents


@pytest.mark.asyncio
async def test_run_agent_preinjects_workspace_tgagents_for_main_agent(workspace_root, monkeypatch):
    cli, _store = _create_cli(workspace_root, monkeypatch)
    (workspace_root / "TGAGENTS.md").write_text("repo rules", encoding="utf-8")
    seen_system_contents: list[str] = []

    async def fake_query_stream(user_input, cancel_event=None):
        del cancel_event
        seen_system_contents.extend(
            str(getattr(message, "content", ""))
            for message in cli._agent.messages
            if message.role == "system"
        )
        yield FinalResponseEvent(content=f"processed:{user_input}")

    monkeypatch.setattr(cli._agent, "query_stream", fake_query_stream)

    final_content = await cli._run_agent("hello", has_image=False)

    assert final_content == "processed:hello"
    assert "test" in seen_system_contents
    assert "repo rules" in seen_system_contents


@pytest.mark.asyncio
async def test_local_bridge_exit_request_stops_loop_and_records_result(workspace_root, monkeypatch):
    cli, store = _create_cli(workspace_root, monkeypatch)

    request = cli._enqueue_local_bridge_input("exit")

    should_continue = await cli._drain_bridge_queue()

    assert should_continue is False
    result = store.find_result(request.request_id)
    assert result is not None
    assert result.final_status == "completed"
    assert result.final_content == "再见！"


@pytest.mark.asyncio
async def test_remote_bridge_message_is_processed_while_prompt_is_waiting(
    workspace_root, monkeypatch
):
    cli, store = _create_cli(workspace_root, monkeypatch)
    seen_calls: list[tuple[str, bool]] = []
    prompt_started = asyncio.Event()
    allow_prompt_to_exit = asyncio.Event()

    class _FakePromptSession:
        async def prompt_async(self):
            prompt_started.set()
            await allow_prompt_to_exit.wait()
            raise EOFError()

    monkeypatch.setattr(prompt_toolkit, "PromptSession", lambda **kwargs: _FakePromptSession())
    monkeypatch.setattr(prompt_patch_stdout, "patch_stdout", lambda *args, **kwargs: nullcontext())

    async def fake_run_agent(user_input, has_image=False):
        seen_calls.append((user_input, has_image))
        return f"processed:{user_input}"

    monkeypatch.setattr(cli, "_run_agent", fake_run_agent)

    async def enqueue_remote():
        await prompt_started.wait()
        store.enqueue_text(
            "remote hello",
            source="remote",
            source_meta={"delivery_id": "delivery-1"},
            remote_response_required=True,
            request_id="remote_delivery-1",
        )
        for _ in range(100):
            result = store.find_result("remote_delivery-1")
            if result is not None:
                allow_prompt_to_exit.set()
                return
            await asyncio.sleep(0.01)
        raise AssertionError("remote request was not processed while prompt was waiting")

    enqueue_task = asyncio.create_task(enqueue_remote())
    await cli.run()
    await enqueue_task

    assert seen_calls == [("remote hello", False)]
    result = store.find_result("remote_delivery-1")
    assert result is not None
    assert result.final_status == "completed"
    assert result.input_content == "remote hello"
    assert result.input_kind == "text"
    assert result.final_content == "processed:remote hello"


@pytest.mark.asyncio
async def test_slash_command_final_content_is_captured(workspace_root, monkeypatch):
    cli, store = _create_cli(workspace_root, monkeypatch)

    request = cli._enqueue_local_bridge_input("/pwd")

    should_continue = await cli._drain_bridge_queue()

    assert should_continue is True
    result = store.find_result(request.request_id)
    assert result is not None
    assert result.input_content == "/pwd"
    assert result.input_kind == "slash"
    assert str(workspace_root).replace("\n", "") in result.final_content.replace("\n", "")
    assert "\x1b" not in result.final_content


@pytest.mark.asyncio
async def test_local_exit_is_not_enqueued_in_bridge_mode(workspace_root, monkeypatch):
    cli, store = _create_cli(workspace_root, monkeypatch)
    prompt_calls = 0

    class _FakePromptSession:
        async def prompt_async(self):
            nonlocal prompt_calls
            prompt_calls += 1
            raise EOFError() if prompt_calls > 1 else Exception("__EXIT_NOW__")

    monkeypatch.setattr(prompt_toolkit, "PromptSession", lambda **kwargs: _FakePromptSession())
    monkeypatch.setattr(prompt_patch_stdout, "patch_stdout", lambda *args, **kwargs: nullcontext())

    async def fake_bridge_session(session):
        user_input = "/exit"
        assert cli._is_immediate_local_exit_input(user_input) is True
        cli._console.print("[yellow]再见！[/yellow]")

    monkeypatch.setattr(cli, "_run_with_bridge_session", fake_bridge_session)

    await cli.run()

    assert store.pending_count() == 0
    assert not list(store.results_completed_dir.glob("*.json"))


@pytest.mark.asyncio
async def test_bridge_mode_uses_raw_patch_stdout(workspace_root, monkeypatch):
    cli, _ = _create_cli(workspace_root, monkeypatch)
    seen_kwargs: list[dict[str, object]] = []

    class _FakePromptSession:
        async def prompt_async(self):
            raise EOFError()

    monkeypatch.setattr(prompt_toolkit, "PromptSession", lambda **kwargs: _FakePromptSession())

    def _fake_patch_stdout(*args, **kwargs):
        seen_kwargs.append(dict(kwargs))
        return nullcontext()

    monkeypatch.setattr(prompt_patch_stdout, "patch_stdout", _fake_patch_stdout)

    await cli.run()

    assert seen_kwargs == [{"raw": True}]


def test_safe_loading_and_console_output_do_not_emit_ansi(workspace_root, monkeypatch):
    cli, _ = _create_cli(workspace_root, monkeypatch)
    output = io.StringIO()
    original_stdout = sys.stdout

    try:
        sys.stdout = output
        indicator = _SafeLoadingIndicator("思考中")
        indicator._show_frame(0)
        indicator._clear()
    finally:
        sys.stdout = original_stdout

    rendered = output.getvalue()
    assert "\x1b" not in rendered
    assert "- 思考中..." in rendered

    with cli._console.capture() as capture:
        cli._console.print("[yellow]再见！[/yellow]")

    assert capture.get() == "再见！\n"
