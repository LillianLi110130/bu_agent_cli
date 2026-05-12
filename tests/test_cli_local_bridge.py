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
from agent_core.llm.messages import AssistantMessage, UserMessage
from agent_core.llm.views import ChatInvokeCompletion
from agent_core.team import TeamRuntime
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


def test_remote_reset_startup_prompt_is_loaded_from_prompt_file(workspace_root, monkeypatch):
    cli, _store = _create_cli(workspace_root, monkeypatch)
    prompt_path = workspace_root / "remote_reset_startup.md"
    prompt_path.write_text("自定义启动提示", encoding="utf-8")
    monkeypatch.setattr(app_module, "_REMOTE_RESET_STARTUP_PROMPT_PATH", prompt_path)

    assert cli._build_remote_reset_startup_prompt() == "自定义启动提示\n\n当前运行模型：echo-worker"


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


def test_team_inbox_watcher_enqueues_clarification_request_via_bridge(
    workspace_root,
    monkeypatch,
):
    async def run_case() -> None:
        cli, store = _create_cli(workspace_root, monkeypatch)
        cli.TEAM_INBOX_POLL_INTERVAL_SECONDS = 0.01
        runtime = TeamRuntime(teams_root=workspace_root / "teams", workspace_root=workspace_root)
        team = runtime.start_team(goal="Coordinate")
        cli._ctx.team_runtime = runtime
        runtime.send_message(
            team_id=team.team_id,
            sender="backend-1",
            recipient="lead",
            type="clarification_request",
            body="Which database should I use?",
            metadata={
                "question": "Which database should I use?",
                "options": ["sqlite", "postgres"],
                "recommended": "sqlite",
                "blocking": True,
            },
        )

        task = asyncio.create_task(cli._consume_team_inbox_auto_triggers())
        try:
            for _ in range(50):
                if store.pending_count() > 0:
                    break
                await asyncio.sleep(0.02)
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        assert store.pending_count() == 1
        request = store.claim_next_pending()
        assert request is not None
        assert request.source == "local"
        assert request.source_meta["kind"] == "team_inbox_auto_trigger"
        assert request.source_meta["team_id"] == team.team_id
        assert "Team Inbox Auto Trigger" in request.content
        assert "Language rules:" in request.content
        assert "prefer Chinese" in request.content
        assert "clarification_request" in request.content
        assert "Which database should I use?" in request.content
        assert runtime.read_lead_inbox(team.team_id, ack=False) == []

    asyncio.run(run_case())


def test_team_inbox_watcher_enqueues_lifecycle_triggers_via_bridge(
    workspace_root,
    monkeypatch,
):
    async def run_case() -> None:
        cli, store = _create_cli(workspace_root, monkeypatch)
        cli.TEAM_INBOX_POLL_INTERVAL_SECONDS = 0.01
        runtime = TeamRuntime(teams_root=workspace_root / "teams", workspace_root=workspace_root)
        team = runtime.start_team(goal="Coordinate")
        cli._ctx.team_runtime = runtime
        for message_type in (
            "task_done_notification",
            "task_blocked_notification",
            "worker_failed",
            "idle_notification",
        ):
            runtime.send_message(
                team_id=team.team_id,
                sender="backend-1",
                recipient="lead",
                type=message_type,
                body=f"{message_type} body",
            )

        task = asyncio.create_task(cli._consume_team_inbox_auto_triggers())
        try:
            for _ in range(50):
                if store.pending_count() > 0:
                    break
                await asyncio.sleep(0.02)
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        assert store.pending_count() == 1
        request = store.claim_next_pending()
        assert request is not None
        assert request.source_meta["kind"] == "team_inbox_auto_trigger"
        assert request.source_meta["team_id"] == team.team_id
        assert set(request.source_meta["message_ids"])
        assert "task_done_notification" in request.content
        assert "task_blocked_notification" in request.content
        assert "worker_failed" in request.content
        assert "idle_notification" in request.content
        assert runtime.read_lead_inbox(team.team_id, ack=False) == []

    asyncio.run(run_case())


def test_team_inbox_watcher_ignores_non_trigger_messages(workspace_root, monkeypatch):
    async def run_case() -> None:
        cli, store = _create_cli(workspace_root, monkeypatch)
        cli.TEAM_INBOX_POLL_INTERVAL_SECONDS = 0.01
        runtime = TeamRuntime(teams_root=workspace_root / "teams", workspace_root=workspace_root)
        team = runtime.start_team(goal="Coordinate")
        cli._ctx.team_runtime = runtime
        runtime.send_message(
            team_id=team.team_id,
            sender="backend-1",
            recipient="lead",
            type="message",
            body="Done",
        )

        task = asyncio.create_task(cli._consume_team_inbox_auto_triggers())
        try:
            await asyncio.sleep(0.05)
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        assert store.pending_count() == 0
        unread = runtime.read_lead_inbox(team.team_id, ack=False)
        assert len(unread) == 1
        assert unread[0].type == "message"

    asyncio.run(run_case())

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
async def test_remote_bridge_image_json_is_processed_as_multimodal_input(
    workspace_root,
    monkeypatch,
):
    cli, store = _create_cli(workspace_root, monkeypatch)
    seen_calls = []

    async def fake_run_agent(user_input, has_image=False):
        seen_calls.append((user_input, has_image))
        return "processed:remote image"

    monkeypatch.setattr(cli, "_run_agent", fake_run_agent)

    payload = (
        '{"msgType":"image","imageDataBase64":["iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwC'
        'AAAAC0lEQVR42mP8/x8AAwMCAO+a4e0AAAAASUVORK5CYII="],"userInput":"帮我看下这张图"}'
    )
    store.enqueue_text(
        payload,
        source="remote",
        source_meta={"delivery_id": "delivery-image-1"},
        remote_response_required=True,
        request_id="remote-image-1",
    )

    should_continue = await cli._drain_bridge_queue()

    assert should_continue is True
    assert len(seen_calls) == 1
    user_input, has_image = seen_calls[0]
    assert has_image is True
    assert isinstance(user_input, list)
    assert user_input[0].type == "text"
    assert user_input[0].text == "帮我看下这张图"
    assert user_input[1].type == "image_url"
    assert user_input[1].image_url.media_type == "image/png"
    assert user_input[1].image_url.url.startswith("data:image/png;base64,")

    result = store.find_result("remote-image-1")
    assert result is not None
    assert result.final_status == "completed"
    assert result.input_kind == "image"
    assert result.final_content == "processed:remote image"


@pytest.mark.asyncio
async def test_remote_bridge_image_json_supports_multiple_images(
    workspace_root,
    monkeypatch,
):
    cli, store = _create_cli(workspace_root, monkeypatch)
    seen_calls = []

    async def fake_run_agent(user_input, has_image=False):
        seen_calls.append((user_input, has_image))
        return "processed:remote images"

    monkeypatch.setattr(cli, "_run_agent", fake_run_agent)

    png_base64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwC"
        "AAAAC0lEQVR42mP8/x8AAwMCAO+a4e0AAAAASUVORK5CYII="
    )
    payload = (
        '{"msgType":"image","imageDataBase64":["'
        + png_base64
        + '","data:image/png;base64,'
        + png_base64
        + '"],"userInput":"对比这两张图"}'
    )
    store.enqueue_text(
        payload,
        source="remote",
        source_meta={"delivery_id": "delivery-images-1"},
        remote_response_required=True,
        request_id="remote-images-1",
    )

    should_continue = await cli._drain_bridge_queue()

    assert should_continue is True
    assert len(seen_calls) == 1
    user_input, has_image = seen_calls[0]
    assert has_image is True
    assert isinstance(user_input, list)
    assert [part.type for part in user_input] == ["text", "image_url", "image_url"]
    assert user_input[0].text == "对比这两张图"
    assert user_input[1].image_url.media_type == "image/png"
    assert user_input[2].image_url.media_type == "image/png"

    result = store.find_result("remote-images-1")
    assert result is not None
    assert result.final_status == "completed"
    assert result.input_kind == "image"
    assert result.final_content == "processed:remote images"


@pytest.mark.asyncio
async def test_remote_bridge_invalid_image_json_fails_request(workspace_root, monkeypatch):
    cli, store = _create_cli(workspace_root, monkeypatch)

    async def fail_run_agent(user_input, has_image=False):
        del user_input, has_image
        raise AssertionError("invalid remote image payload should not reach _run_agent")

    monkeypatch.setattr(cli, "_run_agent", fail_run_agent)

    store.enqueue_text(
        '{"msgType":"image","imageDataBase64":["not-base64"],"userInput":"帮我看下"}',
        source="remote",
        source_meta={"delivery_id": "delivery-image-invalid-1"},
        remote_response_required=True,
        request_id="remote-image-invalid-1",
    )

    should_continue = await cli._drain_bridge_queue()

    assert should_continue is True
    result = store.find_result("remote-image-invalid-1")
    assert result is not None
    assert result.final_status == "failed"
    assert result.error_code == "REMOTE_IMAGE_MESSAGE_INVALID"
    assert "base64" in (result.error_message or "")


@pytest.mark.asyncio
async def test_remote_bridge_image_json_skips_invalid_images_when_others_are_valid(
    workspace_root,
    monkeypatch,
):
    cli, store = _create_cli(workspace_root, monkeypatch)
    seen_calls = []

    async def fake_run_agent(user_input, has_image=False):
        seen_calls.append((user_input, has_image))
        return "processed:partial remote images"

    monkeypatch.setattr(cli, "_run_agent", fake_run_agent)

    png_base64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwC"
        "AAAAC0lEQVR42mP8/x8AAwMCAO+a4e0AAAAASUVORK5CYII="
    )
    store.enqueue_text(
        (
            '{"msgType":"image","imageDataBase64":["not-base64","'
            + png_base64
            + '"],"userInput":"只看有效图片"}'
        ),
        source="remote",
        source_meta={"delivery_id": "delivery-image-partial-1"},
        remote_response_required=True,
        request_id="remote-image-partial-1",
    )

    should_continue = await cli._drain_bridge_queue()

    assert should_continue is True
    assert len(seen_calls) == 1
    user_input, has_image = seen_calls[0]
    assert has_image is True
    assert isinstance(user_input, list)
    assert [part.type for part in user_input] == ["text", "image_url"]
    assert user_input[0].text == "只看有效图片"
    assert user_input[1].image_url.media_type == "image/png"

    result = store.find_result("remote-image-partial-1")
    assert result is not None
    assert result.final_status == "completed"
    assert result.input_kind == "image"
    assert result.final_content == "processed:partial remote images"


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
async def test_local_reset_does_not_trigger_startup_bootstrap(workspace_root, monkeypatch):
    cli, _store = _create_cli(workspace_root, monkeypatch)

    async def fail_query(message: str):
        raise AssertionError(f"local /reset should not call query: {message}")

    monkeypatch.setattr(cli._agent, "query", fail_query)

    handled = await cli._handle_slash_command("/reset")

    assert handled is True


@pytest.mark.asyncio
async def test_remote_reset_bootstraps_new_session_and_returns_greeting(
    workspace_root,
    monkeypatch,
):
    cli, store = _create_cli(workspace_root, monkeypatch)
    (workspace_root / "TGAGENTS.md").write_text("repo rules", encoding="utf-8")
    cli._agent._context.add_message(UserMessage(content="old question"))
    cli._agent._context.add_message(AssistantMessage(content="old answer"))

    seen_user_prompts: list[str] = []

    async def fake_ainvoke(messages, tools=None, tool_choice=None, **kwargs):
        del tools, tool_choice, kwargs
        for message in reversed(messages):
            if getattr(message, "role", None) == "user":
                seen_user_prompts.append(str(getattr(message, "content", "")))
                break
        return ChatInvokeCompletion(content="你好呀，想做什么？")

    async def fail_run_agent(user_input, has_image=False, agent=None):
        del user_input, has_image, agent
        raise AssertionError("remote /reset should not go through _run_agent")

    monkeypatch.setattr(cli._agent.llm, "ainvoke", fake_ainvoke)
    monkeypatch.setattr(cli, "_run_agent", fail_run_agent)

    store.enqueue_text(
        "/reset",
        source="remote",
        source_meta={"delivery_id": "delivery-reset-1"},
        remote_response_required=True,
        request_id="remote-reset-1",
    )

    should_continue = await cli._drain_bridge_queue()

    assert should_continue is True
    result = store.find_result("remote-reset-1")
    assert result is not None
    assert result.final_status == "completed"
    assert result.final_content == "你好呀，想做什么？"
    assert seen_user_prompts
    assert "已通过 /clear 命令启动了新会话" in seen_user_prompts[-1]
    assert "当前运行模型：echo-worker" in seen_user_prompts[-1]
    assert "GLM-4.7" not in seen_user_prompts[-1]

    system_contents = [
        str(getattr(message, "content", ""))
        for message in cli._agent.messages
        if message.role == "system"
    ]
    assert "test" in system_contents
    assert "repo rules" in system_contents
    assert all("old question" not in content for content in system_contents)
    assert all("old answer" not in content for content in system_contents)

    user_contents = [
        str(getattr(message, "content", ""))
        for message in cli._agent.messages
        if message.role == "user"
    ]
    assert any("已通过 /clear 命令启动了新会话" in content for content in user_contents)
    assert all("old question" not in content for content in user_contents)

    assistant_contents = [
        str(getattr(message, "content", ""))
        for message in cli._agent.messages
        if message.role == "assistant"
    ]
    assert assistant_contents == ["你好呀，想做什么？"]


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
