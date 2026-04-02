from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from agent_core import Agent
from agent_core.llm.messages import Function, ToolCall
from cli.session_runtime import CLISessionRuntime
from cli.worker.runtime_factory import EchoLLM
from tools.bash import _AsyncShellResult, bash
from tools.files import read
from tools.sandbox import SandboxContext, get_sandbox_context


@pytest.mark.asyncio
async def test_bash_context_policy_summarizes_and_persists_raw_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TG_AGENT_HOME", str(tmp_path / ".tg_agent"))

    async def fake_run_shell_command(command: str, cwd: str, timeout: int) -> _AsyncShellResult:
        del command, cwd, timeout
        return _AsyncShellResult(
            returncode=1,
            stdout="line one\nline two\nline three",
            stderr="fatal: repository has conflicts",
        )

    bash_module = importlib.import_module("tools.bash")
    monkeypatch.setattr(bash_module, "_run_shell_command", fake_run_shell_command)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sandbox = SandboxContext.create(workspace)
    runtime = CLISessionRuntime.create_for_context(sandbox)
    agent = Agent(
        llm=EchoLLM(prefix="echo:"),
        tools=[bash],
        system_prompt="test",
        dependency_overrides={get_sandbox_context: lambda: sandbox},
    )
    agent.bind_session_runtime(runtime)

    tool_message = await agent._execute_tool_call(
        ToolCall(
            id="call-bash",
            function=Function(name="bash", arguments=json.dumps({"command": "git status"})),
        )
    )

    assert "Bash command: git status" in tool_message.text
    assert "Exit code: 1" in tool_message.text
    assert "fatal: repository has conflicts" in tool_message.text
    assert '"stdout"' not in tool_message.text

    artifact_path = runtime.artifacts_dir / "tool" / "call-bash.json"
    artifact_payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    raw_payload = json.loads(str(artifact_payload["content"]))
    assert raw_payload["command"] == "git status"
    assert raw_payload["returncode"] == 1
    assert raw_payload["stdout"] == "line one\nline two\nline three"


@pytest.mark.asyncio
async def test_read_context_policy_trims_large_output_and_saves_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TG_AGENT_HOME", str(tmp_path / ".tg_agent"))

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "large.txt"
    target.write_text("".join(f"line {i:04d} " + ("x" * 40) + "\n" for i in range(1, 181)), "utf-8")

    sandbox = SandboxContext.create(workspace)
    runtime = CLISessionRuntime.create_for_context(sandbox)
    agent = Agent(
        llm=EchoLLM(prefix="echo:"),
        tools=[read],
        system_prompt="test",
        dependency_overrides={get_sandbox_context: lambda: sandbox},
    )
    agent.bind_session_runtime(runtime)

    raw_result = await read.execute(
        file_path="large.txt",
        _overrides={get_sandbox_context: lambda: sandbox},
    )
    tool_message = await agent._execute_tool_call(
        ToolCall(
            id="call-read-large",
            function=Function(name="read", arguments=json.dumps({"file_path": "large.txt"})),
        )
    )

    assert isinstance(raw_result, str)
    assert len(raw_result) > read.context_config.max_inline_chars
    assert tool_message.text != raw_result
    assert "Read result: [Lines 1-180 of 180]" in tool_message.text
    assert "Context preview:" in tool_message.text
    assert "Full excerpt:" in tool_message.text

    artifact_path = runtime.artifacts_dir / "tool" / "call-read-large.json"
    artifact_payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact_payload["tool_name"] == "read"
    assert artifact_payload["content"] == raw_result


@pytest.mark.asyncio
async def test_read_context_policy_keeps_small_output_inline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TG_AGENT_HOME", str(tmp_path / ".tg_agent"))

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "small.txt"
    target.write_text("alpha\nbeta\ngamma\n", "utf-8")

    sandbox = SandboxContext.create(workspace)
    agent = Agent(
        llm=EchoLLM(prefix="echo:"),
        tools=[read],
        system_prompt="test",
        dependency_overrides={get_sandbox_context: lambda: sandbox},
    )

    raw_result = await read.execute(
        file_path="small.txt",
        _overrides={get_sandbox_context: lambda: sandbox},
    )
    tool_message = await agent._execute_tool_call(
        ToolCall(
            id="call-read-small",
            function=Function(name="read", arguments=json.dumps({"file_path": "small.txt"})),
        )
    )

    assert tool_message.text == raw_result
    assert "[Lines 1-3 of 3]" in tool_message.text
