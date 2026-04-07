from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from agent_core import Agent
from agent_core.tools import tool
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

    artifact_meta = runtime.artifacts_dir / "tool" / "call-bash.meta.json"
    artifact_payload = json.loads(artifact_meta.read_text(encoding="utf-8"))
    assert artifact_payload["content_format"] == "json"
    raw_payload = json.loads(Path(artifact_payload["content_path"]).read_text(encoding="utf-8"))
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
    assert "Artifact content:" in tool_message.text

    artifact_meta = runtime.artifacts_dir / "tool" / "call-read-large.meta.json"
    artifact_payload = json.loads(artifact_meta.read_text(encoding="utf-8"))
    assert artifact_payload["tool_name"] == "read"
    assert artifact_payload["content_format"] == "text"
    artifact_content = Path(artifact_payload["content_path"]).read_text(encoding="utf-8")
    assert artifact_content == raw_result


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


@pytest.mark.asyncio
async def test_read_excel_context_policy_summarizes_large_match_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TG_AGENT_HOME", str(tmp_path / ".tg_agent"))

    @tool(
        "Read workbook",
        name="read_excel",
        context_policy="summarize",
        context_max_inline_chars=2200,
    )
    async def fake_read_excel() -> str:
        payload = {
            "resolved_path": "/workspace/orders.xlsx",
            "sheet_names": ["Sheet1"],
            "selected_sheet": "Sheet1",
            "preview_limits": {
                "find_text": "是",
                "offset_row": 1,
                "context_rows": 0,
                "max_matches": 500,
                "max_rows": 10,
                "max_cols": 15,
            },
            "matches": [
                {
                    "sheet": "Sheet1",
                    "row": index,
                    "matched_columns": [7],
                    "preview_rows": [
                        {
                            "row": index,
                            "values": [
                                f"工单{index}",
                                "托管业务机房维护申请",
                                "是",
                                "定期提数",
                                "2025-01-01",
                                "附加说明" * 8,
                            ],
                        }
                    ],
                }
                for index in range(2, 202)
            ],
            "sheets": [
                {
                    "name": "Sheet1",
                    "row_count": 1955,
                    "column_count": 42,
                    "preview_rows": [
                        {"row": 1, "values": ["标题1", "标题2", "标题3"]},
                        {"row": 2, "values": ["示例", "是", "定期提数"]},
                    ],
                }
            ],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sandbox = SandboxContext.create(workspace)
    runtime = CLISessionRuntime.create_for_context(sandbox)
    agent = Agent(
        llm=EchoLLM(prefix="echo:"),
        tools=[fake_read_excel],
        system_prompt="test",
    )
    agent.bind_session_runtime(runtime)

    tool_message = await agent._execute_tool_call(
        ToolCall(
            id="call-excel",
            function=Function(name="read_excel", arguments="{}"),
        )
    )

    assert "Excel workbook: /workspace/orders.xlsx" in tool_message.text
    assert "Matches returned: 200" in tool_message.text
    assert "Sheet summary: Sheet1 rows=1955, cols=42" in tool_message.text
    assert "Top matches:" in tool_message.text
    assert "Artifact content:" in tool_message.text

    artifact_meta = runtime.artifacts_dir / "tool" / "call-excel.meta.json"
    artifact_payload = json.loads(artifact_meta.read_text(encoding="utf-8"))
    assert artifact_payload["tool_name"] == "read_excel"
    assert artifact_payload["content_format"] == "json"
    assert "\"matches\"" in Path(artifact_payload["content_path"]).read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_reading_runtime_artifact_requires_window_and_skips_repersist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TG_AGENT_HOME", str(tmp_path / ".tg_agent"))

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "large.txt"
    target.write_text("".join(f"line {i:04d}\n" for i in range(1, 301)), "utf-8")

    sandbox = SandboxContext.create(workspace)
    runtime = CLISessionRuntime.create_for_context(sandbox)
    agent = Agent(
        llm=EchoLLM(prefix="echo:"),
        tools=[read],
        system_prompt="test",
        dependency_overrides={get_sandbox_context: lambda: sandbox},
    )
    agent.bind_session_runtime(runtime)

    first_tool_message = await agent._execute_tool_call(
        ToolCall(
            id="call-read-large-window",
            function=Function(name="read", arguments=json.dumps({"file_path": "large.txt"})),
        )
    )
    assert "Artifact content:" in first_tool_message.text

    artifact_meta = runtime.artifacts_dir / "tool" / "call-read-large-window.meta.json"
    artifact_payload = json.loads(artifact_meta.read_text(encoding="utf-8"))
    artifact_content_path = artifact_payload["content_path"]

    missing_window = await read.execute(
        file_path=artifact_content_path,
        _overrides={get_sandbox_context: lambda: sandbox},
    )
    assert "requires explicit offset_line and n_lines" in str(missing_window)

    artifact_tool_message = await agent._execute_tool_call(
        ToolCall(
            id="call-read-artifact-slice",
            function=Function(
                name="read",
                arguments=json.dumps(
                    {
                        "file_path": artifact_content_path,
                        "offset_line": 10,
                        "n_lines": 40,
                    }
                ),
            ),
        )
    )

    assert artifact_tool_message.text.startswith("[Lines 10-49")
    assert "Artifact content:" not in artifact_tool_message.text
    assert not (runtime.artifacts_dir / "tool" / "call-read-artifact-slice.meta.json").exists()
