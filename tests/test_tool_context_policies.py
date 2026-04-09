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


def _read_artifact_parts(path: Path) -> tuple[dict[str, str], str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    marker_index = lines.index("--- artifact_body ---")
    header: dict[str, str] = {}
    for line in lines[1:marker_index]:
        if not line.strip():
            continue
        key, _, value = line.partition(":")
        header[key.strip()] = value.strip()
    body = "\n".join(lines[marker_index + 1 :])
    return header, body


@pytest.mark.asyncio
async def test_bash_context_policy_keeps_small_output_inline(
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

    raw_result = await bash.execute(
        command="git status",
        _overrides={get_sandbox_context: lambda: sandbox},
    )
    tool_message = await agent._execute_tool_call(
        ToolCall(
            id="call-bash",
            function=Function(name="bash", arguments=json.dumps({"command": "git status"})),
        )
    )

    assert tool_message.text == raw_result
    assert '"command": "git status"' in tool_message.text
    assert '"stdout": "line one\\nline two\\nline three"' in tool_message.text
    assert "Artifact file:" not in tool_message.text

    artifact_path = runtime.artifacts_dir / "tool" / "call-bash.artifact.txt"
    assert not artifact_path.exists()


@pytest.mark.asyncio
async def test_bash_context_policy_trims_large_output_and_saves_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TG_AGENT_HOME", str(tmp_path / ".tg_agent"))

    async def fake_run_shell_command(command: str, cwd: str, timeout: int) -> _AsyncShellResult:
        del command, cwd, timeout
        return _AsyncShellResult(
            returncode=1,
            stdout="".join(f"stdout line {index:04d} " + ("x" * 40) + "\n" for index in range(1, 91)),
            stderr="".join(f"stderr line {index:04d} " + ("y" * 40) + "\n" for index in range(1, 81)),
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

    raw_result = await bash.execute(
        command="pytest",
        _overrides={get_sandbox_context: lambda: sandbox},
    )
    tool_message = await agent._execute_tool_call(
        ToolCall(
            id="call-bash-large",
            function=Function(name="bash", arguments=json.dumps({"command": "pytest"})),
        )
    )

    assert isinstance(raw_result, str)
    assert len(raw_result) > bash.context_config.max_inline_chars
    assert tool_message.text != raw_result
    assert "Bash command: pytest" in tool_message.text
    assert "Exit code: 1" in tool_message.text
    assert "Stdout: 90 lines" in tool_message.text
    assert "Stderr: 80 lines" in tool_message.text
    assert "stdout line 0001" in tool_message.text
    assert "stdout line 0090" in tool_message.text
    assert "stderr line 0001" in tool_message.text
    assert "stderr line 0080" in tool_message.text
    assert "Artifact file:" in tool_message.text

    artifact_path = runtime.artifacts_dir / "tool" / "call-bash-large.artifact.txt"
    header, body = _read_artifact_parts(artifact_path)
    assert header["content_format"] == "json"
    assert header["tool_name"] == "bash"
    raw_payload = json.loads(body)
    assert raw_payload["command"] == "pytest"
    assert raw_payload["returncode"] == 1
    assert "stdout line 0001" in raw_payload["stdout"]
    assert "stderr line 0080" in raw_payload["stderr"]


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
    assert "Artifact file:" in tool_message.text

    artifact_path = runtime.artifacts_dir / "tool" / "call-read-large.artifact.txt"
    header, body = _read_artifact_parts(artifact_path)
    assert header["tool_name"] == "read"
    assert header["content_format"] == "text"
    assert body == raw_result


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
    assert "Artifact file:" in tool_message.text

    artifact_path = runtime.artifacts_dir / "tool" / "call-excel.artifact.txt"
    header, body = _read_artifact_parts(artifact_path)
    assert header["tool_name"] == "read_excel"
    assert header["content_format"] == "json"
    assert "\"matches\"" in body


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
    assert "Artifact file:" in first_tool_message.text

    artifact_path = runtime.artifacts_dir / "tool" / "call-read-large-window.artifact.txt"
    header, _body = _read_artifact_parts(artifact_path)
    artifact_body_start_line = int(header["body_start_line"])

    missing_window = await read.execute(
        file_path=str(artifact_path),
        _overrides={get_sandbox_context: lambda: sandbox},
    )
    assert "requires explicit offset_line and n_lines" in str(missing_window)

    header_only = await read.execute(
        file_path=str(artifact_path),
        offset_line=1,
        n_lines=max(1, artifact_body_start_line - 1),
        _overrides={get_sandbox_context: lambda: sandbox},
    )
    assert f"The body starts at line {artifact_body_start_line}." in str(header_only)

    artifact_tool_message = await agent._execute_tool_call(
        ToolCall(
            id="call-read-artifact-slice",
            function=Function(
                name="read",
                arguments=json.dumps(
                    {
                        "file_path": str(artifact_path),
                        "offset_line": artifact_body_start_line,
                        "n_lines": 40,
                    }
                ),
            ),
        )
    )

    expected_end = artifact_body_start_line + 39
    assert artifact_tool_message.text.startswith(f"[Lines {artifact_body_start_line}-{expected_end}")
    assert "Artifact file:" not in artifact_tool_message.text
    assert not (runtime.artifacts_dir / "tool" / "call-read-artifact-slice.artifact.txt").exists()
