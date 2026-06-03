from __future__ import annotations

from agent_core.agent import CompactionConfig
from agent_core.agent.context import (
    ACTIVE_TODO_SNAPSHOT_HEADER,
    FINISH_GUARD_PROMPT_PREFIX,
    INVALID_TOOL_CALL_RECOVERY_PREFIX,
    MICROCOMPACTED_TOOL_RESULT_HEADER,
    ContextManager,
)
from agent_core.agent.context_store import ArtifactStore
from agent_core.llm.messages import AssistantMessage, Function, ToolCall, ToolMessage, UserMessage
from agent_core.llm.views import ChatInvokeUsage


class _FakeLLM:
    model = "fake-model"


def _usage(prompt_tokens: int = 100) -> ChatInvokeUsage:
    return ChatInvokeUsage(
        prompt_tokens=prompt_tokens,
        prompt_cached_tokens=0,
        prompt_cache_creation_tokens=0,
        prompt_image_tokens=0,
        completion_tokens=10,
        total_tokens=prompt_tokens + 10,
    )


def _tool_call(call_id: str = "call-1", name: str = "bash") -> ToolCall:
    return ToolCall(
        id=call_id,
        function=Function(name=name, arguments="{}"),
        type="function",
    )


def test_level1_snips_redundant_runtime_context_only() -> None:
    context = ContextManager()
    context.add_message(UserMessage(content="real user request"), new_user_round=True)
    context.add_message(UserMessage(content=f"{ACTIVE_TODO_SNAPSHOT_HEADER}\n- [ ] old"))
    context.add_message(UserMessage(content=f"{ACTIVE_TODO_SNAPSHOT_HEADER}\n- [ ] new"))
    context.add_message(UserMessage(content=f"{FINISH_GUARD_PROMPT_PREFIX}\nold"))
    context.add_message(UserMessage(content=f"{FINISH_GUARD_PROMPT_PREFIX}\nnew"))

    consumed_recovery = (
        f"{INVALID_TOOL_CALL_RECOVERY_PREFIX}and was rejected before execution.\n"
        "Validation errors:\n- old"
    )
    pending_recovery = (
        f"{INVALID_TOOL_CALL_RECOVERY_PREFIX}and was rejected before execution.\n"
        "Validation errors:\n- new"
    )
    context.add_message(UserMessage(content=consumed_recovery))
    context.add_message(AssistantMessage(content="model consumed the recovery prompt"))
    context.add_message(UserMessage(content=pending_recovery))

    result = context.snip_redundant_runtime_context()

    texts = [getattr(message, "text", "") for message in context.get_messages()]
    assert result.removed_count == 3
    assert "real user request" in texts
    assert f"{ACTIVE_TODO_SNAPSHOT_HEADER}\n- [ ] old" not in texts
    assert f"{ACTIVE_TODO_SNAPSHOT_HEADER}\n- [ ] new" in texts
    assert f"{FINISH_GUARD_PROMPT_PREFIX}\nold" not in texts
    assert f"{FINISH_GUARD_PROMPT_PREFIX}\nnew" in texts
    assert consumed_recovery not in texts
    assert pending_recovery in texts


def test_level2_microcompacts_old_seen_raw_tool_result_and_creates_artifact(tmp_path) -> None:
    context = ContextManager()
    context.bind_filesystem_stores(artifact_store=ArtifactStore(tmp_path))
    context.configure_compaction(CompactionConfig(), _FakeLLM(), None)

    context.add_message(UserMessage(content="round 1"), new_user_round=True)
    assistant = AssistantMessage(content=None, tool_calls=[_tool_call()])
    tool_message = ToolMessage(
        tool_call_id="call-1",
        tool_name="bash",
        content='{"command":"pytest -q","cwd":"D:/repo","returncode":1,'
        '"stdout":"' + ("x" * 2000) + '","stderr":""}',
        context_policy="raw",
    )
    context.add_message(assistant)
    context.add_message(tool_message)
    context.record_prompt_usage(
        model="fake-model",
        messages=context.get_messages(),
        usage=_usage(200),
    )
    context.add_message(UserMessage(content="round 2"), new_user_round=True)
    context.add_message(UserMessage(content="round 3"), new_user_round=True)
    context.add_message(UserMessage(content="round 4"), new_user_round=True)

    result = context.microcompact_tool_messages(
        summarize_tool_message=lambda message, artifact_path: "\n".join(
            [
                MICROCOMPACTED_TOOL_RESULT_HEADER,
                f"Tool: {message.tool_name}",
                f"Call ID: {message.tool_call_id}",
                "Summary: old pytest output.",
                f"Artifact: {artifact_path}",
            ]
        ),
        min_chars=100,
        preserve_recent_rounds=3,
    )

    assert result.microcompacted_count == 1
    assert result.artifact_created_count == 1
    assert tool_message.microcompacted is True
    assert tool_message.context_artifact_path is not None
    assert MICROCOMPACTED_TOOL_RESULT_HEADER in tool_message.text
    assert (tmp_path / "tool" / "call-1.artifact.txt").exists()


def test_level2_skips_recent_or_unseen_tool_results(tmp_path) -> None:
    context = ContextManager()
    context.bind_filesystem_stores(artifact_store=ArtifactStore(tmp_path))
    context.add_message(UserMessage(content="round 1"), new_user_round=True)
    recent_tool = ToolMessage(
        tool_call_id="call-recent",
        tool_name="read",
        content="recent output " * 100,
        context_policy="raw",
    )
    context.add_message(recent_tool)

    result = context.microcompact_tool_messages(
        summarize_tool_message=lambda message, artifact_path: "unused",
        min_chars=100,
        preserve_recent_rounds=3,
    )

    assert result.microcompacted_count == 0
    assert recent_tool.microcompacted is False


def test_level2_reuses_existing_artifact_path_for_old_summary() -> None:
    context = ContextManager()
    context.add_message(UserMessage(content="round 1"), new_user_round=True)
    tool_message = ToolMessage(
        tool_call_id="call-2",
        tool_name="bash",
        content="Bash command: pytest -q\n"
        "Cwd: D:/repo\n"
        "Exit code: 1\n"
        + ("summary preview\n" * 200),
        context_policy="trim",
        context_artifact_path="D:/tmp/call-2.artifact.txt",
    )
    context.add_message(tool_message)
    context.record_prompt_usage(
        model="fake-model",
        messages=context.get_messages(),
        usage=_usage(200),
    )
    context.add_message(UserMessage(content="round 2"), new_user_round=True)
    context.add_message(UserMessage(content="round 3"), new_user_round=True)
    context.add_message(UserMessage(content="round 4"), new_user_round=True)

    result = context.microcompact_tool_messages(
        summarize_tool_message=lambda message, artifact_path: "\n".join(
            [
                MICROCOMPACTED_TOOL_RESULT_HEADER,
                f"Tool: {message.tool_name}",
                f"Call ID: {message.tool_call_id}",
                "Summary: reused artifact.",
                f"Artifact: {artifact_path}",
            ]
        ),
        min_chars=100,
        preserve_recent_rounds=3,
    )

    assert result.microcompacted_count == 1
    assert result.artifact_created_count == 0
    assert tool_message.context_artifact_path == "D:/tmp/call-2.artifact.txt"
    assert "reused artifact" in tool_message.text
