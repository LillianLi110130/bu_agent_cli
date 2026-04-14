from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import MethodType

import pytest

from agent_core.agent.budget import BudgetAssessment
from agent_core.agent.compaction import (
    CompactionConfig,
    CompactionResult,
    CompactionService,
    CompactionWorkingState,
)
from agent_core.agent.context import ContextManager
from agent_core.agent.context_store import ArtifactStore, CheckpointStore, WorkingStateStore
from agent_core.llm.messages import AssistantMessage, BaseMessage, UserMessage
from agent_core.llm.views import ChatInvokeCompletion, ChatInvokeCompletionChunk
from cli.session_runtime import CLISessionRuntime
from tools import SandboxContext

STRUCTURED_COMPACTION_RESPONSE = """<summary>
Investigated the bug and preserved the next steps.
</summary>
<working_state>
{
  "user_goal": "Fix the failing parser flow",
  "user_constraints": ["Do not change the public API"],
  "confirmed_conclusions": ["The regression started in parser.py"],
  "files_reviewed": ["parser.py", "tests/test_parser.py"],
  "files_modified": ["parser.py"],
  "failed_attempts": ["Tried to normalize too early"],
  "remaining_actions": ["Update parser edge case", "Run parser tests"],
  "artifact_refs": ["checkpoint://compaction-inline-1"],
  "recent_history_notes": ["Need to preserve the latest reproduction details"]
}
</working_state>
<checkpoint_ref>
compaction-inline-1
</checkpoint_ref>"""


class FakeCompactionLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.invocations: list[list[BaseMessage]] = []
        self.model = "fake-model"

    @property
    def provider(self) -> str:
        return "fake"

    @property
    def name(self) -> str:
        return self.model

    async def ainvoke(
        self,
        messages: list[BaseMessage],
        tools=None,
        tool_choice=None,
        **kwargs,
    ) -> ChatInvokeCompletion:
        self.invocations.append(list(messages))
        if not self.responses:
            raise AssertionError("No scripted compaction response left")
        return ChatInvokeCompletion(content=self.responses.pop(0))

    async def astream(
        self,
        messages: list[BaseMessage],
        tools=None,
        tool_choice=None,
        **kwargs,
    ) -> AsyncIterator[ChatInvokeCompletionChunk]:
        if False:
            yield ChatInvokeCompletionChunk()


@pytest.mark.asyncio
async def test_compaction_service_parses_structured_working_state():
    llm = FakeCompactionLLM([STRUCTURED_COMPACTION_RESPONSE])
    service = CompactionService(config=CompactionConfig(), llm=llm)

    result = await service.compact([UserMessage(content="Fix the parser flow")], llm)

    assert result.summary == "Investigated the bug and preserved the next steps."
    assert result.working_state is not None
    assert result.working_state.user_goal == "Fix the failing parser flow"
    assert result.working_state.files_reviewed == ["parser.py", "tests/test_parser.py"]
    assert result.checkpoint_ref == "compaction-inline-1"
    compacted_message = service.create_compacted_messages(result)[0]
    assert "[Compacted Working Set]" in compacted_message.content
    assert "Remaining Actions:" in compacted_message.content
    assert "Checkpoint Ref: compaction-inline-1" in compacted_message.content


def test_compaction_merge_replaces_stale_active_state():
    service = CompactionService(config=CompactionConfig())
    previous = CompactionResult(
        compacted=True,
        summary="previous summary",
        working_state=CompactionWorkingState(
            confirmed_conclusions=["Git repository path was unknown"],
            remaining_actions=["Ask user for the git repository path"],
            recent_history_notes=["User was chatting casually"],
        ),
        checkpoint_ref="checkpoint://0001",
    )
    current = CompactionResult(
        compacted=True,
        summary="current summary",
        working_state=CompactionWorkingState(
            confirmed_conclusions=["Git pull completed successfully"],
            remaining_actions=["Check shutdown task configuration"],
            recent_history_notes=["Windows shutdown prevention query is in progress"],
        ),
        checkpoint_ref="checkpoint://0002",
    )

    merged = service.merge_results(previous, current)

    assert merged.working_state is not None
    assert merged.working_state.confirmed_conclusions == [
        "Git repository path was unknown",
        "Git pull completed successfully",
    ]
    assert merged.working_state.remaining_actions == ["Check shutdown task configuration"]
    assert merged.working_state.recent_history_notes == [
        "Windows shutdown prevention query is in progress"
    ]
    assert merged.checkpoint_ref == "checkpoint://0002"


@pytest.mark.asyncio
async def test_check_and_compact_injects_working_set_plus_recent_history():
    llm = FakeCompactionLLM([STRUCTURED_COMPACTION_RESPONSE])
    context = ContextManager(
        messages=[
            UserMessage(content="user-1"),
            AssistantMessage(content="assistant-1"),
            UserMessage(content="user-2"),
            AssistantMessage(content="assistant-2"),
            UserMessage(content="user-3"),
        ]
    )
    context.configure_compaction(
        config=CompactionConfig(preserve_recent_messages=2, preserve_recent_token_ratio=1.0),
        llm=llm,
        token_cost=None,
    )

    async def fake_assess_budget(self, *, model: str, usage=None, trigger=None):
        return BudgetAssessment(
            model=model,
            context_limit=100,
            warn_threshold=50,
            compact_threshold=60,
            hard_threshold=90,
            baseline_prompt_tokens=0,
            incremental_tokens=80,
            estimated_tokens=80,
            message_count=len(self._messages),
            warn_threshold_ratio=0.5,
            compact_threshold_ratio=0.6,
            hard_threshold_ratio=0.9,
            threshold_utilization=1.33,
            context_utilization=0.8,
            trigger=trigger,
        )

    context.assess_budget = MethodType(fake_assess_budget, context)
    changed = await context.check_and_compact(llm)

    assert changed is True
    contents = [getattr(message, "content", "") for message in context.get_messages()]
    assert "[Compacted Working Set]" in contents[0]
    assert contents[1:] == ["assistant-2", "user-3"]
    assert context.summarized_boundary == 1


@pytest.mark.asyncio
async def test_check_and_compact_passes_recent_tail_as_reference():
    llm = FakeCompactionLLM([STRUCTURED_COMPACTION_RESPONSE])
    context = ContextManager(
        messages=[
            UserMessage(content="rope是怎么实现的"),
            AssistantMessage(content="RoPE rotates query/key pairs with sinusoidal angles."),
            UserMessage(content="再看看DDP怎么做的"),
        ]
    )
    context.configure_compaction(
        config=CompactionConfig(preserve_recent_messages=2, preserve_recent_token_ratio=1.0),
        llm=llm,
        token_cost=None,
    )

    async def fake_assess_budget(self, *, model: str, usage=None, trigger=None):
        return BudgetAssessment(
            model=model,
            context_limit=2000,
            warn_threshold=1000,
            compact_threshold=1000,
            hard_threshold=1800,
            baseline_prompt_tokens=0,
            incremental_tokens=1200,
            estimated_tokens=1200,
            message_count=len(self._messages),
            warn_threshold_ratio=0.5,
            compact_threshold_ratio=0.5,
            hard_threshold_ratio=0.9,
            threshold_utilization=1.2,
            context_utilization=0.6,
            trigger=trigger,
        )

    context.assess_budget = MethodType(fake_assess_budget, context)
    changed = await context.check_and_compact(llm)

    assert changed is True
    compaction_input = "\n".join(
        str(getattr(message, "content", "")) for message in llm.invocations[0]
    )
    assert "rope是怎么实现的" in compaction_input
    assert "<recent_tail_reference>" in compaction_input
    assert "RoPE rotates query/key pairs with sinusoidal angles." in compaction_input
    assert "再看看DDP怎么做的" in compaction_input
    assert "Do not list actions as remaining if these messages already answered" in compaction_input


@pytest.mark.asyncio
async def test_check_and_compact_limits_recent_tail_by_token_budget():
    llm = FakeCompactionLLM([STRUCTURED_COMPACTION_RESPONSE])
    huge_tool_like_tail = "x" * 1200
    context = ContextManager(
        messages=[
            UserMessage(content="older-user"),
            AssistantMessage(content="older-assistant"),
            AssistantMessage(content=huge_tool_like_tail),
        ]
    )
    context.configure_compaction(
        config=CompactionConfig(preserve_recent_messages=2, preserve_recent_token_ratio=0.05),
        llm=llm,
        token_cost=None,
    )

    async def fake_assess_budget(self, *, model: str, usage=None, trigger=None):
        return BudgetAssessment(
            model=model,
            context_limit=1000,
            warn_threshold=600,
            compact_threshold=200,
            hard_threshold=900,
            baseline_prompt_tokens=0,
            incremental_tokens=400,
            estimated_tokens=400,
            message_count=len(self._messages),
            warn_threshold_ratio=0.6,
            compact_threshold_ratio=0.2,
            hard_threshold_ratio=0.9,
            threshold_utilization=2.0,
            context_utilization=0.4,
            trigger=trigger,
        )

    context.assess_budget = MethodType(fake_assess_budget, context)

    changed = await context.check_and_compact(llm)

    assert changed is True
    assert len(context.get_messages()) == 1
    assert "[Compacted Working Set]" in str(context.get_messages()[0].content)


@pytest.mark.asyncio
async def test_sliding_window_does_not_recompact_existing_working_set():
    llm = FakeCompactionLLM([STRUCTURED_COMPACTION_RESPONSE, STRUCTURED_COMPACTION_RESPONSE])
    context = ContextManager(
        messages=[
            UserMessage(content="older-user"),
            AssistantMessage(content="older-assistant"),
            UserMessage(content="latest-user"),
            AssistantMessage(content="latest-assistant"),
        ],
        sliding_window_messages=1,
    )
    context.configure_compaction(
        config=CompactionConfig(preserve_recent_messages=1),
        llm=llm,
        token_cost=None,
    )

    first_result = await context._compaction_service.compact(  # type: ignore[union-attr]
        [UserMessage(content="older-user"), AssistantMessage(content="older-assistant")],
        llm,
    )
    context.apply_compaction_result(
        first_result,
        recent_messages=[
            UserMessage(content="latest-user"),
            AssistantMessage(content="latest-assistant"),
        ],
    )
    llm.invocations.clear()

    changed = await context.apply_sliding_window_by_messages(
        keep_count=1,
        llm=llm,
        buffer=0,
    )

    assert changed is True
    compacted_input = llm.invocations[0]
    assert all(
        "[Compacted Working Set]" not in str(getattr(message, "content", ""))
        for message in compacted_input
    )


@pytest.mark.asyncio
async def test_compaction_with_filesystem_stores_persists_checkpoint_and_working_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("HOME", str(tmp_path))
    llm = FakeCompactionLLM([STRUCTURED_COMPACTION_RESPONSE])
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sandbox = SandboxContext.create(workspace)
    runtime = CLISessionRuntime.create_for_context(
        sandbox,
        now=datetime(2026, 4, 1, 15, 30, 15, tzinfo=timezone(timedelta(hours=8))),
    )
    context = ContextManager(
        messages=[
            UserMessage(content="user-1"),
            AssistantMessage(content="assistant-1"),
            UserMessage(content="user-2"),
            AssistantMessage(content="assistant-2"),
            UserMessage(content="user-3"),
        ]
    )
    context.bind_filesystem_stores(
        artifact_store=ArtifactStore(runtime.artifacts_dir),
        checkpoint_store=CheckpointStore(runtime.checkpoints_dir),
        working_state_store=WorkingStateStore(
            runtime.working_state_path,
            session_id=runtime.session_id,
        ),
    )
    context.configure_compaction(
        config=CompactionConfig(preserve_recent_messages=2, preserve_recent_token_ratio=1.0),
        llm=llm,
        token_cost=None,
    )

    async def fake_assess_budget(self, *, model: str, usage=None, trigger=None):
        return BudgetAssessment(
            model=model,
            context_limit=100,
            warn_threshold=50,
            compact_threshold=60,
            hard_threshold=90,
            baseline_prompt_tokens=0,
            incremental_tokens=80,
            estimated_tokens=80,
            message_count=len(self._messages),
            warn_threshold_ratio=0.5,
            compact_threshold_ratio=0.6,
            hard_threshold_ratio=0.9,
            threshold_utilization=1.33,
            context_utilization=0.8,
            trigger=trigger,
        )

    context.assess_budget = MethodType(fake_assess_budget, context)
    changed = await context.check_and_compact(llm)

    assert changed is True
    checkpoint_path = runtime.checkpoints_dir / "0001.jsonl"
    assert checkpoint_path.exists()

    working_state = json.loads(runtime.working_state_path.read_text(encoding="utf-8"))
    assert working_state["session_id"] == runtime.session_id
    assert working_state["goal"] == "Fix the failing parser flow"
    assert working_state["checkpoint_refs"] == ["checkpoint://0001"]
    assert working_state["checkpoint_paths"] == [str(checkpoint_path)]

    compacted_content = str(context.get_messages()[0].content)
    assert "Checkpoint Ref: checkpoint://0001" in compacted_content
    assert f"Checkpoint Path: {checkpoint_path}" in compacted_content


def test_working_state_store_replaces_stale_active_state(tmp_path: Path):
    state_path = tmp_path / "working_state.json"
    store = WorkingStateStore(state_path, session_id="session-1")
    store.ensure_initialized()
    state_path.write_text(
        json.dumps(
            {
                "session_id": "session-1",
                "goal": "Prevent computer from auto-shutting down",
                "constraints": ["Windows environment"],
                "confirmed_facts": ["Git repository path was unknown"],
                "decisions": [],
                "failed_attempts": [],
                "files_checked": [],
                "files_modified": [],
                "open_questions": ["User was chatting casually"],
                "next_steps": ["Ask user for the git repository path"],
                "artifact_refs": [],
                "checkpoint_refs": ["checkpoint://0001"],
                "checkpoint_paths": [],
                "last_compacted_at": "2026-04-14T08:00:00+08:00",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    store.write_result(
        CompactionResult(
            compacted=True,
            summary="current summary",
            working_state=CompactionWorkingState(
                confirmed_conclusions=["Git pull completed successfully"],
                remaining_actions=["Check shutdown task configuration"],
                recent_history_notes=["Windows shutdown prevention query is in progress"],
            ),
            checkpoint_ref="checkpoint://0002",
            checkpoint_path=str(tmp_path / "0002.jsonl"),
        )
    )

    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["confirmed_facts"] == [
        "Git repository path was unknown",
        "Git pull completed successfully",
    ]
    assert payload["next_steps"] == ["Check shutdown task configuration"]
    assert payload["open_questions"] == ["Windows shutdown prevention query is in progress"]
    assert payload["checkpoint_refs"] == ["checkpoint://0001", "checkpoint://0002"]
