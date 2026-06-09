from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from agent_core.agent.context import ContextManager
from agent_core.llm.messages import UserMessage
from agent_core.mcp.instructions import (
    MAX_MCP_INSTRUCTIONS_LENGTH,
    get_mcp_instructions_delta,
    sync_mcp_instruction_reminders,
)


class FakeManager:
    def __init__(self) -> None:
        self.current: dict[str, str] = {}

    def instructions(self) -> dict[str, str]:
        return self.current


def _agent_with_context() -> SimpleNamespace:
    return SimpleNamespace(_context=ContextManager())


def test_sync_mcp_instruction_reminders_adds_and_removes_delta(tmp_path: Path) -> None:
    ctx = SimpleNamespace(root_dir=tmp_path)
    manager = FakeManager()
    ctx.mcp_manager = manager
    agent = _agent_with_context()

    manager.current = {"codegraph": "Prefer codegraph_search before broad text search."}
    sync_mcp_instruction_reminders(agent, ctx)

    messages = agent._context.get_messages()
    assert len(messages) == 1
    assert isinstance(messages[0], UserMessage)
    assert "# MCP Server Instructions" in messages[0].text
    assert "## codegraph" in messages[0].text

    sync_mcp_instruction_reminders(agent, ctx)
    assert len(agent._context.get_messages()) == 1

    manager.current = {}
    sync_mcp_instruction_reminders(agent, ctx)

    messages = agent._context.get_messages()
    assert len(messages) == 2
    assert "have disconnected" in messages[-1].text
    assert "codegraph" in messages[-1].text

    manager.current = {"codegraph": "Use codegraph again."}
    sync_mcp_instruction_reminders(agent, ctx)

    messages = agent._context.get_messages()
    assert len(messages) == 3
    assert "Use codegraph again." in messages[-1].text


def test_get_mcp_instructions_delta_tracks_announced_servers() -> None:
    existing = UserMessage(
        content=(
            '[MCP Instructions Delta]\n{"addedNames":["codegraph"],"removedNames":[]}\n'
            "<system-reminder>old</system-reminder>"
        )
    )

    delta = get_mcp_instructions_delta(
        {"codegraph": "Already known.", "search": "New server."},
        [existing],
    )

    assert delta.added_names == ["search"]
    assert delta.removed_names == []


def test_mcp_instruction_blocks_are_truncated() -> None:
    delta = get_mcp_instructions_delta({"big": "x" * 3000}, [])

    assert delta.added_names == ["big"]
    assert len(delta.added_blocks[0]) <= MAX_MCP_INSTRUCTIONS_LENGTH + len("## big\n...")
