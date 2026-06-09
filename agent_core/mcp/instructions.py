"""MCP server instruction context deltas."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

from agent_core.llm.messages import BaseMessage, UserMessage

MAX_MCP_INSTRUCTIONS_LENGTH = 2048
MCP_INSTRUCTIONS_DELTA_MARKER = "[MCP Instructions Delta]"
_META_RE = re.compile(r"^\[MCP Instructions Delta\]\s*(\{.*\})$", re.MULTILINE)


@dataclass(slots=True)
class MCPInstructionsDelta:
    added_names: list[str]
    added_blocks: list[str]
    removed_names: list[str]

    @property
    def is_empty(self) -> bool:
        return not self.added_names and not self.removed_names


def sync_mcp_instruction_reminders(agent: Any, ctx: Any) -> None:
    """Append MCP instruction deltas to the agent context when they change."""
    manager = getattr(ctx, "mcp_manager", None)
    agent_context = getattr(agent, "_context", None)
    if manager is None or agent_context is None:
        return

    delta = get_mcp_instructions_delta(
        manager.instructions(),
        agent_context.get_messages(),
    )
    if delta.is_empty:
        return
    agent_context.add_message(UserMessage(content=render_mcp_instructions_delta(delta)))


def get_mcp_instructions_delta(
    current_instructions: dict[str, str],
    messages: Iterable[BaseMessage],
) -> MCPInstructionsDelta:
    """Return newly connected and disconnected MCP instruction changes."""
    announced = _announced_instruction_servers(messages)
    current_names = {
        name
        for name, instructions in current_instructions.items()
        if instructions and instructions.strip()
    }

    added_names = sorted(current_names - announced)
    removed_names = sorted(announced - current_names)
    added_blocks = [
        f"## {name}\n{_truncate_instructions(current_instructions[name].strip())}"
        for name in added_names
    ]
    return MCPInstructionsDelta(
        added_names=added_names,
        added_blocks=added_blocks,
        removed_names=removed_names,
    )


def render_mcp_instructions_delta(delta: MCPInstructionsDelta) -> str:
    """Render a Claude Code-like MCP instructions delta reminder."""
    metadata = {
        "addedNames": delta.added_names,
        "removedNames": delta.removed_names,
    }
    sections = [
        MCP_INSTRUCTIONS_DELTA_MARKER,
        json.dumps(metadata, ensure_ascii=False, separators=(",", ":")),
        "",
        "<system-reminder>",
    ]
    if delta.added_blocks:
        sections.extend(
            [
                "# MCP Server Instructions",
                "",
                "The following MCP servers have provided instructions for how to use their tools and resources:",
                "",
                "\n\n".join(delta.added_blocks),
            ]
        )
    if delta.removed_names:
        if delta.added_blocks:
            sections.append("")
        sections.extend(
            [
                "The following MCP servers have disconnected. Their instructions above no longer apply:",
                ", ".join(delta.removed_names),
            ]
        )
    sections.append("</system-reminder>")
    return "\n".join(sections)


def _announced_instruction_servers(messages: Iterable[BaseMessage]) -> set[str]:
    announced: set[str] = set()
    for message in messages:
        if not isinstance(message, UserMessage):
            continue
        text = message.text
        if MCP_INSTRUCTIONS_DELTA_MARKER not in text:
            continue
        for match in _META_RE.finditer(text):
            with_json = match.group(1)
            try:
                payload = json.loads(with_json)
            except json.JSONDecodeError:
                continue
            added = payload.get("addedNames", [])
            removed = payload.get("removedNames", [])
            if isinstance(added, list):
                announced.update(name for name in added if isinstance(name, str))
            if isinstance(removed, list):
                announced.difference_update(name for name in removed if isinstance(name, str))
    return announced


def _truncate_instructions(instructions: str) -> str:
    if len(instructions) <= MAX_MCP_INSTRUCTIONS_LENGTH:
        return instructions
    return instructions[:MAX_MCP_INSTRUCTIONS_LENGTH].rstrip() + "..."
