"""Streaming parser for model thinking tags."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

THINK_OPEN_TAG = "<think>"
THINK_CLOSE_TAG = "</think>"

ThinkingParseEventKind = Literal["text", "thinking_start", "thinking_delta", "thinking_end"]


@dataclass(frozen=True)
class ThinkingParseEvent:
    kind: ThinkingParseEventKind
    content: str = ""


class ThinkTagParser:
    """Parse explicit and implicit leading think blocks from streaming text.

    Supported formats:
    - <think>thinking</think>answer
    - thinking</think>answer

    The implicit form is only recognized before any normal text has been emitted.
    Until the closing tag appears, the leading text is buffered so normal answers
    without think tags can still be flushed as regular text at the end.
    """

    def __init__(self, *, allow_implicit_leading: bool = True) -> None:
        self.allow_implicit_leading = allow_implicit_leading
        self.in_think = False
        self.think_end = False
        self.tag_buffer = ""
        self.filtered_content = ""
        self.think_id = "think_0"
        self._normal_text_emitted = False
        self._open_len = len(THINK_OPEN_TAG)
        self._close_len = len(THINK_CLOSE_TAG)
        self._thinking_started = False
        self._pending_compat_events: list[ThinkingParseEvent] = []

    def feed_events(self, delta: str) -> list[ThinkingParseEvent]:
        """Parse one stream delta into text/thinking events."""
        if not delta:
            return []

        self.tag_buffer += delta
        events: list[ThinkingParseEvent] = []

        while self.tag_buffer:
            if self.think_end:
                self._emit_text(self.tag_buffer, events)
                self.tag_buffer = ""
                break

            if self.in_think:
                close_idx = self.tag_buffer.find(THINK_CLOSE_TAG)
                if close_idx != -1:
                    thinking = self.tag_buffer[:close_idx]
                    self._emit_thinking_delta(thinking, events)
                    self.tag_buffer = self.tag_buffer[close_idx + self._close_len :]
                    self.in_think = False
                    self.think_end = True
                    self._emit_thinking_end(events)
                    continue

                safe_len = len(self.tag_buffer) - (self._close_len - 1)
                if safe_len > 0:
                    thinking = self.tag_buffer[:safe_len]
                    self.tag_buffer = self.tag_buffer[safe_len:]
                    self._emit_thinking_delta(thinking, events)
                    continue
                break

            open_idx = self.tag_buffer.find(THINK_OPEN_TAG)
            close_idx = self.tag_buffer.find(THINK_CLOSE_TAG)

            if (
                self.allow_implicit_leading
                and not self._normal_text_emitted
                and close_idx != -1
                and (open_idx == -1 or close_idx < open_idx)
            ):
                thinking = self.tag_buffer[:close_idx]
                self._emit_thinking_start(events)
                self._emit_thinking_delta(thinking, events)
                self._emit_thinking_end(events)
                self.tag_buffer = self.tag_buffer[close_idx + self._close_len :]
                self.think_end = True
                continue

            if open_idx != -1:
                normal_text = self.tag_buffer[:open_idx]
                self._emit_text(normal_text, events)
                self.tag_buffer = self.tag_buffer[open_idx + self._open_len :]
                self.in_think = True
                self._emit_thinking_start(events)
                continue

            if self.allow_implicit_leading and not self._normal_text_emitted:
                break

            safe_len = len(self.tag_buffer) - (self._open_len - 1)
            if safe_len > 0:
                text = self.tag_buffer[:safe_len]
                self.tag_buffer = self.tag_buffer[safe_len:]
                self._emit_text(text, events)
                continue
            break

        return events

    def feed(self, delta: str) -> tuple[str | None, str | None, str | None, bool]:
        """Backward-compatible single-event parser API.

        Returns (normal_text, thinking_content, event_type, just_ended).
        """
        if delta or not self._pending_compat_events:
            self._pending_compat_events.extend(self.feed_events(delta))
        if not self._pending_compat_events:
            return None, None, None, False

        first = self._pending_compat_events.pop(0)
        if first.kind == "text":
            return first.content or None, None, None, False
        if first.kind == "thinking_start":
            return None, None, "start", False
        if first.kind == "thinking_delta":
            return None, first.content or None, None, False
        return None, None, "end", True

    def flush_events(self) -> list[ThinkingParseEvent]:
        """Flush remaining buffered text at stream end."""
        events: list[ThinkingParseEvent] = []
        if self.tag_buffer:
            if self.in_think:
                text = THINK_OPEN_TAG + self.tag_buffer
                self.in_think = False
            else:
                text = self.tag_buffer
            self.tag_buffer = ""
            self._emit_text(text, events)
        return events

    def flush(self) -> tuple[str, bool]:
        """Backward-compatible flush API returning text and prior think state."""
        was_in_think = self.in_think
        events = self.flush_events()
        text = "".join(event.content for event in events if event.kind == "text")
        return text, was_in_think

    def get_filtered_content(self) -> str:
        return self.filtered_content

    def _emit_text(self, text: str, events: list[ThinkingParseEvent]) -> None:
        if not text:
            return
        self.filtered_content += text
        self._normal_text_emitted = True
        events.append(ThinkingParseEvent("text", text))

    def _emit_thinking_start(self, events: list[ThinkingParseEvent]) -> None:
        if self._thinking_started:
            return
        self._thinking_started = True
        events.append(ThinkingParseEvent("thinking_start"))

    def _emit_thinking_delta(self, text: str, events: list[ThinkingParseEvent]) -> None:
        if not text:
            return
        self._emit_thinking_start(events)
        events.append(ThinkingParseEvent("thinking_delta", text))

    def _emit_thinking_end(self, events: list[ThinkingParseEvent]) -> None:
        if not self._thinking_started:
            return
        self._thinking_started = False
        events.append(ThinkingParseEvent("thinking_end"))
