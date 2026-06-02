from __future__ import annotations

import asyncio

import cli.app as app_module
from cli.app import TGAgentCLI, _HistoryConsole
from cli.tui_app import TGAgentTUI


class _FakeLoop:
    @staticmethod
    def time() -> float:
        return 10.0


class _FakeCLI:
    def __init__(self) -> None:
        self.repaint_calls: list[dict[str, bool]] = []

    def _repaint_output_history(self, **kwargs: bool) -> None:
        self.repaint_calls.append(kwargs)


class _FakeConsole:
    def __init__(self) -> None:
        self.printed: list[str] = []
        self.clear_calls = 0

    def print(self, value: str) -> None:
        self.printed.append(value)

    def clear(self) -> None:
        self.clear_calls += 1


def _make_tui(monkeypatch, *, legacy_windows: bool):
    cli = _FakeCLI()
    erased: list[bool] = []
    invalidated: list[bool] = []
    tui = object.__new__(TGAgentTUI)
    tui._cli = cli
    tui._resize_repaint_at = 0.0
    tui._resize_repaint_in_progress = False
    tui._erase_prompt_renderer = lambda: erased.append(True)
    tui._invalidate_prompt = lambda: invalidated.append(True)
    tui._is_legacy_windows_console = lambda: legacy_windows
    monkeypatch.setattr(asyncio, "get_running_loop", lambda: _FakeLoop())
    return tui, cli, erased, invalidated


def test_legacy_windows_resize_repaint_uses_direct_terminal_output(monkeypatch) -> None:
    tui, cli, erased, invalidated = _make_tui(monkeypatch, legacy_windows=True)

    tui._repaint_after_resize_if_ready()

    assert erased == [True]
    assert invalidated == [True]
    assert cli.repaint_calls == [
        {
            "preserve_activity": True,
            "direct_terminal_output": True,
        }
    ]


def test_modern_terminal_resize_repaint_keeps_normal_output(monkeypatch) -> None:
    tui, cli, erased, invalidated = _make_tui(monkeypatch, legacy_windows=False)

    tui._repaint_after_resize_if_ready()

    assert erased == []
    assert invalidated == [True]
    assert cli.repaint_calls == []


def test_history_console_restores_primary_after_direct_output() -> None:
    normal = _FakeConsole()
    direct = _FakeConsole()
    console = _HistoryConsole(normal, lambda _args, _kwargs: None)

    with console.use_primary(direct):
        console.print("direct")

    console.print("normal")

    assert direct.printed == ["direct"]
    assert normal.printed == ["normal"]


def test_direct_windows_repaint_clear_uses_cls(monkeypatch) -> None:
    console = _FakeConsole()
    commands: list[str] = []
    cli = object.__new__(TGAgentCLI)
    cli._console = console
    monkeypatch.setattr(app_module.os, "name", "nt")
    monkeypatch.setattr(app_module.os, "system", lambda command: commands.append(command))

    cli._clear_terminal_screen(direct_terminal_output=True)

    assert commands == ["cls"]
    assert console.clear_calls == 0


def test_vscode_terminal_is_not_treated_as_legacy_windows_console(monkeypatch) -> None:
    monkeypatch.setattr(app_module.os, "name", "nt")
    monkeypatch.delenv("WT_SESSION", raising=False)
    monkeypatch.setenv("TERM_PROGRAM", "vscode")

    assert TGAgentTUI._is_legacy_windows_console() is False
