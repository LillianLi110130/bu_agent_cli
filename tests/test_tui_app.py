import cli.tui_app as tui_app_module
from cli.tui_app import TGAgentTUI


def test_agents_reload_suspends_prompt_for_confirmation() -> None:
    assert TGAgentTUI._should_suspend_prompt_for_input("/agents reload") is True


def test_non_interactive_agents_commands_do_not_suspend_prompt() -> None:
    assert TGAgentTUI._should_suspend_prompt_for_input("/agents list") is False


def test_prompt_message_renders_activity_elapsed_time(monkeypatch) -> None:
    tui = TGAgentTUI.__new__(TGAgentTUI)
    tui._prompt_spinner_index = 0
    started_at = 100.0

    class _FakeCLI:
        def _get_terminal_activity_status(self):
            return "思考中"

        def _get_terminal_activity_started_at(self):
            return started_at

        def _terminal_approval_prompt_lines(self):
            return []

    tui._cli = _FakeCLI()
    monkeypatch.setattr(tui_app_module.time, "monotonic", lambda: 105.9)

    rendered = tui._render_prompt_message()

    assert "- 思考中 (5s)..." in rendered
