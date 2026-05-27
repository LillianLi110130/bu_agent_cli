from cli.tui_app import TGAgentTUI


def test_agents_reload_suspends_prompt_for_confirmation() -> None:
    assert TGAgentTUI._should_suspend_prompt_for_input("/agents reload") is True


def test_non_interactive_agents_commands_do_not_suspend_prompt() -> None:
    assert TGAgentTUI._should_suspend_prompt_for_input("/agents list") is False
