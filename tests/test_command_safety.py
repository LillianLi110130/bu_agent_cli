from __future__ import annotations

import pytest

from agent_core.agent.command_safety import (
    check_dangerous_command,
    normalize_command,
)


def test_command_safety_normalizes_ansi_null_and_width_chars() -> None:
    command = "\x1b[31mＧＩＴ\x00   RESET   --HARD\x1b[0m"

    assert normalize_command(command) == "git reset --hard"


def test_command_safety_classifies_block_ask_and_allow() -> None:
    assert check_dangerous_command("rm -rf /").action == "block"
    assert check_dangerous_command("git reset --hard").action == "block"
    assert check_dangerous_command("rm -rf build").action == "ask"
    assert check_dangerous_command("pytest -q").action == "allow"


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf .tg_agent",
        "rm -r .tg_agent",
        "rm --recursive .tg_agent",
        "rm -f -r .tg_agent",
        "rm -rf path/.tg_agent",
        "rm -rf ~/.tg_agent",
        "rm -rf /home/me/.tg_agent",
        "rmdir /s /q .tg_agent",
        "rd /s .tg_agent",
        r"rmdir /s C:\Users\me\.tg_agent",
        "Remove-Item -Recurse .tg_agent",
        "Remove-Item .tg_agent -Recurse -Force",
        "rm -Recurse .tg_agent",
        "rmdir -Recurse .tg_agent",
    ],
)
def test_command_safety_blocks_tg_agent_directory_deletion(command: str) -> None:
    result = check_dangerous_command(command)

    assert result.action == "block"
    assert any(
        finding.rule_id.startswith("delete_tg_agent_dir")
        for finding in result.blocked_findings
    )


def test_command_safety_does_not_hard_block_tg_agent_child_cleanup() -> None:
    result = check_dangerous_command("rm -rf .tg_agent/shell_tasks/old")

    assert result.action == "ask"
    assert not any(
        finding.rule_id.startswith("delete_tg_agent_dir")
        for finding in result.blocked_findings
    )


@pytest.mark.parametrize(
    "command",
    [
        "rm -f -r build",
        "rm --recursive build",
        "rm -rf -- build",
    ],
)
def test_command_safety_asks_for_rm_recursive_variants(command: str) -> None:
    assert check_dangerous_command(command).action == "ask"


@pytest.mark.parametrize(
    "command",
    [
        "Remove-Item -Recurse build",
        "Remove-Item build -Recurse",
        "rm -Recurse build",
        "rmdir -Recurse build",
        "rmdir /s build",
    ],
)
def test_command_safety_asks_for_powershell_recursive_delete_without_force(
    command: str,
) -> None:
    assert check_dangerous_command(command).action == "ask"


@pytest.mark.parametrize(
    "command",
    [
        "git push -f",
        "git push origin main -f",
        "git push --force",
        "git push --force-with-lease",
    ],
)
def test_command_safety_asks_for_explicit_git_force_push(command: str) -> None:
    assert check_dangerous_command(command).action == "ask"


def test_command_safety_allows_git_push_follow_tags() -> None:
    assert check_dangerous_command("git push --follow-tags").action == "allow"
