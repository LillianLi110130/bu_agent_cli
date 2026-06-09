from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_core.mcp.config import load_mcp_config, parse_mcp_config, set_mcp_server_disabled


def test_parse_mcp_config_reads_claude_style_mcp_servers() -> None:
    config = parse_mcp_config(
        {
            "mcpServers": {
                "codegraph": {
                    "type": "stdio",
                    "command": "codegraph",
                    "args": ["serve", "--mcp"],
                    "env": {"FOO": 1},
                }
            }
        }
    )

    server = config.servers["codegraph"]
    assert server.type == "stdio"
    assert server.command == "codegraph"
    assert server.args == ["serve", "--mcp"]
    assert server.env == {"FOO": "1"}
    assert server.disabled is False


def test_parse_mcp_config_reads_disabled_server() -> None:
    config = parse_mcp_config(
        {
            "mcpServers": {
                "codegraph": {
                    "command": "codegraph",
                    "disabled": True,
                }
            }
        }
    )

    assert config.servers["codegraph"].disabled is True


def test_parse_mcp_config_rejects_unsupported_transport() -> None:
    with pytest.raises(ValueError, match="type must be 'stdio'"):
        parse_mcp_config({"mcpServers": {"remote": {"type": "http", "command": "x"}}})


def test_load_mcp_config_reads_user_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    settings_dir = home / ".tg_agent"
    settings_dir.mkdir(parents=True)
    (settings_dir / "settings.json").write_text(
        json.dumps({"mcpServers": {"codegraph": {"command": "codegraph"}}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(home))

    config = load_mcp_config()

    assert config.servers["codegraph"].command == "codegraph"


def test_load_mcp_config_reads_project_config_with_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    settings_dir = home / ".tg_agent"
    settings_dir.mkdir(parents=True)
    (settings_dir / "settings.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "codegraph": {
                        "command": "user-codegraph",
                        "args": ["serve", "--mcp"],
                    },
                    "user_only": {"command": "user-only"},
                }
            }
        ),
        encoding="utf-8",
    )
    workspace = tmp_path / "workspace"
    project_dir = workspace / ".tg_agent"
    project_dir.mkdir(parents=True)
    (project_dir / "mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "codegraph": {
                        "command": "project-codegraph",
                        "args": ["serve", "--mcp"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(home))

    config = load_mcp_config(workspace)

    assert config.servers["codegraph"].command == "project-codegraph"
    assert config.servers["codegraph"].source == "project"
    assert config.servers["user_only"].source == "user"
    assert config.project_config_path == str(project_dir / "mcp.json")


def test_set_mcp_server_disabled_updates_project_source(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    project_dir = workspace / ".tg_agent"
    project_dir.mkdir(parents=True)
    config_path = project_dir / "mcp.json"
    config_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "codegraph": {
                        "command": "codegraph",
                        "args": ["serve", "--mcp"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    changed = set_mcp_server_disabled(
        workspace_root=workspace,
        server_name="codegraph",
        source="project",
        disabled=True,
    )

    data = json.loads(config_path.read_text(encoding="utf-8"))
    assert changed == config_path
    assert data["mcpServers"]["codegraph"]["disabled"] is True


def test_set_mcp_server_disabled_updates_user_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    settings_dir = home / ".tg_agent"
    settings_dir.mkdir(parents=True)
    settings_path = settings_dir / "settings.json"
    settings_path.write_text(
        json.dumps({"mcpServers": {"codegraph": {"command": "codegraph"}}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(home))

    changed = set_mcp_server_disabled(
        workspace_root=tmp_path,
        server_name="codegraph",
        source="user",
        disabled=True,
    )

    data = json.loads(settings_path.read_text(encoding="utf-8"))
    assert changed == settings_path
    assert data["mcpServers"]["codegraph"]["disabled"] is True
