from __future__ import annotations

import asyncio
import importlib
import json
from pathlib import Path

import pytest

from tools import ALL_TOOLS
from tools.browser_harness import browser_harness
from tools.sandbox import SandboxContext, get_sandbox_context

browser_harness_module = importlib.import_module("tools.browser_harness")


@pytest.fixture
def workspace_root() -> Path:
    return Path.cwd().resolve()


@pytest.mark.asyncio
async def test_browser_harness_passes_script_to_stdin_without_shell(
    monkeypatch: pytest.MonkeyPatch,
    workspace_root: Path,
) -> None:
    captured: dict[str, object] = {}

    class FakeProcess:
        returncode = 0

        def __init__(self, args: list[str], **kwargs: object) -> None:
            captured["args"] = args
            captured["kwargs"] = kwargs

        def communicate(self, input: str | None = None) -> tuple[str, str]:
            captured["input"] = input
            return "page ok", ""

    monkeypatch.setattr(browser_harness_module.subprocess, "Popen", FakeProcess)
    ctx = SandboxContext.create(workspace_root)
    script = 'new_tab("https://example.com?a=1&enterpriseDomain=cmb")\nprint(page_info())\n'

    result = await browser_harness.execute(
        script=script,
        timeout=5,
        _overrides={get_sandbox_context: lambda: ctx},
    )

    payload = json.loads(result)
    assert payload["ok"] is True
    assert payload["stdout"] == "page ok"
    assert captured["args"] == ["browser-harness"]
    assert captured["input"] == script
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["shell"] is False
    assert kwargs["stdin"] == browser_harness_module.subprocess.PIPE
    assert kwargs["cwd"] == str(workspace_root)


@pytest.mark.asyncio
async def test_browser_harness_reports_timeout(
    monkeypatch: pytest.MonkeyPatch,
    workspace_root: Path,
) -> None:
    async def fake_run_browser_harness(*, script: str, cwd: str, timeout: int):
        raise asyncio.TimeoutError

    monkeypatch.setattr(browser_harness_module, "_run_browser_harness", fake_run_browser_harness)
    ctx = SandboxContext.create(workspace_root)

    result = await browser_harness.execute(
        script="print(page_info())\n",
        timeout=1,
        _overrides={get_sandbox_context: lambda: ctx},
    )

    payload = json.loads(result)
    assert payload["ok"] is False
    assert payload["timed_out"] is True
    assert payload["returncode"] is None
    assert "timed out after 1s" in payload["stderr"]


@pytest.mark.asyncio
async def test_browser_harness_reports_missing_command(
    monkeypatch: pytest.MonkeyPatch,
    workspace_root: Path,
) -> None:
    def fake_popen(*args: object, **kwargs: object) -> object:
        raise FileNotFoundError

    monkeypatch.setattr(browser_harness_module.subprocess, "Popen", fake_popen)
    ctx = SandboxContext.create(workspace_root)

    result = await browser_harness.execute(
        script="print(page_info())\n",
        timeout=5,
        _overrides={get_sandbox_context: lambda: ctx},
    )

    assert result == "Error: browser-harness not found on PATH"


def test_browser_harness_is_registered() -> None:
    assert browser_harness in ALL_TOOLS
    assert browser_harness.name == "browser_harness"
