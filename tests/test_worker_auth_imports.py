from __future__ import annotations

import subprocess
import sys


def test_worker_auth_import_does_not_load_circular_agent_stack() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "import cli.worker.auth"],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_llm_message_import_does_not_load_worker_auth() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import agent_core.llm.messages; "
                "assert 'cli.worker.auth' not in sys.modules, sys.modules.keys()"
            ),
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
