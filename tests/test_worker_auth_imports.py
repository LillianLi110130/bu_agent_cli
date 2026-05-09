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
