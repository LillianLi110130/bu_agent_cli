from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    payload = json.load(sys.stdin)
    working_dir = Path(payload["working_dir"])
    target = payload["args_text"].strip() or "."

    print(
        f"Summarize the workspace at {working_dir} with focus on '{target}'. "
        "List the most relevant files, key risks, and recommended next steps."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
