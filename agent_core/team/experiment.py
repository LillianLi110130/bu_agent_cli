"""Feature gate for the experimental agent team runtime."""

from __future__ import annotations

import os

TEAM_EXPERIMENT_ENV = "TG_AGENT_TEAM_EXPERIMENT"
_TRUTHY = {"1", "true", "yes", "on"}


def is_team_experiment_enabled() -> bool:
    return os.getenv(TEAM_EXPERIMENT_ENV, "").strip().lower() in _TRUTHY


def team_experiment_disabled_message() -> str:
    return (
        "Agent team experimental feature is disabled. "
        f"Set {TEAM_EXPERIMENT_ENV}=1 to enable it."
    )
