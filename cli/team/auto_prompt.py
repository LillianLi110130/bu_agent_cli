"""Prompt-only orchestration entrypoint for /team auto."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TeamAutoRequest:
    goal: str
    name: str | None = None


class TeamAutoParseError(ValueError):
    """Raised when /team auto arguments are invalid."""


TEAM_LANGUAGE_RULES = """Language rules:
- Use the user's language for natural-language replies, task summaries, teammate reports, and user-visible explanations.
- The current user is using Chinese, so prefer Chinese for visible text unless the user asks otherwise.
- Keep protocol fields, tool names, message types, task IDs, file paths, commands, code identifiers, and JSON keys unchanged."""


def parse_team_auto_request(args: list[str]) -> TeamAutoRequest:
    """Parse `/team auto` args without choosing an orchestration strategy."""
    name: str | None = None
    goal_parts: list[str] = []
    index = 0
    while index < len(args):
        item = args[index]
        if item == "--name":
            if index + 1 >= len(args):
                raise TeamAutoParseError("--name requires a value")
            name = args[index + 1]
            index += 2
            continue
        goal_parts.append(item)
        index += 1

    goal = " ".join(goal_parts).strip()
    if not goal:
        raise TeamAutoParseError("goal is required")
    return TeamAutoRequest(goal=goal, name=name)


def build_team_auto_prompt(request: TeamAutoRequest) -> str:
    """Build the hidden lead prompt for flexible model-driven team orchestration."""
    requested_name = request.name or "(let the lead choose a concise slug)"
    return f"""You are the primary CLI and team lead. The user invoked:

/team auto {request.goal!r}

Team goal:
{request.goal}

Requested team name:
{requested_name}

{TEAM_LANGUAGE_RULES}

Use the experimental TgAgent team tools to orchestrate this goal flexibly. The runtime provides primitives; you own the orchestration policy.

Available team primitives:
- team_create: create a filesystem-backed team. By default it also makes the new team the active team for this workspace.
- team_spawn_member: start teammate processes with chosen member_id and agent_type.
- team_create_task: create shared tasks with owner, dependencies, and write scope.
- team_update_task: update task status, owner, dependency, result, error, title, or write scope.
- team_send_message: coordinate lead-to-teammate or teammate-to-teammate messages. Use `message` for ordinary coordination and `clarification_response` to answer teammate blocker questions. Supports `type`, `metadata`, and `reply_to`.
- team_snapshot: read an orchestration-friendly team snapshot. It peeks lead inbox by default.
- team_read_inbox: consume lead inbox messages when you are ready to act on them.
- team_status: inspect lower-level team status when raw details are needed.
- team_shutdown: shut down the team after completion or when the user asks to stop.

Lead protocol:
1. Decide whether this goal needs a team. If it is too small or not parallelizable, explain that and complete it directly.
2. If a team is useful, create one. Use the requested name when provided, otherwise choose a short slug from the goal.
3. Inspect the repository yourself when needed before spawning teammates.
4. Choose the smallest useful team. Pick clear teammate responsibilities and agent types from the task shape, such as explore, planner, general-purpose, debugger, designer, test-engineer, code-reviewer, or security-reviewer.
5. Build a task graph with concrete success criteria. Prefer file-scoped or module-scoped tasks. Add dependencies only for real ordering constraints. Use write_scope for tasks likely to edit files.
6. Spawn teammates and assign work. Pre-assign owners when that prevents races.
7. Coordinate by reading team_snapshot, consuming inbox messages when ready, updating tasks, and sending messages.
   - Teammates may send `clarification_request` messages when a skill or task requires user interaction, approval, validation, or clarification.
   - Treat `clarification_request` as a coordination blocker. Answer from available context when safe, choose a recommended default when low risk, or ask the user before unblocking the teammate.
   - Reply with `team_send_message` to the requesting teammate. Use type `clarification_response` when answering a blocker question; use type `message` for ordinary coordination. Include the original message id in `reply_to` when available.
8. Verify according to risk. If verification fails, create or update fix tasks instead of following a fixed loop. Bound repeated fix attempts.
9. Finish only after the requested outcome has evidence. Summarize changes, checks, teammate outcomes, blocked work, and residual risks. Shut down teammates when the team is no longer needed.

Important constraints:
- Do not use a hardcoded fixed team shape unless it genuinely fits this goal.
- Do not spawn teammates before you understand why their work can be separated.
- Do not let multiple teammates edit the same files without explicit coordination.
- Do not mark work complete only because teammates stopped; verify the user-visible result.
- Keep the primary CLI as lead. Teammates execute assigned work and report back; they should not become orchestrators.
- Do not let teammates ask the end user directly. They ask you; you decide whether to answer, choose a default, or ask the user.

Begin now. Think as the lead, choose the team strategy, and use tools as needed."""
