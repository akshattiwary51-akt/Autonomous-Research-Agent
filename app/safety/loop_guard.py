"""Loop-prevention safety checks (Step 11).

Consolidates the hard bounds that must be enforced regardless of what the
LLM decides:
  A. Maximum iteration bound.
  D. Maximum total tool-call bound (a second, independent ceiling — in
     this codebase one iteration currently maps to at most one tool
     attempt, but this is kept as a distinct check so the two bounds stay
     independently configurable, e.g. a very high max_iterations with a
     tighter max_tool_calls).

Tool-call hashing (Step 11.B) lives in `app.tools.base` (tightly coupled
to the tool-call shape); zero-yield tracking (Step 11.C) lives in
`app.agent`/`app.graph.reflection_node` since it's produced as a
byproduct of tool execution. This module is where all of it is *consulted*
before deciding whether to keep going.
"""

from __future__ import annotations

from typing import Optional

from app.config import Settings
from app.state import ResearchState


def should_force_synthesis(
    state: ResearchState, settings: Settings
) -> tuple[bool, Optional[str]]:
    """Hard-bound check consulted before every agent reasoning turn.

    Returns (True, reason) if execution must stop now regardless of what
    the LLM would otherwise choose.
    """
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", settings.max_iterations)
    if iteration >= max_iterations:
        return True, "Maximum iteration bound reached."

    tool_calls_made = len(state.get("tool_call_history") or [])
    if tool_calls_made >= settings.max_tool_calls:
        return True, "Maximum tool-call bound reached."

    return False, None


def reflection_cycle_exhausted(state: ResearchState, settings: Settings) -> bool:
    """Practical cycle detection (Step 11's "cycle detection where
    practical"): if the reflection node has already fired
    `max_reflection_attempts` times this run and the agent is *still*
    hitting the zero-yield threshold again, further reflection is very
    unlikely to help — this is an agent<->reflection oscillation, not
    genuine progress. The router should give up gracefully (force
    synthesis) rather than loop.
    """
    return state.get("reflection_count", 0) >= settings.max_reflection_attempts
