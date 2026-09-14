"""Conditional routing for the research agent graph (Step 12).

Routing decisions are driven entirely by structured state (`pending_action`,
`status`, `consecutive_zero_yield_count`, `reflection_count`) — never by
re-invoking the LLM — so routing itself is deterministic and cheap to test.
"""

from __future__ import annotations

from typing import Literal

from app.config import get_settings
from app.safety.loop_guard import reflection_cycle_exhausted
from app.state import ResearchState

RouteAfterAgent = Literal["tool", "reflection", "synthesis", "end"]
RouteAfterLoopStep = Literal["agent"]


def route_after_agent(state: ResearchState) -> RouteAfterAgent:
    """Decide where to go after the agent node has reasoned about its next
    action (Step 12's Router).

    - "error" decisions go straight to synthesis, which will produce a
      best-effort/error report rather than crashing (Step 17).
    - "synthesize" decisions go to synthesis.
    - "call_tool" decisions normally go to the tool node, UNLESS the
      zero-yield threshold has been hit, in which case we force a
      reflection pass first (Step 11.C) instead of burning another
      iteration on what is likely to be another unproductive search —
      UNLESS reflection has already been tried `max_reflection_attempts`
      times without resolving the problem, in which case we give up
      gracefully into synthesis rather than cycle indefinitely between
      agent and reflection (Step 11's cycle-detection requirement).
    """
    pending = state.get("pending_action") or {}
    action = pending.get("action")

    if action == "error":
        return "synthesis"

    if action == "synthesize":
        return "synthesis"

    if action == "call_tool":
        settings = get_settings()
        zero_yield = state.get("consecutive_zero_yield_count", 0)
        if zero_yield >= settings.zero_yield_reflection_threshold:
            if reflection_cycle_exhausted(state, settings):
                return "synthesis"
            return "reflection"
        return "tool"

    # Unexpected/missing pending_action — fail safe into synthesis rather
    # than looping forever on an undefined action.
    return "synthesis"
