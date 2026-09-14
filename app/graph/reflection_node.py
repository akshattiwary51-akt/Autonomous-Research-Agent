"""Reflection node (Step 11.C) — full implementation.

Diagnoses why recent searches were unproductive and proposes a genuinely
different next step via `ReflectionAdvisor`, then surfaces that guidance
to the agent's next reasoning turn through `state["reflection_guidance"]`
(picked up by `render_state_summary` in the agent's prompt).

Also guarantees forward progress and eventual termination:
- increments `iteration` (counts as effort spent, contributing to the
  hard `max_iterations` bound).
- resets `consecutive_zero_yield_count` to 0 so the very next search gets
  a fair chance under the new strategy.
- increments `reflection_count`, which the router uses for cycle
  detection (Step 11's "cycle detection where practical") — if reflection
  keeps firing without resolving the problem, the router eventually gives
  up into synthesis rather than oscillating forever.
"""

from __future__ import annotations

from typing import Any, Callable

from app.llm.reflection import ReflectionAdvisor
from app.logging_utils import get_logger
from app.state import ResearchState

logger = get_logger(__name__)


def build_reflection_node(
    advisor: ReflectionAdvisor,
    available_tools: list[str],
) -> Callable[[ResearchState], dict[str, Any]]:
    def reflection_node(state: ResearchState) -> dict[str, Any]:
        iteration = state.get("iteration", 0) + 1
        zero_yield = state.get("consecutive_zero_yield_count", 0)

        guidance = advisor.reflect(state, available_tools)

        logger.info(
            "Reflection triggered at iteration %s after %s unproductive searches. "
            "Diagnosis: %s",
            iteration, zero_yield, guidance.diagnosis,
        )

        return {
            "iteration": iteration,
            "consecutive_zero_yield_count": 0,
            "reflection_count": 1,
            "status": "reflecting",
            "reflection_guidance": guidance.as_dict(),
            "scratchpad": [
                {
                    "stage": "reflection",
                    "unproductive_searches": zero_yield,
                    "diagnosis": guidance.diagnosis,
                    "suggested_strategy_change": guidance.suggested_strategy_change,
                    "suggested_tool": guidance.suggested_tool,
                    "suggested_query": guidance.suggested_query,
                }
            ],
        }

    return reflection_node
