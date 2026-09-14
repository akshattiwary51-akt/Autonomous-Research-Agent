"""Agent node -- the Reason step of the ReAct loop, as a LangGraph node.

Only *decides* the next action here; execution happens in tool_node so the
router can inspect the decision (and possibly redirect to reflection)
before any tool actually runs.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable

from app.agent import decide_action
from app.llm.base import ReasoningClient
from app.safety.circuit_breaker import CircuitBreaker
from app.state import ResearchState
from app.tools.base import ToolRegistry


def build_agent_node(
    registry: ToolRegistry,
    reasoning_client: ReasoningClient,
    circuit_breaker: CircuitBreaker | None = None,
) -> Callable[[ResearchState], dict[str, Any]]:
    """Returns a LangGraph-compatible node function bound to the given
    tool registry and reasoning client (dependency injection via closure,
    since LangGraph nodes only receive `state`)."""

    def agent_node(state: ResearchState) -> dict[str, Any]:
        decision = decide_action(state, registry, reasoning_client, circuit_breaker)
        return {"pending_action": asdict(decision)}

    return agent_node
