"""Tool node -- the Act + Observe step, as a LangGraph node.

Consumes `state["pending_action"]` (set by agent_node) and executes it via
the shared `execute_tool_call` logic from app.agent, so tool execution
semantics (dedup, error handling, state updates) are identical whether
called from the plain-Python loop or the LangGraph graph.
"""

from __future__ import annotations

from typing import Any, Callable

from app.agent import execute_tool_call
from app.llm.base import AgentDecision
from app.safety.circuit_breaker import CircuitBreaker
from app.state import ResearchState
from app.tools.base import ToolRegistry


def build_tool_node(
    registry: ToolRegistry,
    circuit_breaker: CircuitBreaker | None = None,
) -> Callable[[ResearchState], dict[str, Any]]:
    def tool_node(state: ResearchState) -> dict[str, Any]:
        pending = state.get("pending_action") or {}
        decision = AgentDecision(
            action=pending.get("action", "call_tool"),
            tool_name=pending.get("tool_name"),
            tool_args=pending.get("tool_args"),
            sub_question=pending.get("sub_question"),
            rationale_summary=pending.get("rationale_summary"),
        )
        return execute_tool_call(state, registry, decision, circuit_breaker)

    return tool_node
