"""Core ReAct agent step.

This module contains the framework-agnostic Reason -> Act -> Observe logic
(Step 6). It depends only on `ResearchState`, `ToolRegistry`, and the
`ReasoningClient` interface — no LangGraph import here — so it can be unit
tested in isolation with a fake LLM client and dummy tools.

Phase 6 wraps `run_agent_step` (plus a thin tool-execution wrapper) as
LangGraph nodes; the logic itself does not change.
"""

from __future__ import annotations

from typing import Any

from app.config import get_settings
from app.llm.base import AgentDecision, ReasoningClient
from app.logging_utils import IterationLog, get_logger, log_iteration
from app.safety.circuit_breaker import CircuitBreaker
from app.safety.loop_guard import should_force_synthesis
from app.state import ResearchState, SearchRecord, ToolCallRecord, paper_to_state_dict
from app.tools.base import ToolRegistry, hash_tool_call

logger = get_logger(__name__)


def is_duplicate_call(state: ResearchState, tool_name: str, args: dict[str, Any]) -> bool:
    """Basic Step 11.B duplicate detection: has this exact (tool, normalized
    args) combination already been executed this run?

    Note: full reflection-driven strategy change (Step 11.C) is layered on
    top of this in the safety module (Phase 9) — this function only answers
    "have we seen this exact call before", not "should we change strategy".
    """
    call_hash = hash_tool_call(tool_name, args)
    return call_hash in (state.get("tool_call_hashes") or [])


def decide_action(
    state: ResearchState,
    registry: ToolRegistry,
    reasoning_client: ReasoningClient,
    circuit_breaker: CircuitBreaker | None = None,
) -> AgentDecision:
    """Reason step: ask the LLM what to do next, but enforce hard bounds
    (Step 11.A max iterations, plus the max-tool-calls bound) before even
    consulting the LLM.

    When a `circuit_breaker` is provided, any tool currently rate-limited
    (open) is excluded from the schemas offered to the LLM this turn, so
    provider failover happens immediately rather than waiting for
    reflection to notice the pattern across iterations.
    """
    force_stop, reason = should_force_synthesis(state, get_settings())
    if force_stop:
        return AgentDecision(action="synthesize", rationale_summary=reason)

    if circuit_breaker is not None:
        available = circuit_breaker.available_tools(registry.names())
        schemas = registry.schemas_for(available)
    else:
        schemas = registry.schemas()

    return reasoning_client.decide_next_action(state, schemas)


def execute_tool_call(
    state: ResearchState,
    registry: ToolRegistry,
    decision: AgentDecision,
    circuit_breaker: CircuitBreaker | None = None,
) -> dict[str, Any]:
    """Act + Observe step: execute the chosen tool (or handle a bad/duplicate
    choice gracefully) and return a partial state update.

    Never raises for expected failure modes (unknown tool name, duplicate
    call, tool execution failure) per Step 17 — those are recorded in state
    instead so the agent can reason about them on the next turn.
    """
    iteration = state.get("iteration", 0) + 1
    tool_name = decision.tool_name or ""
    tool_args = dict(decision.tool_args or {})
    query = str(tool_args.get("query", ""))

    # --- Unknown tool guard ---
    if tool_name not in registry:
        logger.warning("Agent selected unknown tool: %r", tool_name)
        record = SearchRecord(
            iteration=iteration, tool=tool_name, query=query,
            result_count=0, useful_result_count=0, success=False,
            error=f"Unknown tool '{tool_name}'.",
        )
        log_iteration(IterationLog(
            iteration=iteration, tool=tool_name, query=query,
            status="tool_error", termination_reason=None,
        ))
        return {
            "iteration": iteration,
            "search_history": [record.as_dict()],
            "failed_queries": [query] if query else [],
            "last_tool_papers": [],
            "consecutive_zero_yield_count": state.get("consecutive_zero_yield_count", 0) + 1,
            "status": "researching",
        }

    # --- Duplicate call guard (Step 11.B) ---
    call_hash = hash_tool_call(tool_name, tool_args)
    if is_duplicate_call(state, tool_name, tool_args):
        logger.info("Skipping duplicate tool call: %s(%r)", tool_name, tool_args)
        record = SearchRecord(
            iteration=iteration, tool=tool_name, query=query,
            result_count=0, useful_result_count=0, success=False,
            error="Duplicate call skipped (identical tool+args already executed).",
        )
        log_iteration(IterationLog(
            iteration=iteration, tool=tool_name, query=query,
            status="duplicate_skipped",
        ))
        return {
            "iteration": iteration,
            "search_history": [record.as_dict()],
            "last_tool_papers": [],
            "consecutive_zero_yield_count": state.get("consecutive_zero_yield_count", 0) + 1,
            "status": "researching",
        }

    # --- Execute ---
    tool = registry.get(tool_name)
    result = tool.run(**tool_args)

    if circuit_breaker is not None:
        circuit_breaker.record_result(tool_name, result.success, result.rate_limited)

    call_record = ToolCallRecord(
        iteration=iteration, tool_name=tool_name, args=tool_args, call_hash=call_hash,
    )

    useful_count = result.result_count if result.success else 0
    search_record = SearchRecord(
        iteration=iteration, tool=tool_name, query=query,
        result_count=result.result_count, useful_result_count=useful_count,
        success=result.success, error=result.error,
        sub_question=decision.sub_question, rationale=decision.rationale_summary,
    )

    log_iteration(IterationLog(
        iteration=iteration, tool=tool_name, query=query,
        result_count=result.result_count, useful_result_count=useful_count,
        status="ok" if result.success else "tool_error",
    ))

    zero_yield = state.get("consecutive_zero_yield_count", 0)
    zero_yield = 0 if useful_count > 0 else zero_yield + 1

    update: dict[str, Any] = {
        "iteration": iteration,
        "search_history": [search_record.as_dict()],
        "tool_call_history": [call_record.as_dict()],
        "tool_call_hashes": [call_hash],
        "useful_results_count": state.get("useful_results_count", 0) + useful_count,
        "consecutive_zero_yield_count": zero_yield,
        "status": "researching",
        "scratchpad": [
            {
                "stage": "tool_execution",
                "objective": decision.sub_question,
                "tool": tool_name,
                "query": query,
                "rationale": decision.rationale_summary,
                "observation_summary": (
                    f"{result.result_count} result(s), {useful_count} usable"
                    if result.success else f"failed: {result.error}"
                ),
            }
        ],
    }

    if result.success:
        paper_dicts = [paper_to_state_dict(p) for p in result.papers]
        update["retrieved_papers"] = paper_dicts
        update["last_tool_papers"] = paper_dicts
        if result.result_count == 0 and query:
            update["failed_queries"] = [query]
    else:
        update["last_tool_papers"] = []
        update["failed_queries"] = [query] if query else []

    return update


def run_agent_step(
    state: ResearchState,
    registry: ToolRegistry,
    reasoning_client: ReasoningClient,
) -> dict[str, Any]:
    """One full Reason -> Act -> Observe cycle. Returns a partial state
    update (LangGraph-node-compatible)."""
    decision = decide_action(state, registry, reasoning_client)

    if decision.action == "error":
        logger.warning("Agent reasoning returned an error: %s", decision.rationale_summary)
        return {
            "status": "error",
            "error": decision.rationale_summary,
            "termination_reason": "llm_error",
        }

    if decision.action == "synthesize":
        return {
            "status": "synthesizing",
            "termination_reason": decision.rationale_summary or "agent_declared_sufficient_evidence",
        }

    return execute_tool_call(state, registry, decision)
