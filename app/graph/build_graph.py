r"""Builds the research agent's LangGraph `StateGraph` (Step 12).

    START -> planner -> agent -> [router] -> tool -> evidence -> agent (loop)
                                          \-> reflection --------> agent (loop)
                                          \-> synthesis -> END

Termination is guaranteed by two independent mechanisms:
  1. `decide_action` (app.agent) forces `pending_action.action = "synthesize"`
     once `iteration >= max_iterations` (Step 11.A), regardless of what the
     LLM would otherwise choose.
  2. Every edge that loops back to `agent` (tool_node, reflection_node)
     only does so after incrementing `iteration` — so even in the worst
     case of the LLM always choosing to call tools, the loop is bounded.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from app.graph.agent_node import build_agent_node
from app.graph.evidence_node import build_evidence_node
from app.graph.planner_node import build_planner_node
from app.graph.reflection_node import build_reflection_node
from app.graph.router import route_after_agent
from app.graph.synthesis_node import build_synthesis_node
from app.graph.tool_node import build_tool_node
from app.evidence.evaluator import (
    ContradictionDetector,
    EvidenceEvaluator,
    NoOpContradictionDetector,
    NoOpEvidenceEvaluator,
)
from app.fulltext.fetcher import FullTextFetcher, NoOpFullTextFetcher
from app.llm.base import ReasoningClient
from app.llm.decomposition import NoOpDecomposer, QueryDecomposer
from app.llm.reflection import NoOpReflectionAdvisor, ReflectionAdvisor
from app.report.gap_discovery import GapDiscoverer, NoOpGapDiscoverer
from app.report.narrative import NarrativeWriter, NoOpNarrativeWriter
from app.safety.circuit_breaker import CircuitBreaker
from app.state import ResearchState
from app.tools.base import ToolRegistry


def build_research_graph(
    registry: ToolRegistry,
    reasoning_client: ReasoningClient,
    decomposer: QueryDecomposer | None = None,
    evidence_evaluator: EvidenceEvaluator | None = None,
    contradiction_detector: ContradictionDetector | None = None,
    reflection_advisor: ReflectionAdvisor | None = None,
    gap_discoverer: GapDiscoverer | None = None,
    narrative_writer: NarrativeWriter | None = None,
    fulltext_fetcher: FullTextFetcher | None = None,
    circuit_breaker: CircuitBreaker | None = None,
    checkpointer=None,
    enable_hitl: bool = False,
):
    """Compile the full research agent graph.

    `decomposer` defaults to `NoOpDecomposer` (no decomposition, agent
    reasons over the raw question); `evidence_evaluator` defaults to
    `NoOpEvidenceEvaluator` and `contradiction_detector` to
    `NoOpContradictionDetector` (no structured evidence extracted);
    `reflection_advisor` defaults to `NoOpReflectionAdvisor` (generic
    fallback guidance, no LLM call); `gap_discoverer` defaults to
    `NoOpGapDiscoverer` and `narrative_writer` to `NoOpNarrativeWriter`
    (deterministic, template-based report sections, no LLM call);
    `fulltext_fetcher` defaults to `NoOpFullTextFetcher` (evidence stays
    grounded in abstracts only) — so existing callers/tests that don't
    pass these keep working unchanged.

    `circuit_breaker` is the one exception to "defaults to off": it
    defaults to a live `CircuitBreaker` instance rather than a NoOp, since
    it only ever prevents wasted iterations on a tool that just signaled
    it's rate-limited (HTTP 429) — no added LLM cost or latency, so
    there's no reason to make it opt-in. Pass an explicit instance only if
    you want a non-default cooldown period.

    `checkpointer` is left as an explicit optional parameter so callers can
    pass in a `MemorySaver` / `SqliteSaver` from `app.checkpointing` (Step
    13) for persistence/resume.

    `enable_hitl` (Step 14, off by default): when True, the compiled graph
    pauses (`interrupt_before=["tool"]`) immediately before every tool
    execution, so a human can approve/reject it — e.g. via a CLI prompt or
    an external review step — before any external API call is made. The
    run resumes exactly where it paused by invoking the graph again with
    the same `thread_id` and no new input. Requires a checkpointer, since
    the paused state must be persisted to resume later.
    """
    decomposer = decomposer or NoOpDecomposer()
    evidence_evaluator = evidence_evaluator or NoOpEvidenceEvaluator()
    contradiction_detector = contradiction_detector or NoOpContradictionDetector()
    reflection_advisor = reflection_advisor or NoOpReflectionAdvisor()
    gap_discoverer = gap_discoverer or NoOpGapDiscoverer()
    narrative_writer = narrative_writer or NoOpNarrativeWriter()
    fulltext_fetcher = fulltext_fetcher or NoOpFullTextFetcher()
    # Unlike the other components (which default to NoOp — off — since
    # they cost extra LLM calls/latency), the circuit breaker defaults to
    # ON: it only ever prevents wasted iterations on a tool that just
    # signaled it's rate-limited, with zero added cost or latency.
    circuit_breaker = circuit_breaker or CircuitBreaker()

    graph = StateGraph(ResearchState)

    graph.add_node("planner", build_planner_node(decomposer))
    graph.add_node("agent", build_agent_node(registry, reasoning_client, circuit_breaker))
    graph.add_node("tool", build_tool_node(registry, circuit_breaker))
    graph.add_node("evidence_eval", build_evidence_node(evidence_evaluator, contradiction_detector, fulltext_fetcher))
    graph.add_node("reflection", build_reflection_node(reflection_advisor, registry.names()))
    graph.add_node("synthesis", build_synthesis_node(gap_discoverer, narrative_writer))

    graph.set_entry_point("planner")
    graph.add_edge("planner", "agent")

    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "tool": "tool",
            "reflection": "reflection",
            "synthesis": "synthesis",
            "end": "synthesis",  # fail-safe path also lands in synthesis
        },
    )

    graph.add_edge("tool", "evidence_eval")
    graph.add_edge("evidence_eval", "agent")
    graph.add_edge("reflection", "agent")
    graph.add_edge("synthesis", END)

    compile_kwargs: dict[str, Any] = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer
    if enable_hitl:
        # Step 14: pause before every tool execution so a human can
        # approve/reject it before any (potentially large-scale or
        # sensitive) external API call is made. Requires a checkpointer —
        # LangGraph persists the paused state so the run can be resumed
        # after approval, possibly in a different process.
        if checkpointer is None:
            raise ValueError(
                "enable_hitl=True requires a checkpointer (HITL pauses "
                "execution and resumes later, which needs persisted state)."
            )
        compile_kwargs["interrupt_before"] = ["tool"]

    return graph.compile(**compile_kwargs)
