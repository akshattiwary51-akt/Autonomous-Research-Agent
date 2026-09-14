"""Structured research state for the ReAct research agent.

Rather than relying on raw chat history alone, the agent's working memory
is an explicit, typed `ResearchState`. This is what every graph node reads
from and writes back to.

We use a `TypedDict` (LangGraph's preferred state shape) with `Annotated`
reducers on fields that multiple nodes append to across iterations
(`operator.add` for list concatenation). Scalar fields (iteration counters,
status, final_report) are plain — LangGraph's default merge behavior is
"last write wins" for those, which is what we want.
"""

from __future__ import annotations

import operator
from dataclasses import asdict, dataclass
from typing import Annotated, Any, Literal, Optional, TypedDict

from app.tools.base import Paper

# --------------------------------------------------------------------------
# Supporting record types
# --------------------------------------------------------------------------


@dataclass
class SearchRecord:
    """One entry in the search history (Step 3 / Step 18)."""

    iteration: int
    tool: str
    query: str
    result_count: int
    useful_result_count: int
    success: bool
    error: Optional[str] = None
    sub_question: Optional[str] = None
    rationale: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceItem:
    """A single piece of structured evidence extracted from a paper
    (Step 9) — never a raw dumped abstract.
    """

    paper_id: str
    paper_title: str
    claim: str
    supporting_info: str
    relevance: Literal["high", "medium", "low", "irrelevant"]
    confidence: Literal["strong", "moderate", "weak"]
    related_sub_question: Optional[str] = None
    source: Optional[str] = None
    url: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Contradiction:
    """A detected conflict between two pieces of evidence."""

    description: str
    evidence_a_paper_id: str
    evidence_b_paper_id: str
    explanation: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ResearchGap:
    """An evidence-backed potential research gap (Step 10).

    `is_inference` must always be True unless a paper *explicitly* states
    the gap — per Step 10 / Step 16, inferred gaps must be clearly labeled
    as agent synthesis, never presented as an established fact.
    """

    statement: str
    supporting_evidence_paper_ids: list[str]
    significance: str
    is_inference: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToolCallRecord:
    """Record of a single tool invocation, used for dedup + observability."""

    iteration: int
    tool_name: str
    args: dict[str, Any]
    call_hash: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# Reducers
# --------------------------------------------------------------------------


def keep_max(a: int, b: int) -> int:
    """Reducer for counters that should only ever increase."""
    return max(a, b)


# --------------------------------------------------------------------------
# Core state
# --------------------------------------------------------------------------


class ResearchState(TypedDict, total=False):
    """The full working memory of a research run.

    `total=False` so nodes can return partial updates (LangGraph merges
    them into the running state) without needing every key present.
    """

    # --- Identity / input ---
    research_question: str

    # --- Conversation (kept for LLM context; never treated as the source
    # of truth for structured data — that lives in the typed fields below) ---
    messages: Annotated[list[dict[str, Any]], operator.add]

    # --- Iteration control (Step 3, Step 11.A) ---
    iteration: int
    max_iterations: int

    # --- Planning (Step 7) ---
    sub_questions: list[str]

    # --- Search / tool tracking (Step 3, Step 8, Step 11.B) ---
    search_history: Annotated[list[dict[str, Any]], operator.add]
    failed_queries: Annotated[list[str], operator.add]
    tool_call_history: Annotated[list[dict[str, Any]], operator.add]
    tool_call_hashes: Annotated[list[str], operator.add]

    # --- Raw + structured results (Step 4, Step 9) ---
    retrieved_papers: Annotated[list[dict[str, Any]], operator.add]
    # Papers fetched by the MOST RECENT tool call only (overwritten each
    # call, not accumulated) — lets evidence_node evaluate just the new
    # batch instead of re-scoring the entire accumulated pile every round.
    last_tool_papers: list[dict[str, Any]]
    evidence: Annotated[list[dict[str, Any]], operator.add]
    contradictions: Annotated[list[dict[str, Any]], operator.add]

    # --- Loop-prevention bookkeeping (Step 11.C) ---
    useful_results_count: int
    consecutive_zero_yield_count: int

    # --- Reflection (Step 6 safe metadata, never raw CoT) ---
    scratchpad: Annotated[list[dict[str, Any]], operator.add]

    # --- Graph orchestration bookkeeping (Phase 6) ---
    # The agent node's most recent decision, consumed by the router and the
    # tool node. Plain dict (not AgentDecision) to stay JSON-serializable
    # for checkpointing. Last-write-wins (no reducer) since only one
    # decision is ever "pending" at a time.
    pending_action: Optional[dict[str, Any]]

    # --- Reflection bookkeeping (Phase 9 / Step 11.C) ---
    # How many times the reflection node has fired this run — accumulates
    # via operator.add so the router can detect a reflect<->agent cycle
    # and force synthesis rather than loop indefinitely even if the LLM
    # keeps disregarding reflection guidance.
    reflection_count: Annotated[int, operator.add]
    # The most recent strategy-change guidance from the reflection node,
    # surfaced to the agent's next reasoning turn via the prompt. Plain
    # dict, last-write-wins (superseded each time reflection fires).
    reflection_guidance: Optional[dict[str, Any]]

    # --- Synthesis output (Step 10, Step 15) ---
    research_gaps: Annotated[list[dict[str, Any]], operator.add]
    final_report: Optional[str]

    # --- Status / control flow ---
    status: Literal[
        "planning",
        "researching",
        "reflecting",
        "synthesizing",
        "done",
        "error",
    ]
    termination_reason: Optional[str]
    error: Optional[str]


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------


def create_initial_state(
    research_question: str,
    max_iterations: int = 6,
) -> ResearchState:
    """Build a fresh `ResearchState` for a new research run."""
    if not research_question or not research_question.strip():
        raise ValueError("research_question must be a non-empty string.")

    return ResearchState(
        research_question=research_question.strip(),
        messages=[],
        iteration=0,
        max_iterations=max_iterations,
        sub_questions=[],
        search_history=[],
        failed_queries=[],
        tool_call_history=[],
        tool_call_hashes=[],
        retrieved_papers=[],
        last_tool_papers=[],
        evidence=[],
        contradictions=[],
        useful_results_count=0,
        consecutive_zero_yield_count=0,
        scratchpad=[],
        research_gaps=[],
        final_report=None,
        status="planning",
        termination_reason=None,
        error=None,
        pending_action=None,
        reflection_count=0,
        reflection_guidance=None,
    )


def paper_to_state_dict(paper: Paper) -> dict[str, Any]:
    """Helper: convert a `Paper` dataclass into the plain dict form stored
    in `ResearchState["retrieved_papers"]` (TypedDict/LangGraph state must
    stay JSON-serializable for checkpointing)."""
    return paper.as_dict()
