"""Renders `ResearchState` into a compact, structured text summary suitable
for feeding to the LLM as context.

Kept separate from the OpenAI client so it can be unit tested without any
LLM dependency, and reused by other prompts (evidence evaluation,
synthesis) later.
"""

from __future__ import annotations

from app.state import ResearchState

MAX_EVIDENCE_PREVIEW = 8
MAX_SEARCH_HISTORY_PREVIEW = 10


def render_state_summary(state: ResearchState) -> str:
    lines: list[str] = []

    lines.append(f"Research question: {state.get('research_question', '')}")

    sub_questions = state.get("sub_questions") or []
    if sub_questions:
        lines.append("\nSub-questions:")
        for i, sq in enumerate(sub_questions, 1):
            lines.append(f"  {i}. {sq}")

    lines.append(
        f"\nIteration: {state.get('iteration', 0)} / {state.get('max_iterations', 0)}"
    )

    search_history = state.get("search_history") or []
    if search_history:
        lines.append("\nPrior searches (most recent last):")
        for rec in search_history[-MAX_SEARCH_HISTORY_PREVIEW:]:
            status = "ok" if rec.get("success") else f"FAILED ({rec.get('error')})"
            line = (
                f"  - [{rec.get('tool')}] query={rec.get('query')!r} "
                f"results={rec.get('result_count')} "
                f"useful={rec.get('useful_result_count')} status={status}"
            )
            if rec.get("sub_question"):
                line += f" (targeting: {rec['sub_question']!r})"
            if rec.get("rationale"):
                line += f" [rationale: {rec['rationale']}]"
            lines.append(line)

    failed_queries = state.get("failed_queries") or []
    if failed_queries:
        lines.append(f"\nQueries that produced no useful evidence: {failed_queries}")

    evidence = state.get("evidence") or []
    if evidence:
        lines.append(f"\nEvidence collected so far ({len(evidence)} items, showing latest {MAX_EVIDENCE_PREVIEW}):")
        for ev in evidence[-MAX_EVIDENCE_PREVIEW:]:
            lines.append(
                f"  - [{ev.get('relevance')}/{ev.get('confidence')}] "
                f"{ev.get('paper_title')}: {ev.get('claim')}"
            )
    else:
        lines.append("\nNo evidence collected yet.")

    contradictions = state.get("contradictions") or []
    if contradictions:
        lines.append(f"\nContradictions detected ({len(contradictions)}):")
        for c in contradictions:
            lines.append(f"  - {c.get('description')} ({c.get('evidence_a_paper_id')} vs {c.get('evidence_b_paper_id')})")

    zero_yield = state.get("consecutive_zero_yield_count", 0)
    if zero_yield:
        lines.append(
            f"\nWarning: {zero_yield} consecutive searches produced no useful results. "
            "Consider a substantially different query or tool."
        )

    guidance = state.get("reflection_guidance")
    if guidance:
        lines.append("\nReflection guidance (from strategy-change analysis — follow this):")
        lines.append(f"  Diagnosis: {guidance.get('diagnosis')}")
        lines.append(f"  Suggested strategy change: {guidance.get('suggested_strategy_change')}")
        if guidance.get("suggested_tool"):
            lines.append(f"  Suggested tool: {guidance['suggested_tool']}")
        if guidance.get("suggested_query"):
            lines.append(f"  Suggested query: {guidance['suggested_query']}")

    return "\n".join(lines)
