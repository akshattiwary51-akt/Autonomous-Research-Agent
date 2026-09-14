"""Assembles the final research report (Step 15).

Pure function -- no LLM calls here. Structured sections (Methodology, Key
Findings, Contradictions, Research Gaps, References) are built directly
from `ResearchState` + already-computed `gaps`, so they carry zero
fabrication risk by construction. Narrative sections (Executive Summary,
Comparative Analysis, Limitations, Conclusion) are passed in as an
already-generated `NarrativeSections` (see app.report.narrative) -- this
module doesn't care whether that came from the deterministic fallback or
a grounded LLM writer.
"""

from __future__ import annotations

from typing import Any

from app.report.narrative import NarrativeSections
from app.state import Contradiction, EvidenceItem, ResearchGap, ResearchState

_RELEVANCE_ORDER = {"high": 0, "medium": 1, "low": 2, "irrelevant": 3}


def build_report(
    state: ResearchState,
    evidence: list[EvidenceItem],
    contradictions: list[Contradiction],
    gaps: list[ResearchGap],
    narrative: NarrativeSections,
) -> str:
    sections = [
        _research_question_section(state),
        _executive_summary_section(narrative),
        _methodology_section(state),
        _key_findings_section(evidence),
        _comparative_analysis_section(narrative),
        _contradictions_section(contradictions),
        _limitations_section(narrative),
        _research_gaps_section(gaps),
        _conclusion_section(narrative),
        _references_section(evidence, state),
    ]
    return "\n\n".join(s for s in sections if s)


def _research_question_section(state: ResearchState) -> str:
    return f"# Research Question\n\n{state.get('research_question', '')}"


def _executive_summary_section(narrative: NarrativeSections) -> str:
    return f"# Executive Summary\n\n{narrative.executive_summary}"


def _methodology_section(state: ResearchState) -> str:
    tool_history = state.get("tool_call_history") or []
    search_history = state.get("search_history") or []

    tools_used = sorted({rec.get("tool_name") for rec in tool_history if rec.get("tool_name")})
    queries = [rec.get("query") for rec in search_history if rec.get("query")]
    seen: set[str] = set()
    unique_queries: list[str] = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            unique_queries.append(q)

    lines = ["# Research Methodology", ""]
    lines.append(f"- Tools used: {', '.join(tools_used) if tools_used else 'none'}")
    lines.append(f"- Iterations run: {state.get('iteration', 0)} / {state.get('max_iterations', 0)}")
    lines.append(f"- Total search attempts: {len(search_history)}")
    sub_questions = state.get("sub_questions") or []
    if sub_questions:
        lines.append(f"- Question decomposed into {len(sub_questions)} sub-question(s):")
        for sq in sub_questions:
            lines.append(f"  - {sq}")
    if unique_queries:
        lines.append("- Major queries:")
        for q in unique_queries[:15]:
            lines.append(f"  - \"{q}\"")
    reflection_count = state.get("reflection_count", 0)
    if reflection_count:
        lines.append(f"- Search strategy was revised {reflection_count} time(s) via reflection.")

    return "\n".join(lines)


def _key_findings_section(evidence: list[EvidenceItem]) -> str:
    if not evidence:
        return "# Key Findings\n\nNo evidence was collected during this run."

    ranked = sorted(evidence, key=lambda e: _RELEVANCE_ORDER.get(e.relevance, 4))
    substantive = [e for e in ranked if e.relevance != "irrelevant"]
    if not substantive:
        return "# Key Findings\n\nNo evidence rated above 'irrelevant' was collected."

    lines = ["# Key Findings", ""]
    for e in substantive:
        lines.append(f"### {e.claim}")
        lines.append(f"- Supporting paper: {e.paper_title}" + (f" ({e.url})" if e.url else ""))
        lines.append(f"- Relevance: {e.relevance} | Confidence: {e.confidence}")
        if e.supporting_info:
            lines.append(f"- Supporting detail: {e.supporting_info}")
        if e.related_sub_question:
            lines.append(f"- Addresses sub-question: {e.related_sub_question}")
        lines.append("")

    return "\n".join(lines).rstrip()


def _comparative_analysis_section(narrative: NarrativeSections) -> str:
    return f"# Comparative Analysis\n\n{narrative.comparative_analysis}"


def _contradictions_section(contradictions: list[Contradiction]) -> str:
    if not contradictions:
        return "# Contradictions / Conflicting Findings\n\nNo contradictions were detected among the collected evidence."

    lines = ["# Contradictions / Conflicting Findings", ""]
    for c in contradictions:
        lines.append(f"- **{c.description}**")
        lines.append(f"  - {c.explanation}")
        lines.append(f"  - Sources: {c.evidence_a_paper_id} vs {c.evidence_b_paper_id}")
    return "\n".join(lines)


def _limitations_section(narrative: NarrativeSections) -> str:
    return f"# Limitations in Existing Research\n\n{narrative.limitations}"


def _research_gaps_section(gaps: list[ResearchGap]) -> str:
    if not gaps:
        return "# Research Gaps\n\nNo evidence-backed research gaps were identified in this run."

    lines = ["# Research Gaps", ""]
    for g in gaps:
        inference_note = (
            " *(Inferred by the research agent from patterns across the "
            "evidence below -- not a claim directly stated by any single "
            "source.)*" if g.is_inference else ""
        )
        lines.append(f"### {g.statement}{inference_note}")
        lines.append(f"- Significance: {g.significance}")
        lines.append(f"- Supporting evidence (paper ids): {', '.join(g.supporting_evidence_paper_ids)}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _conclusion_section(narrative: NarrativeSections) -> str:
    return f"# Conclusion\n\n{narrative.conclusion}"


def _references_section(evidence: list[EvidenceItem], state: ResearchState) -> str:
    if not evidence:
        return "# References\n\nNo sources were cited in this report."

    papers_by_id: dict[str, dict[str, Any]] = {
        p.get("paper_id"): p for p in (state.get("retrieved_papers") or []) if p.get("paper_id")
    }

    cited_ids: list[str] = []
    seen: set[str] = set()
    for e in evidence:
        if e.paper_id not in seen:
            seen.add(e.paper_id)
            cited_ids.append(e.paper_id)

    lines = ["# References", ""]
    for pid in cited_ids:
        paper = papers_by_id.get(pid)
        if paper:
            authors = ", ".join(paper.get("authors") or []) or "Unknown authors"
            published = paper.get("published") or "n.d."
            url = paper.get("url") or ""
            lines.append(f"- {paper.get('title', pid)} -- {authors} ({published}). {url}")
        else:
            lines.append(f"- [paper_id={pid}] (metadata unavailable)")
    return "\n".join(lines)
