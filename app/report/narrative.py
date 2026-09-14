"""Interface for the narrative (prose) sections of the final report:
Executive Summary, Comparative Analysis, Limitations, Conclusion.

`NoOpNarrativeWriter` is not a placeholder to be discarded later — it's a
permanent, deterministic fallback with zero fabrication risk (built
directly from counts/structure in state, no free-text LLM generation) that
`OpenAINarrativeWriter` (Phase 10's grounded LLM writer) falls back to
per-section whenever its own citation-fabrication guard trips.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.state import Contradiction, EvidenceItem, ResearchGap


@dataclass
class NarrativeSections:
    executive_summary: str
    comparative_analysis: str
    limitations: str
    conclusion: str


class NarrativeWriter(ABC):
    @abstractmethod
    def write(
        self,
        research_question: str,
        evidence: list[EvidenceItem],
        contradictions: list[Contradiction],
        gaps: list[ResearchGap],
        iterations_run: int,
        max_iterations: int,
    ) -> NarrativeSections:
        """Must never raise for expected failure modes — fall back to a
        deterministic summary instead (Step 17)."""
        raise NotImplementedError


class NoOpNarrativeWriter(NarrativeWriter):
    """Deterministic, template-based narrative — always factually accurate
    because it only ever states counts and structure already present in
    state, never free-generated claims."""

    def write(
        self,
        research_question: str,
        evidence: list[EvidenceItem],
        contradictions: list[Contradiction],
        gaps: list[ResearchGap],
        iterations_run: int,
        max_iterations: int,
    ) -> NarrativeSections:
        n_papers = len({e.paper_id for e in evidence})
        high_medium = [e for e in evidence if e.relevance in ("high", "medium")]
        weak = [e for e in evidence if e.confidence == "weak"]

        executive_summary = (
            f"This report investigates: {research_question} "
            f"Across {iterations_run} of {max_iterations} available search "
            f"iterations, the agent collected {len(evidence)} structured "
            f"evidence item(s) from {n_papers} paper(s), of which "
            f"{len(high_medium)} were rated high or medium relevance. "
            f"{len(contradictions)} contradiction(s) and {len(gaps)} "
            f"potential research gap(s) were identified."
        )

        if evidence:
            by_relevance: dict[str, int] = {}
            for e in evidence:
                by_relevance[e.relevance] = by_relevance.get(e.relevance, 0) + 1
            breakdown = ", ".join(f"{v} {k}" for k, v in sorted(by_relevance.items()))
            comparative_analysis = (
                f"Evidence relevance breakdown: {breakdown}. "
                f"{'No direct contradictions were detected among the collected evidence.' if not contradictions else f'{len(contradictions)} contradiction(s) were detected — see the Contradictions section below.'}"
            )
        else:
            comparative_analysis = "No evidence was collected, so no comparative analysis is possible."

        limitations_parts = [
            "This report is based on paper titles and abstracts retrieved "
            "via automated academic search APIs, not full paper text.",
        ]
        if iterations_run >= max_iterations:
            limitations_parts.append(
                f"The research run reached its {max_iterations}-iteration bound; "
                "additional searching may have surfaced further evidence."
            )
        if weak:
            limitations_parts.append(
                f"{len(weak)} evidence item(s) were rated 'weak' confidence "
                "(the abstract's statement was hedged or ambiguous)."
            )
        limitations = " ".join(limitations_parts)

        if gaps:
            conclusion = (
                f"Based on {len(evidence)} evidence item(s) from {n_papers} paper(s), "
                f"this investigation identified {len(gaps)} potential research "
                "gap(s), detailed below. These are the agent's inferences from "
                "patterns in the collected evidence, not claims made directly "
                "by any single paper."
            )
        elif evidence:
            conclusion = (
                f"Based on {len(evidence)} evidence item(s) from {n_papers} paper(s), "
                "no clear evidence-backed research gaps were identified in this run."
            )
        else:
            conclusion = "No evidence was collected during this run, so no conclusions can be drawn."

        return NarrativeSections(
            executive_summary=executive_summary,
            comparative_analysis=comparative_analysis,
            limitations=limitations,
            conclusion=conclusion,
        )
