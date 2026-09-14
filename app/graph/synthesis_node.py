"""Synthesis node (Step 15) -- full implementation.

Orchestrates:
  1. Gap discovery (Step 10) -- evidence-grounded, labeled as inference.
  2. Narrative writing (Executive Summary, Comparative Analysis,
     Limitations, Conclusion) -- grounded, with a fabrication guard.
  3. Deterministic report assembly (Step 15's full structure).

Reads `pending_action` (set by agent_node) to distinguish a normal
"agent declared sufficient evidence" / "hit max iterations" termination
from an LLM-error termination, since `state["status"]` isn't populated
until this node runs.
"""

from __future__ import annotations

from typing import Any, Callable

from app.evidence.models import evidence_items_from_dicts
from app.logging_utils import get_logger
from app.report.builder import build_report
from app.report.gap_discovery import GapDiscoverer
from app.report.narrative import NarrativeWriter
from app.state import Contradiction, ResearchState

logger = get_logger(__name__)


def _contradictions_from_dicts(items: list[dict[str, Any]]) -> list[Contradiction]:
    result = []
    for d in items:
        try:
            result.append(Contradiction(**d))
        except TypeError:
            continue
    return result


def build_synthesis_node(
    gap_discoverer: GapDiscoverer,
    narrative_writer: NarrativeWriter,
) -> Callable[[ResearchState], dict[str, Any]]:
    def synthesis_node(state: ResearchState) -> dict[str, Any]:
        pending = state.get("pending_action") or {}
        action = pending.get("action")
        rationale = pending.get("rationale_summary")

        if action == "error":
            error_message = rationale or "unknown error"
            evidence = evidence_items_from_dicts(state.get("evidence") or [])
            report = (
                "# Research Report (Incomplete -- Error)\n\n"
                f"Research question: {state.get('research_question', '')}\n\n"
                f"The research run stopped early due to an error: {error_message}\n\n"
                f"Papers retrieved before the error: {len(state.get('retrieved_papers', []))}\n"
                f"Evidence collected before the error: {len(evidence)}\n"
            )
            return {
                "final_report": report,
                "status": "done",
                "error": error_message,
                "termination_reason": rationale or "llm_error",
            }

        research_question = state.get("research_question", "")
        evidence = evidence_items_from_dicts(state.get("evidence") or [])
        contradictions = _contradictions_from_dicts(state.get("contradictions") or [])

        gaps = gap_discoverer.discover_gaps(evidence, contradictions, research_question)

        narrative = narrative_writer.write(
            research_question=research_question,
            evidence=evidence,
            contradictions=contradictions,
            gaps=gaps,
            iterations_run=state.get("iteration", 0),
            max_iterations=state.get("max_iterations", 0),
        )

        report = build_report(state, evidence, contradictions, gaps, narrative)

        logger.info(
            "Synthesis complete: %d evidence item(s), %d contradiction(s), %d gap(s).",
            len(evidence), len(contradictions), len(gaps),
        )

        return {
            "final_report": report,
            "research_gaps": [g.as_dict() for g in gaps],
            "status": "done",
            "termination_reason": rationale or "synthesis_complete",
        }

    return synthesis_node
