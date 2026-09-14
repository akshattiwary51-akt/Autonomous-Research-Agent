"""Evidence node -- Tool Node -> Evidence Store -> Agent (Step 2's target
architecture; Step 9's evidence evaluation).

Consumes `state["last_tool_papers"]` (the papers from the MOST RECENT tool
call only, set by tool_node) so it evaluates just the new batch each round
instead of re-scoring the entire accumulated pile every iteration.

Optionally fetches each paper's full text (via `FullTextFetcher`) before
evidence extraction -- a real accuracy upgrade over abstract-only
grounding. Defaults to `NoOpFullTextFetcher`, which changes nothing about
prior behavior unless explicitly enabled.
"""

from __future__ import annotations

from typing import Any, Callable

from app.evidence.evaluator import ContradictionDetector, EvidenceEvaluator
from app.evidence.models import evidence_items_from_dicts
from app.fulltext.fetcher import FullTextFetcher, NoOpFullTextFetcher
from app.logging_utils import get_logger
from app.state import ResearchState

logger = get_logger(__name__)


def _augment_with_fulltext(
    papers: list[dict[str, Any]], fetcher: FullTextFetcher
) -> tuple[list[dict[str, Any]], int]:
    """Attaches `full_text`/`full_text_truncated` to each paper dict where
    a fetch succeeds; leaves the paper unchanged (abstract-only) otherwise.
    Never raises -- `FullTextFetcher.fetch` is contractually failure-safe."""
    augmented: list[dict[str, Any]] = []
    fetched_count = 0
    for paper in papers:
        result = fetcher.fetch(paper)
        if result.text:
            paper = {**paper, "full_text": result.text, "full_text_truncated": result.truncated}
            fetched_count += 1
        else:
            if result.error:
                logger.info(
                    "No full text for %r: %s (falling back to abstract)",
                    paper.get("title"), result.error,
                )
        augmented.append(paper)
    return augmented, fetched_count


def build_evidence_node(
    evaluator: EvidenceEvaluator,
    contradiction_detector: ContradictionDetector,
    fulltext_fetcher: FullTextFetcher | None = None,
) -> Callable[[ResearchState], dict[str, Any]]:
    fulltext_fetcher = fulltext_fetcher or NoOpFullTextFetcher()

    def evidence_node(state: ResearchState) -> dict[str, Any]:
        new_papers = state.get("last_tool_papers") or []

        if not new_papers:
            # Nothing new this round (duplicate call, failed tool, or
            # unknown tool) -- nothing to evaluate.
            return {"status": "researching"}

        research_question = state.get("research_question", "")
        sub_questions = state.get("sub_questions") or []

        augmented_papers, fulltext_fetched_count = _augment_with_fulltext(
            new_papers, fulltext_fetcher
        )

        new_evidence = evaluator.evaluate_papers(augmented_papers, research_question, sub_questions)

        update: dict[str, Any] = {"status": "researching"}

        contradictions = []
        if new_evidence:
            existing_evidence = evidence_items_from_dicts(state.get("evidence") or [])
            contradictions = contradiction_detector.detect(new_evidence, existing_evidence)

            update["evidence"] = [e.as_dict() for e in new_evidence]

        if contradictions:
            update["contradictions"] = [c.as_dict() for c in contradictions]

        relevant_count = sum(1 for e in new_evidence if e.relevance in ("high", "medium"))
        logger.info(
            "Evidence node: %d papers (%d with full text) -> %d evidence items "
            "(%d high/medium relevance), %d contradictions.",
            len(new_papers), fulltext_fetched_count, len(new_evidence),
            relevant_count, len(contradictions),
        )

        update["scratchpad"] = [{
            "stage": "evidence_evaluation",
            "papers_evaluated": len(new_papers),
            "full_text_fetched": fulltext_fetched_count,
            "evidence_extracted": len(new_evidence),
            "high_or_medium_relevance": relevant_count,
            "contradictions_found": len(contradictions),
        }]

        return update

    return evidence_node
