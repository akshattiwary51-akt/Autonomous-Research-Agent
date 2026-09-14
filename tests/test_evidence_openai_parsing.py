"""Tests for the parsing/filtering logic inside
`OpenAIEvidenceEvaluator`/`OpenAIContradictionDetector` — specifically the
anti-fabrication guarantee (Step 16): any paper_id not present in the
input papers must be dropped, never surfaced as a real citation.

These test the static parsing helpers directly, so no live LLM call is
needed — only the `bind_tools(...).invoke(...)` call itself is untestable
offline, and that's exercised for graceful-failure behavior in the
`no_credentials` tests below (mirroring the pattern already established
for OpenAIReasoningClient / OpenAIQueryDecomposer).
"""

from __future__ import annotations

from app.evidence.openai_evaluator import (
    OpenAIContradictionDetector,
    OpenAIEvidenceEvaluator,
)
from app.state import EvidenceItem
from app.tools.registry import build_default_registry


class TestEvidenceParsingFiltersFabrication:
    def test_drops_evidence_with_unknown_paper_id(self):
        papers_by_id = {"real_id": {"title": "Real Paper", "source": "arxiv", "url": "http://x"}}
        raw_items = [
            {
                "paper_id": "totally_made_up_id",
                "claim": "fabricated claim",
                "supporting_info": "n/a",
                "relevance": "high",
                "confidence": "strong",
            },
            {
                "paper_id": "real_id",
                "claim": "a real claim",
                "supporting_info": "grounded in abstract",
                "relevance": "high",
                "confidence": "strong",
            },
        ]
        result = OpenAIEvidenceEvaluator._parse_evidence_items(
            raw_items, valid_ids={"real_id"}, papers_by_id=papers_by_id,
        )
        assert len(result) == 1
        assert result[0].paper_id == "real_id"

    def test_drops_evidence_with_empty_claim(self):
        papers_by_id = {"p1": {"title": "T", "source": "arxiv", "url": None}}
        raw_items = [{
            "paper_id": "p1", "claim": "   ", "supporting_info": "x",
            "relevance": "high", "confidence": "strong",
        }]
        result = OpenAIEvidenceEvaluator._parse_evidence_items(
            raw_items, valid_ids={"p1"}, papers_by_id=papers_by_id,
        )
        assert result == []

    def test_invalid_relevance_defaults_to_low_not_dropped(self):
        papers_by_id = {"p1": {"title": "T", "source": "arxiv", "url": None}}
        raw_items = [{
            "paper_id": "p1", "claim": "a claim", "supporting_info": "x",
            "relevance": "extremely high (not a real enum value)", "confidence": "strong",
        }]
        result = OpenAIEvidenceEvaluator._parse_evidence_items(
            raw_items, valid_ids={"p1"}, papers_by_id=papers_by_id,
        )
        assert len(result) == 1
        assert result[0].relevance == "low"

    def test_invalid_confidence_defaults_to_weak(self):
        papers_by_id = {"p1": {"title": "T", "source": "arxiv", "url": None}}
        raw_items = [{
            "paper_id": "p1", "claim": "a claim", "supporting_info": "x",
            "relevance": "high", "confidence": "extremely sure",
        }]
        result = OpenAIEvidenceEvaluator._parse_evidence_items(
            raw_items, valid_ids={"p1"}, papers_by_id=papers_by_id,
        )
        assert result[0].confidence == "weak"

    def test_evidence_item_inherits_source_and_url_from_paper(self):
        papers_by_id = {"p1": {"title": "T", "source": "crossref", "url": "https://doi.org/x"}}
        raw_items = [{
            "paper_id": "p1", "claim": "a claim", "supporting_info": "x",
            "relevance": "medium", "confidence": "moderate",
        }]
        result = OpenAIEvidenceEvaluator._parse_evidence_items(
            raw_items, valid_ids={"p1"}, papers_by_id=papers_by_id,
        )
        assert result[0].source == "crossref"
        assert result[0].url == "https://doi.org/x"

    def test_evaluate_papers_returns_empty_without_llm_call_for_empty_papers_list(self):
        evaluator = OpenAIEvidenceEvaluator()
        result = evaluator.evaluate_papers([], "research question", [])
        assert result == []


class TestContradictionParsingFiltersFabrication:
    def test_drops_contradiction_referencing_unknown_paper_id(self):
        raw = [
            {
                "description": "fabricated conflict",
                "evidence_a_paper_id": "unknown_id",
                "evidence_b_paper_id": "p2",
                "explanation": "n/a",
            },
            {
                "description": "real conflict",
                "evidence_a_paper_id": "p1",
                "evidence_b_paper_id": "p2",
                "explanation": "genuinely conflicting claims",
            },
        ]
        result = OpenAIContradictionDetector._parse_contradictions(raw, valid_ids={"p1", "p2"})
        assert len(result) == 1
        assert result[0].evidence_a_paper_id == "p1"

    def test_drops_contradiction_with_missing_explanation(self):
        raw = [{
            "description": "conflict", "evidence_a_paper_id": "p1",
            "evidence_b_paper_id": "p2", "explanation": "",
        }]
        result = OpenAIContradictionDetector._parse_contradictions(raw, valid_ids={"p1", "p2"})
        assert result == []

    def test_detect_returns_empty_without_llm_call_when_either_side_empty(self):
        detector = OpenAIContradictionDetector()
        ev = EvidenceItem(paper_id="p1", paper_title="T", claim="c", supporting_info="s",
                            relevance="high", confidence="strong")
        assert detector.detect([], [ev]) == []
        assert detector.detect([ev], []) == []


class TestGracefulFailureWithoutCredentials:
    def test_evaluate_papers_fails_gracefully_without_api_key(self):
        evaluator = OpenAIEvidenceEvaluator()
        papers = [{"paper_id": "p1", "title": "T", "abstract": "abs", "source": "arxiv", "url": None}]
        # No OPENAI_API_KEY configured in this environment -> should return
        # [] rather than raise, per Step 17.
        result = evaluator.evaluate_papers(papers, "research question", [])
        assert result == []

    def test_detect_fails_gracefully_without_api_key(self):
        detector = OpenAIContradictionDetector()
        ev1 = EvidenceItem(paper_id="p1", paper_title="T1", claim="c1", supporting_info="s",
                             relevance="high", confidence="strong")
        ev2 = EvidenceItem(paper_id="p2", paper_title="T2", claim="c2", supporting_info="s",
                             relevance="high", confidence="strong")
        result = detector.detect([ev1], [ev2])
        assert result == []


class TestToolRegistryUnaffected:
    """Sanity check that Phase 8 changes didn't disturb the tool registry
    from earlier phases."""

    def test_registry_still_has_three_tools(self):
        registry = build_default_registry()
        assert len(registry.names()) == 3
