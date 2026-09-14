"""Tests for Step 10 (gap discovery) and Step 15/16 (report synthesis,
citation traceability, anti-fabrication)."""

from __future__ import annotations

import responses

from app.graph.build_graph import build_research_graph
from app.graph.synthesis_node import build_synthesis_node
from app.llm.base import AgentDecision, ReasoningClient
from app.report.builder import build_report
from app.report.gap_discovery import GapDiscoverer, NoOpGapDiscoverer
from app.report.narrative import NarrativeSections, NarrativeWriter, NoOpNarrativeWriter
from app.report.openai_gap_discovery import OpenAIGapDiscoverer
from app.report.openai_narrative import guard_section
from app.state import Contradiction, EvidenceItem, ResearchGap, create_initial_state
from app.tools.arxiv_tool import ARXIV_API_URL
from app.tools.registry import build_default_registry

SAMPLE_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.12345v1</id>
    <title>Multimodal RAG for Scientific Documents</title>
    <summary>We show multimodal RAG struggles with figures and tables.</summary>
    <published>2024-01-20T00:00:00Z</published>
    <author><name>Jane Doe</name></author>
  </entry>
</feed>"""


def make_evidence(paper_id="p1", **overrides) -> EvidenceItem:
    defaults = dict(
        paper_id=paper_id, paper_title="Test Paper", claim="Test claim",
        supporting_info="supporting text", relevance="high", confidence="strong",
        source="arxiv", url="http://arxiv.org/abs/p1",
    )
    defaults.update(overrides)
    return EvidenceItem(**defaults)


class FakeGapDiscoverer(GapDiscoverer):
    def __init__(self, gaps: list[ResearchGap]):
        self._gaps = gaps
        self.calls = 0

    def discover_gaps(self, evidence, contradictions, research_question):
        self.calls += 1
        return self._gaps


class FakeNarrativeWriter(NarrativeWriter):
    def __init__(self, sections: NarrativeSections):
        self._sections = sections

    def write(self, research_question, evidence, contradictions, gaps, iterations_run, max_iterations):
        return self._sections


# --------------------------------------------------------------------------
# Gap discovery anti-fabrication (parsing logic, no live LLM)
# --------------------------------------------------------------------------


class TestGapDiscoveryParsing:
    def test_drops_gap_with_no_valid_supporting_ids(self):
        raw_gaps = [{
            "statement": "A gap not backed by real evidence",
            "supporting_evidence_paper_ids": ["made_up_id"],
            "significance": "would matter if real",
        }]
        result = OpenAIGapDiscoverer._parse_gaps(raw_gaps, valid_ids={"p1", "p2"})
        assert result == []

    def test_keeps_gap_with_at_least_one_valid_id_drops_invalid_ones(self):
        raw_gaps = [{
            "statement": "Real gap",
            "supporting_evidence_paper_ids": ["p1", "made_up_id"],
            "significance": "matters",
        }]
        result = OpenAIGapDiscoverer._parse_gaps(raw_gaps, valid_ids={"p1", "p2"})
        assert len(result) == 1
        assert result[0].supporting_evidence_paper_ids == ["p1"]

    def test_gaps_always_marked_as_inference(self):
        raw_gaps = [{
            "statement": "Gap", "supporting_evidence_paper_ids": ["p1"], "significance": "x",
        }]
        result = OpenAIGapDiscoverer._parse_gaps(raw_gaps, valid_ids={"p1"})
        assert result[0].is_inference is True

    def test_discover_gaps_returns_empty_without_llm_call_for_no_evidence(self):
        discoverer = OpenAIGapDiscoverer()
        result = discoverer.discover_gaps([], [], "research question")
        assert result == []

    def test_discover_gaps_fails_gracefully_without_api_key(self):
        discoverer = OpenAIGapDiscoverer()
        ev = make_evidence()
        result = discoverer.discover_gaps([ev], [], "research question")
        assert result == []


# --------------------------------------------------------------------------
# Narrative fabrication guard
# --------------------------------------------------------------------------


class TestNarrativeFabricationGuard:
    def test_passes_through_clean_text(self):
        text = "The evidence suggests limited generalization across datasets."
        result = guard_section(text, known_ids={"2401.12345"}, section_name="x", fallback="FALLBACK")
        assert result == text

    def test_rejects_text_with_unknown_arxiv_id(self):
        text = "As shown in paper 2401.12345, and further confirmed in 9999.99999, results vary."
        result = guard_section(text, known_ids={"2401.12345"}, section_name="x", fallback="FALLBACK")
        assert result == "FALLBACK"

    def test_accepts_text_referencing_only_known_arxiv_id(self):
        text = "As shown in paper 2401.12345, results vary across settings."
        result = guard_section(text, known_ids={"2401.12345"}, section_name="x", fallback="FALLBACK")
        assert result == text

    def test_rejects_text_with_unknown_doi(self):
        text = "See https://doi.org/10.1234/fake.made.up for details."
        result = guard_section(text, known_ids={"2401.12345"}, section_name="x", fallback="FALLBACK")
        assert result == "FALLBACK"


# --------------------------------------------------------------------------
# Report builder structure (Step 15)
# --------------------------------------------------------------------------


class TestReportBuilderStructure:
    def test_report_contains_all_required_sections(self):
        state = create_initial_state("What are the limitations of multimodal RAG?", max_iterations=6)
        state["retrieved_papers"] = [{
            "paper_id": "p1", "title": "Test Paper", "authors": ["A. Author"],
            "published": "2024-01-01", "url": "http://arxiv.org/abs/p1", "source": "arxiv",
        }]
        evidence = [make_evidence()]
        contradictions: list[Contradiction] = []
        gaps = [ResearchGap(
            statement="No large-scale benchmark exists",
            supporting_evidence_paper_ids=["p1"],
            significance="blocks generalization claims",
        )]
        narrative = NarrativeSections(
            executive_summary="Summary.", comparative_analysis="Analysis.",
            limitations="Limitations.", conclusion="Conclusion.",
        )

        report = build_report(state, evidence, contradictions, gaps, narrative)

        for heading in [
            "# Research Question", "# Executive Summary", "# Research Methodology",
            "# Key Findings", "# Comparative Analysis",
            "# Contradictions / Conflicting Findings", "# Limitations in Existing Research",
            "# Research Gaps", "# Conclusion", "# References",
        ]:
            assert heading in report, f"missing section: {heading}"

    def test_key_findings_reflects_evidence_claim_and_relevance(self):
        state = create_initial_state("q")
        evidence = [make_evidence(claim="Struggles with figures", relevance="high")]
        narrative = NarrativeSections("s", "c", "l", "conc")
        report = build_report(state, evidence, [], [], narrative)
        assert "Struggles with figures" in report
        assert "high" in report

    def test_research_gaps_section_labels_inference(self):
        state = create_initial_state("q")
        gaps = [ResearchGap(
            statement="Gap statement", supporting_evidence_paper_ids=["p1"],
            significance="matters", is_inference=True,
        )]
        narrative = NarrativeSections("s", "c", "l", "conc")
        report = build_report(state, [], [], gaps, narrative)
        assert "Inferred by the research agent" in report

    def test_references_only_cite_papers_with_evidence(self):
        state = create_initial_state("q")
        state["retrieved_papers"] = [
            {"paper_id": "p1", "title": "Cited Paper", "authors": [], "published": "2024", "url": "http://x", "source": "arxiv"},
            {"paper_id": "p2", "title": "Uncited Paper", "authors": [], "published": "2024", "url": "http://y", "source": "arxiv"},
        ]
        evidence = [make_evidence(paper_id="p1")]  # only p1 has evidence
        narrative = NarrativeSections("s", "c", "l", "conc")
        report = build_report(state, evidence, [], [], narrative)
        assert "Cited Paper" in report
        assert "Uncited Paper" not in report

    def test_empty_evidence_produces_honest_no_data_sections(self):
        state = create_initial_state("q")
        narrative = NoOpNarrativeWriter().write("q", [], [], [], 0, 6)
        report = build_report(state, [], [], [], narrative)
        assert "No evidence was collected" in report
        assert "No sources were cited" in report


# --------------------------------------------------------------------------
# NoOpNarrativeWriter deterministic content
# --------------------------------------------------------------------------


class TestNoOpNarrativeWriter:
    def test_executive_summary_reflects_actual_counts(self):
        evidence = [make_evidence(), make_evidence(paper_id="p2", relevance="medium")]
        writer = NoOpNarrativeWriter()
        sections = writer.write("q", evidence, [], [], iterations_run=3, max_iterations=6)
        assert "2 structured" in sections.executive_summary
        assert "3 of 6" in sections.executive_summary

    def test_limitations_mentions_max_iterations_reached(self):
        writer = NoOpNarrativeWriter()
        sections = writer.write("q", [], [], [], iterations_run=6, max_iterations=6)
        assert "6-iteration bound" in sections.limitations


# --------------------------------------------------------------------------
# synthesis_node orchestration
# --------------------------------------------------------------------------


class TestSynthesisNode:
    def test_orchestrates_gap_discovery_and_narrative_and_builds_report(self):
        gap = ResearchGap(
            statement="Underexplored at scale", supporting_evidence_paper_ids=["p1"],
            significance="limits real-world applicability",
        )
        narrative = NarrativeSections("Summary text", "Analysis text", "Limits text", "Conclusion text")
        node = build_synthesis_node(FakeGapDiscoverer([gap]), FakeNarrativeWriter(narrative))

        state = create_initial_state("test question", max_iterations=6)
        state["evidence"] = [make_evidence().as_dict()]
        state["pending_action"] = {"action": "synthesize", "rationale_summary": "enough evidence"}

        update = node(state)

        assert update["status"] == "done"
        assert update["termination_reason"] == "enough evidence"
        assert len(update["research_gaps"]) == 1
        assert "Summary text" in update["final_report"]
        assert "Underexplored at scale" in update["final_report"]

    def test_error_path_produces_error_report_not_full_synthesis(self):
        node = build_synthesis_node(NoOpGapDiscoverer(), NoOpNarrativeWriter())
        state = create_initial_state("test question")
        state["pending_action"] = {"action": "error", "rationale_summary": "LLM call failed: timeout"}

        update = node(state)

        assert update["status"] == "done"
        assert update["error"] == "LLM call failed: timeout"
        assert "Error" in update["final_report"]
        assert "research_gaps" not in update  # gap discovery skipped on error path


# --------------------------------------------------------------------------
# End-to-end: full graph produces a complete, well-formed report
# --------------------------------------------------------------------------


class TestSynthesisIntegratesWithGraph:
    @responses.activate
    def test_full_graph_run_produces_complete_report(self):
        responses.add(responses.GET, ARXIV_API_URL, body=SAMPLE_ATOM, status=200)
        registry = build_default_registry()

        class ScriptedClient(ReasoningClient):
            def __init__(self):
                self.n = 0

            def decide_next_action(self, state, tool_schemas):
                self.n += 1
                if self.n == 1:
                    return AgentDecision(
                        action="call_tool", tool_name="arxiv_search",
                        tool_args={"query": "multimodal RAG"},
                    )
                return AgentDecision(action="synthesize", rationale_summary="enough evidence")

        gap = ResearchGap(
            statement="No large-scale benchmark", supporting_evidence_paper_ids=["2401.12345v1"],
            significance="limits confidence in generalization",
        )
        app = build_research_graph(
            registry, ScriptedClient(),
            gap_discoverer=FakeGapDiscoverer([gap]),
        )

        result = app.invoke(
            create_initial_state("What are the limitations of multimodal RAG?", max_iterations=6),
            config={"recursion_limit": 50},
        )

        assert result["status"] == "done"
        report = result["final_report"]
        assert "# Research Question" in report
        assert "# References" in report
        assert len(result["research_gaps"]) == 1
