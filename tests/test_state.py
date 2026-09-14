"""Tests covering Step 19 item 10 (state transitions) plus supporting
dataclasses in app/state.py."""

from __future__ import annotations

import pytest
from langgraph.graph import END, StateGraph

from app.state import (
    Contradiction,
    EvidenceItem,
    ResearchGap,
    ResearchState,
    SearchRecord,
    ToolCallRecord,
    create_initial_state,
    paper_to_state_dict,
)
from app.tools.base import Paper


class TestCreateInitialState:
    def test_sets_expected_defaults(self):
        state = create_initial_state("What are the limitations of X?", max_iterations=4)
        assert state["research_question"] == "What are the limitations of X?"
        assert state["iteration"] == 0
        assert state["max_iterations"] == 4
        assert state["status"] == "planning"
        assert state["final_report"] is None
        assert state["error"] is None
        assert state["pending_action"] is None
        for list_field in (
            "messages", "sub_questions", "search_history", "failed_queries",
            "tool_call_history", "tool_call_hashes", "retrieved_papers",
            "evidence", "contradictions", "scratchpad", "research_gaps",
        ):
            assert state[list_field] == []

    def test_strips_whitespace_from_question(self):
        state = create_initial_state("  padded question  ")
        assert state["research_question"] == "padded question"

    def test_rejects_empty_question(self):
        with pytest.raises(ValueError):
            create_initial_state("")

    def test_rejects_whitespace_only_question(self):
        with pytest.raises(ValueError):
            create_initial_state("     ")


class TestSupportingDataclasses:
    def test_search_record_round_trips_to_dict(self):
        rec = SearchRecord(
            iteration=1, tool="arxiv_search", query="rag",
            result_count=5, useful_result_count=3, success=True,
        )
        d = rec.as_dict()
        assert d["tool"] == "arxiv_search"
        assert d["error"] is None

    def test_evidence_item_requires_relevance_and_confidence_literals(self):
        ev = EvidenceItem(
            paper_id="p1", paper_title="Title", claim="claim text",
            supporting_info="info", relevance="high", confidence="strong",
        )
        assert ev.as_dict()["relevance"] == "high"

    def test_research_gap_defaults_to_inference(self):
        gap = ResearchGap(
            statement="No large-scale eval exists",
            supporting_evidence_paper_ids=["p1"],
            significance="blocks generalization claims",
        )
        assert gap.is_inference is True

    def test_contradiction_dict_shape(self):
        c = Contradiction(
            description="conflicting claims about X",
            evidence_a_paper_id="p1",
            evidence_b_paper_id="p2",
            explanation="paper1 says A, paper2 says not-A",
        )
        d = c.as_dict()
        assert d["evidence_a_paper_id"] == "p1"

    def test_tool_call_record_dict_shape(self):
        rec = ToolCallRecord(
            iteration=1, tool_name="arxiv_search",
            args={"query": "rag"}, call_hash="deadbeef",
        )
        assert rec.as_dict()["call_hash"] == "deadbeef"

    def test_paper_to_state_dict_is_json_serializable_shape(self):
        paper = Paper(
            title="T", source="arxiv", paper_id="123",
            authors=["A"], abstract="abs", published="2024-01-01",
            url="http://arxiv.org/abs/123",
        )
        d = paper_to_state_dict(paper)
        assert d == {
            "title": "T", "source": "arxiv", "paper_id": "123",
            "url": "http://arxiv.org/abs/123", "authors": ["A"],
            "abstract": "abs", "published": "2024-01-01",
            "open_access_pdf_url": None,
        }


class TestStateTransitionsViaLangGraph:
    """Verifies the Annotated[list, operator.add] reducers actually merge
    state correctly across multiple graph nodes (Step 19 item 10), and that
    scalar fields use last-write-wins semantics."""

    def test_list_fields_accumulate_across_nodes(self):
        def node_a(state: ResearchState) -> dict:
            return {
                "search_history": [{"iteration": 1, "tool": "arxiv_search"}],
                "iteration": 1,
            }

        def node_b(state: ResearchState) -> dict:
            return {
                "search_history": [{"iteration": 2, "tool": "semantic_scholar_search"}],
                "iteration": 2,
            }

        graph = StateGraph(ResearchState)
        graph.add_node("a", node_a)
        graph.add_node("b", node_b)
        graph.set_entry_point("a")
        graph.add_edge("a", "b")
        graph.add_edge("b", END)
        compiled = graph.compile()

        result = compiled.invoke(create_initial_state("test question"))

        assert len(result["search_history"]) == 2
        assert result["search_history"][0]["tool"] == "arxiv_search"
        assert result["search_history"][1]["tool"] == "semantic_scholar_search"
        # scalar field: last write wins, not summed/merged
        assert result["iteration"] == 2

    def test_evidence_and_tool_call_hashes_accumulate_independently(self):
        def node_a(state: ResearchState) -> dict:
            return {
                "evidence": [{"paper_id": "p1", "claim": "c1"}],
                "tool_call_hashes": ["hash1"],
            }

        def node_b(state: ResearchState) -> dict:
            return {
                "evidence": [{"paper_id": "p2", "claim": "c2"}],
                "tool_call_hashes": ["hash2"],
            }

        graph = StateGraph(ResearchState)
        graph.add_node("a", node_a)
        graph.add_node("b", node_b)
        graph.set_entry_point("a")
        graph.add_edge("a", "b")
        graph.add_edge("b", END)
        compiled = graph.compile()

        result = compiled.invoke(create_initial_state("test question"))

        assert [e["paper_id"] for e in result["evidence"]] == ["p1", "p2"]
        assert result["tool_call_hashes"] == ["hash1", "hash2"]
