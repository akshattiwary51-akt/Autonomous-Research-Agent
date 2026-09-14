"""Tests for Step 19 item 9 (query refinement) plus query decomposition
(Step 7) — the planner and the rationale/sub_question metadata threaded
through tool calls.
"""

from __future__ import annotations

import responses

from app.agent import execute_tool_call
from app.graph.build_graph import build_research_graph
from app.graph.planner_node import build_planner_node
from app.llm.base import AgentDecision, ReasoningClient
from app.llm.decomposition import NoOpDecomposer, QueryDecomposer
from app.state import create_initial_state
from app.tools.arxiv_tool import ARXIV_API_URL
from app.tools.base import with_reasoning_fields
from app.tools.registry import build_default_registry

SAMPLE_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.12345v1</id>
    <title>Multimodal RAG for Scientific Documents</title>
    <summary>abstract text</summary>
  </entry>
</feed>"""


class FakeDecomposer(QueryDecomposer):
    def __init__(self, sub_questions: list[str]):
        self._sub_questions = sub_questions
        self.calls = 0

    def decompose(self, research_question: str, max_sub_questions: int = 6) -> list[str]:
        self.calls += 1
        return self._sub_questions[:max_sub_questions]


class FailingDecomposer(QueryDecomposer):
    """Simulates an LLM failure — should surface as an empty list, never raise."""

    def decompose(self, research_question: str, max_sub_questions: int = 6) -> list[str]:
        return []  # a real implementation would catch its own exception


# --------------------------------------------------------------------------
# Tool schemas expose reasoning fields
# --------------------------------------------------------------------------


class TestToolSchemasExposeReasoningFields:
    def test_with_reasoning_fields_adds_sub_question_and_rationale(self):
        base = {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
        augmented = with_reasoning_fields(base)
        assert "sub_question" in augmented["properties"]
        assert "rationale" in augmented["properties"]
        # required list untouched — both new fields stay optional
        assert augmented["required"] == ["query"]

    def test_registered_tools_all_expose_reasoning_fields(self):
        registry = build_default_registry()
        for schema in registry.schemas():
            params = schema["function"]["parameters"]["properties"]
            assert "rationale" in params
            assert "sub_question" in params


# --------------------------------------------------------------------------
# Planner node / decomposition
# --------------------------------------------------------------------------


class TestPlannerNode:
    def test_planner_populates_sub_questions_from_decomposer(self):
        decomposer = FakeDecomposer([
            "What multimodal RAG approaches currently exist?",
            "What limitations have been reported?",
        ])
        node = build_planner_node(decomposer)
        state = create_initial_state("What are the limitations of multimodal RAG?")

        update = node(state)

        assert update["sub_questions"] == [
            "What multimodal RAG approaches currently exist?",
            "What limitations have been reported?",
        ]
        assert update["status"] == "researching"
        assert decomposer.calls == 1

    def test_planner_falls_back_gracefully_when_decomposer_fails(self):
        node = build_planner_node(FailingDecomposer())
        state = create_initial_state("test question")

        update = node(state)

        assert update["sub_questions"] == []
        assert update["status"] == "researching"  # doesn't crash or block progress

    def test_noop_decomposer_used_by_default_produces_no_sub_questions(self):
        node = build_planner_node(NoOpDecomposer())
        state = create_initial_state("test question")
        update = node(state)
        assert update["sub_questions"] == []

    def test_planner_respects_max_sub_questions_cap(self):
        decomposer = FakeDecomposer([f"sub-question {i}" for i in range(10)])
        node = build_planner_node(decomposer, max_sub_questions=3)
        state = create_initial_state("test question")
        update = node(state)
        assert len(update["sub_questions"]) == 3


# --------------------------------------------------------------------------
# Rationale / sub_question threading through tool execution (Step 8)
# --------------------------------------------------------------------------


class TestQueryRefinementMetadata:
    @responses.activate
    def test_rationale_and_sub_question_recorded_in_search_history(self):
        responses.add(responses.GET, ARXIV_API_URL, body=SAMPLE_ATOM, status=200)
        state = create_initial_state("multimodal RAG limitations")
        registry = build_default_registry()
        decision = AgentDecision(
            action="call_tool",
            tool_name="arxiv_search",
            tool_args={"query": "multimodal RAG failure modes"},
            sub_question="What failure modes have been reported?",
            rationale_summary="Previous broader query returned generic results; narrowing to failure modes specifically.",
        )

        update = execute_tool_call(state, registry, decision)

        record = update["search_history"][0]
        assert record["sub_question"] == "What failure modes have been reported?"
        assert "narrowing to failure modes" in record["rationale"]

    @responses.activate
    def test_scratchpad_captures_safe_structured_metadata_not_raw_cot(self):
        responses.add(responses.GET, ARXIV_API_URL, body=SAMPLE_ATOM, status=200)
        state = create_initial_state("test question")
        registry = build_default_registry()
        decision = AgentDecision(
            action="call_tool", tool_name="arxiv_search",
            tool_args={"query": "rag"}, sub_question="sub-q", rationale_summary="because X",
        )

        update = execute_tool_call(state, registry, decision)

        note = update["scratchpad"][0]
        assert note["stage"] == "tool_execution"
        assert note["objective"] == "sub-q"
        assert note["tool"] == "arxiv_search"
        assert note["rationale"] == "because X"
        assert "usable" in note["observation_summary"]


# --------------------------------------------------------------------------
# OpenAI client correctly strips reasoning fields out of tool_args
# --------------------------------------------------------------------------


class TestOpenAIClientStripsReasoningFields:
    def test_sub_question_and_rationale_not_leaked_into_tool_args(self):
        """Simulates what OpenAIReasoningClient.decide_next_action does
        with a raw tool-call args dict, without needing a live LLM."""
        raw_args = {
            "query": "vision language models scientific PDFs",
            "max_results": 5,
            "sub_question": "What datasets are used for evaluation?",
            "rationale": "Broadening scope after narrow query returned nothing.",
        }

        # Replicate the parsing logic path (same as openai_client.py)
        args = dict(raw_args)
        sub_question = args.pop("sub_question", None)
        rationale = args.pop("rationale", None)

        assert args == {"query": "vision language models scientific PDFs", "max_results": 5}
        assert sub_question == "What datasets are used for evaluation?"
        assert rationale == "Broadening scope after narrow query returned nothing."


# --------------------------------------------------------------------------
# End-to-end: planner feeds graph, agent sees sub_questions in prompt context
# --------------------------------------------------------------------------


class TestPlannerIntegratesWithGraph:
    @responses.activate
    def test_graph_wires_decomposer_and_sub_questions_flow_through(self):
        responses.add(responses.GET, ARXIV_API_URL, body=SAMPLE_ATOM, status=200)
        registry = build_default_registry()
        decomposer = FakeDecomposer(["Sub-question one?", "Sub-question two?"])

        class ScriptedClient(ReasoningClient):
            def __init__(self):
                self.seen_sub_questions_in_state = None

            def decide_next_action(self, state, tool_schemas):
                self.seen_sub_questions_in_state = state.get("sub_questions")
                return AgentDecision(action="synthesize", rationale_summary="done")

        client = ScriptedClient()
        app = build_research_graph(registry, client, decomposer=decomposer)

        result = app.invoke(
            create_initial_state("test question", max_iterations=6),
            config={"recursion_limit": 50},
        )

        assert result["sub_questions"] == ["Sub-question one?", "Sub-question two?"]
        assert client.seen_sub_questions_in_state == ["Sub-question one?", "Sub-question two?"]
