"""Tests for app/graph/* — routing logic and full compiled-graph behavior.

Covers Step 19 items 7 (max iteration enforcement), 8 (zero-yield
reflection), and general state-transition/termination correctness for the
compiled LangGraph.
"""

from __future__ import annotations

import responses

from app.graph.build_graph import build_research_graph
from app.graph.router import route_after_agent
from app.llm.base import AgentDecision, ReasoningClient
from app.state import create_initial_state
from app.tools.arxiv_tool import ARXIV_API_URL
from app.tools.registry import build_default_registry

SAMPLE_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.12345v1</id>
    <title>Multimodal RAG for Scientific Documents</title>
    <summary>abstract text</summary>
    <published>2024-01-20T00:00:00Z</published>
    <author><name>Jane Doe</name></author>
  </entry>
</feed>"""

EMPTY_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"></feed>"""


class ScriptedClient(ReasoningClient):
    """Returns scripted decisions in order; falls back to synthesize once
    exhausted so tests never hang waiting on an empty script."""

    def __init__(self, decisions: list[AgentDecision]):
        self.decisions = list(decisions)
        self.calls = 0

    def decide_next_action(self, state, tool_schemas):
        self.calls += 1
        if self.decisions:
            return self.decisions.pop(0)
        return AgentDecision(action="synthesize", rationale_summary="fallback")


class RepeatingClient(ReasoningClient):
    """Always proposes the same tool call — used to test dedup + bounded
    termination under worst-case agent behavior."""

    def __init__(self, query: str = "same query always"):
        self.query = query
        self.calls = 0

    def decide_next_action(self, state, tool_schemas):
        self.calls += 1
        return AgentDecision(action="call_tool", tool_name="arxiv_search", tool_args={"query": self.query})


class NeverConvergesClient(ReasoningClient):
    """Always proposes a NEW distinct query — proves the max-iterations
    bound works even when duplicate detection can't help."""

    def __init__(self):
        self.n = 0

    def decide_next_action(self, state, tool_schemas):
        self.n += 1
        return AgentDecision(action="call_tool", tool_name="arxiv_search", tool_args={"query": f"query {self.n}"})


# --------------------------------------------------------------------------
# Router unit tests (no graph compilation needed)
# --------------------------------------------------------------------------


class TestRouteAfterAgent:
    def test_routes_to_tool_on_call_tool_when_zero_yield_low(self):
        state = create_initial_state("q")
        state["pending_action"] = {"action": "call_tool", "tool_name": "arxiv_search", "tool_args": {"query": "x"}}
        state["consecutive_zero_yield_count"] = 0
        assert route_after_agent(state) == "tool"

    def test_routes_to_reflection_when_zero_yield_at_threshold(self):
        state = create_initial_state("q")
        state["pending_action"] = {"action": "call_tool", "tool_name": "arxiv_search", "tool_args": {"query": "x"}}
        state["consecutive_zero_yield_count"] = 2  # default threshold
        assert route_after_agent(state) == "reflection"

    def test_routes_to_synthesis_on_synthesize_action(self):
        state = create_initial_state("q")
        state["pending_action"] = {"action": "synthesize", "rationale_summary": "done"}
        assert route_after_agent(state) == "synthesis"

    def test_routes_to_synthesis_on_error_action(self):
        state = create_initial_state("q")
        state["pending_action"] = {"action": "error", "rationale_summary": "boom"}
        assert route_after_agent(state) == "synthesis"

    def test_fails_safe_to_synthesis_on_missing_pending_action(self):
        state = create_initial_state("q")
        state["pending_action"] = None
        assert route_after_agent(state) == "synthesis"


# --------------------------------------------------------------------------
# Full compiled-graph tests
# --------------------------------------------------------------------------


class TestCompiledGraphHappyPath:
    @responses.activate
    def test_tool_call_then_synthesize_terminates_cleanly(self):
        responses.add(responses.GET, ARXIV_API_URL, body=SAMPLE_ATOM, status=200)
        registry = build_default_registry()
        client = ScriptedClient([
            AgentDecision(action="call_tool", tool_name="arxiv_search", tool_args={"query": "multimodal RAG"}),
            AgentDecision(action="synthesize", rationale_summary="enough evidence"),
        ])
        app = build_research_graph(registry, client)

        result = app.invoke(
            create_initial_state("What are the limitations of multimodal RAG?", max_iterations=6),
            config={"recursion_limit": 50},
        )

        assert result["status"] == "done"
        assert result["iteration"] == 1
        assert result["termination_reason"] == "enough evidence"
        assert len(result["retrieved_papers"]) == 1
        assert result["final_report"] is not None
        assert "multimodal RAG" in result["final_report"] or "limitations" in result["final_report"]

    @responses.activate
    def test_multiple_tool_calls_accumulate_evidence_across_iterations(self):
        responses.add(responses.GET, ARXIV_API_URL, body=SAMPLE_ATOM, status=200)
        registry = build_default_registry()
        client = ScriptedClient([
            AgentDecision(action="call_tool", tool_name="arxiv_search", tool_args={"query": "topic A"}),
            AgentDecision(action="call_tool", tool_name="arxiv_search", tool_args={"query": "topic B"}),
            AgentDecision(action="synthesize", rationale_summary="done"),
        ])
        app = build_research_graph(registry, client)

        result = app.invoke(
            create_initial_state("test question", max_iterations=6),
            config={"recursion_limit": 50},
        )

        assert result["iteration"] == 2
        assert len(result["search_history"]) == 2
        assert len(result["retrieved_papers"]) == 2  # one paper per successful call


class TestCompiledGraphTermination:
    @responses.activate
    def test_terminates_at_max_iterations_even_if_llm_never_wants_to_stop(self):
        responses.add(responses.GET, ARXIV_API_URL, body=EMPTY_ATOM, status=200)
        registry = build_default_registry()
        client = NeverConvergesClient()
        app = build_research_graph(registry, client)

        result = app.invoke(
            create_initial_state("obscure topic", max_iterations=4),
            config={"recursion_limit": 50},
        )

        assert result["status"] == "done"
        assert result["iteration"] == 4
        # LLM consulted once per iteration + the final forced-synthesize turn
        assert client.n <= 5

    @responses.activate
    def test_zero_yield_triggers_reflection_before_max_iterations(self):
        responses.add(responses.GET, ARXIV_API_URL, body=EMPTY_ATOM, status=200)
        registry = build_default_registry()
        client = NeverConvergesClient()
        app = build_research_graph(registry, client)

        result = app.invoke(
            create_initial_state("obscure topic", max_iterations=4),
            config={"recursion_limit": 50},
        )

        reflection_notes = [
            note for note in result["scratchpad"]
            if note.get("stage") == "reflection"
        ]
        assert len(reflection_notes) >= 1

    @responses.activate
    def test_repeated_identical_tool_call_never_hits_network_twice(self):
        # Only one mocked response registered — a second real HTTP call
        # for the same args would raise since `responses` has no matching
        # mock queued for it (it does allow re-matching the same rule
        # multiple times by default, so we instead assert on the recorded
        # duplicate-skip status entries directly).
        responses.add(responses.GET, ARXIV_API_URL, body=SAMPLE_ATOM, status=200)
        registry = build_default_registry()
        client = RepeatingClient()
        app = build_research_graph(registry, client)

        result = app.invoke(
            create_initial_state("test", max_iterations=5),
            config={"recursion_limit": 50},
        )

        assert result["status"] == "done"
        duplicate_records = [
            rec for rec in result["search_history"]
            if rec.get("error") and "Duplicate" in rec["error"]
        ]
        assert len(duplicate_records) >= 1

    def test_llm_error_terminates_gracefully_with_error_report(self):
        class AlwaysErrorsClient(ReasoningClient):
            def decide_next_action(self, state, tool_schemas):
                return AgentDecision(action="error", rationale_summary="LLM call failed: simulated")

        registry = build_default_registry()
        client = AlwaysErrorsClient()
        app = build_research_graph(registry, client)

        result = app.invoke(
            create_initial_state("test", max_iterations=5),
            config={"recursion_limit": 50},
        )

        assert result["status"] == "done"  # synthesis still completes, doesn't crash
        assert "Error" in result["final_report"] or "error" in result["final_report"].lower()
