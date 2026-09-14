"""Tests for Step 11's full implementation: reflection-driven strategy
change (Step 11.C), cycle detection, and the independent max-tool-calls
bound.
"""

from __future__ import annotations

import responses

from app.config import Settings
from app.graph.build_graph import build_research_graph
from app.graph.reflection_node import build_reflection_node
from app.graph.router import route_after_agent
from app.llm.base import AgentDecision, ReasoningClient
from app.llm.reflection import NoOpReflectionAdvisor, ReflectionAdvisor, ReflectionGuidance
from app.safety.loop_guard import reflection_cycle_exhausted, should_force_synthesis
from app.state import create_initial_state
from app.tools.arxiv_tool import ARXIV_API_URL
from app.tools.registry import build_default_registry

EMPTY_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"></feed>"""


class FakeReflectionAdvisor(ReflectionAdvisor):
    def __init__(self, guidance: ReflectionGuidance):
        self._guidance = guidance
        self.calls = 0

    def reflect(self, state, available_tools):
        self.calls += 1
        return self._guidance


# --------------------------------------------------------------------------
# loop_guard unit tests
# --------------------------------------------------------------------------


class TestShouldForceSynthesis:
    def test_false_when_under_both_bounds(self):
        settings = Settings(max_iterations=6, max_tool_calls=12)
        state = create_initial_state("q", max_iterations=6)
        state["iteration"] = 2
        state["tool_call_history"] = [{}] * 3
        force, reason = should_force_synthesis(state, settings)
        assert force is False
        assert reason is None

    def test_true_at_max_iterations(self):
        settings = Settings(max_iterations=3, max_tool_calls=100)
        state = create_initial_state("q", max_iterations=3)
        state["iteration"] = 3
        force, reason = should_force_synthesis(state, settings)
        assert force is True
        assert "iteration" in reason.lower()

    def test_true_at_max_tool_calls_even_under_iteration_bound(self):
        """max_tool_calls acts as an INDEPENDENT ceiling — a run with a
        generous iteration budget but a tight tool-call budget must still
        stop."""
        settings = Settings(max_iterations=50, max_tool_calls=2)
        state = create_initial_state("q", max_iterations=50)
        state["iteration"] = 2
        state["tool_call_history"] = [{}, {}]
        force, reason = should_force_synthesis(state, settings)
        assert force is True
        assert "tool-call" in reason.lower()


class TestReflectionCycleExhausted:
    def test_false_below_threshold(self):
        settings = Settings(max_reflection_attempts=2)
        state = create_initial_state("q")
        state["reflection_count"] = 1
        assert reflection_cycle_exhausted(state, settings) is False

    def test_true_at_threshold(self):
        settings = Settings(max_reflection_attempts=2)
        state = create_initial_state("q")
        state["reflection_count"] = 2
        assert reflection_cycle_exhausted(state, settings) is True


# --------------------------------------------------------------------------
# Router: cycle detection escalates to synthesis instead of reflection
# --------------------------------------------------------------------------


class TestRouterCycleDetection:
    def test_routes_to_reflection_below_reflection_attempt_cap(self):
        state = create_initial_state("q")
        state["pending_action"] = {"action": "call_tool", "tool_name": "arxiv_search", "tool_args": {}}
        state["consecutive_zero_yield_count"] = 5
        state["reflection_count"] = 0
        assert route_after_agent(state) == "reflection"

    def test_routes_to_synthesis_once_reflection_cap_reached(self):
        state = create_initial_state("q")
        state["pending_action"] = {"action": "call_tool", "tool_name": "arxiv_search", "tool_args": {}}
        state["consecutive_zero_yield_count"] = 5
        state["reflection_count"] = 2  # default max_reflection_attempts
        assert route_after_agent(state) == "synthesis"


# --------------------------------------------------------------------------
# reflection_node behavior
# --------------------------------------------------------------------------


class TestReflectionNode:
    def test_guidance_stored_in_state_and_scratchpad(self):
        guidance = ReflectionGuidance(
            diagnosis="Queries were too close to the raw research question.",
            suggested_strategy_change="Target a specific sub-question with domain terminology.",
            suggested_tool="semantic_scholar_search",
            suggested_query="vision-language retrieval failure modes benchmark",
        )
        advisor = FakeReflectionAdvisor(guidance)
        node = build_reflection_node(advisor, ["arxiv_search", "semantic_scholar_search"])
        state = create_initial_state("q", max_iterations=6)
        state["consecutive_zero_yield_count"] = 2

        update = node(state)

        assert advisor.calls == 1
        assert update["iteration"] == 1
        assert update["consecutive_zero_yield_count"] == 0
        assert update["reflection_count"] == 1
        assert update["reflection_guidance"]["suggested_tool"] == "semantic_scholar_search"
        assert update["scratchpad"][0]["stage"] == "reflection"
        assert update["scratchpad"][0]["diagnosis"] == guidance.diagnosis

    def test_noop_advisor_produces_generic_but_present_guidance(self):
        node = build_reflection_node(NoOpReflectionAdvisor(), ["arxiv_search"])
        state = create_initial_state("q")
        update = node(state)
        assert update["reflection_guidance"]["diagnosis"]
        assert update["reflection_guidance"]["suggested_strategy_change"]


# --------------------------------------------------------------------------
# End-to-end: reflection guidance surfaces to the next agent turn
# --------------------------------------------------------------------------


class TestReflectionIntegratesWithGraph:
    @responses.activate
    def test_agent_sees_reflection_guidance_on_next_turn(self):
        responses.add(responses.GET, ARXIV_API_URL, body=EMPTY_ATOM, status=200)
        registry = build_default_registry()

        guidance = ReflectionGuidance(
            diagnosis="Too generic.",
            suggested_strategy_change="Be specific about failure modes.",
        )
        advisor = FakeReflectionAdvisor(guidance)

        class ScriptedClient(ReasoningClient):
            def __init__(self):
                self.n = 0
                self.saw_guidance_at_call = None

            def decide_next_action(self, state, tool_schemas):
                self.n += 1
                if state.get("reflection_guidance") and self.saw_guidance_at_call is None:
                    self.saw_guidance_at_call = self.n
                if self.n <= 3:
                    return AgentDecision(
                        action="call_tool", tool_name="arxiv_search",
                        tool_args={"query": f"query {self.n}"},
                    )
                return AgentDecision(action="synthesize", rationale_summary="done")

        client = ScriptedClient()
        app = build_research_graph(registry, client, reflection_advisor=advisor)

        result = app.invoke(
            create_initial_state("test question", max_iterations=6),
            config={"recursion_limit": 50},
        )

        assert advisor.calls >= 1
        assert client.saw_guidance_at_call is not None
        assert result["status"] == "done"

    @responses.activate
    def test_repeated_reflection_eventually_gives_up_gracefully(self):
        """Worst case: LLM always searches with a new distinct query
        (never duplicates) but never finds anything AND never resolves the
        zero-yield problem despite reflection guidance. The graph must
        still terminate — via cycle detection, likely before max_iterations."""
        responses.add(responses.GET, ARXIV_API_URL, body=EMPTY_ATOM, status=200)
        registry = build_default_registry()

        class NeverConvergesClient(ReasoningClient):
            def __init__(self):
                self.n = 0

            def decide_next_action(self, state, tool_schemas):
                self.n += 1
                return AgentDecision(
                    action="call_tool", tool_name="arxiv_search",
                    tool_args={"query": f"query {self.n}"},
                )

        client = NeverConvergesClient()
        app = build_research_graph(
            registry, client,
            reflection_advisor=NoOpReflectionAdvisor(),
        )

        result = app.invoke(
            create_initial_state("obscure topic", max_iterations=10),
            config={"recursion_limit": 100},
        )

        assert result["status"] == "done"
        reflection_notes = [n for n in result["scratchpad"] if n.get("stage") == "reflection"]
        # Should have given up via cycle detection (reflection fired at
        # most max_reflection_attempts+1 times) well before iteration 10.
        assert len(reflection_notes) <= 3
        assert result["iteration"] < 10
