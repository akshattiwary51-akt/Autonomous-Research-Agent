"""Tests for app/safety/circuit_breaker.py and its integration into the
agent loop and graph -- the automatic provider-failover fix motivated by
real 429s observed from ArXiv and Semantic Scholar in live usage.
"""

from __future__ import annotations

import time

import responses

from app.agent import decide_action, execute_tool_call
from app.graph.build_graph import build_research_graph
from app.llm.base import AgentDecision, ReasoningClient
from app.safety.circuit_breaker import CircuitBreaker
from app.state import create_initial_state
from app.tools.arxiv_tool import ARXIV_API_URL, ArxivTool
from app.tools.base import RateLimitError
from app.tools.crossref_tool import CROSSREF_API_URL, CrossrefTool
from app.tools.registry import build_default_registry
from app.tools.semantic_scholar_tool import S2_API_URL, SemanticScholarTool

SAMPLE_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.12345v1</id>
    <title>Test Paper</title>
    <summary>abstract text</summary>
  </entry>
</feed>"""


# --------------------------------------------------------------------------
# CircuitBreaker core logic (deterministic, using monkeypatched time)
# --------------------------------------------------------------------------


class TestCircuitBreakerCore:
    def test_closed_by_default(self):
        cb = CircuitBreaker(cooldown_seconds=60)
        assert cb.is_open("arxiv_search") is False

    def test_opens_on_rate_limited_result(self):
        cb = CircuitBreaker(cooldown_seconds=60)
        cb.record_result("arxiv_search", success=False, rate_limited=True)
        assert cb.is_open("arxiv_search") is True

    def test_does_not_open_on_non_rate_limited_failure(self):
        """A timeout or malformed response shouldn't trip the breaker --
        only an explicit 429 signal, since switching tools doesn't
        necessarily fix a timeout."""
        cb = CircuitBreaker(cooldown_seconds=60)
        cb.record_result("arxiv_search", success=False, rate_limited=False)
        assert cb.is_open("arxiv_search") is False

    def test_closes_immediately_on_success(self):
        cb = CircuitBreaker(cooldown_seconds=60)
        cb.record_result("arxiv_search", success=False, rate_limited=True)
        assert cb.is_open("arxiv_search") is True
        cb.record_result("arxiv_search", success=True, rate_limited=False)
        assert cb.is_open("arxiv_search") is False

    def test_reopens_via_cooldown_expiry(self, monkeypatch):
        cb = CircuitBreaker(cooldown_seconds=10)
        fake_time = [1000.0]
        monkeypatch.setattr(time, "monotonic", lambda: fake_time[0])

        cb.record_result("arxiv_search", success=False, rate_limited=True)
        assert cb.is_open("arxiv_search") is True

        fake_time[0] += 5  # still within cooldown
        assert cb.is_open("arxiv_search") is True

        fake_time[0] += 6  # now past cooldown (11s elapsed total)
        assert cb.is_open("arxiv_search") is False

    def test_only_the_tripped_tool_is_affected(self):
        cb = CircuitBreaker(cooldown_seconds=60)
        cb.record_result("arxiv_search", success=False, rate_limited=True)
        assert cb.is_open("arxiv_search") is True
        assert cb.is_open("semantic_scholar_search") is False

    def test_available_tools_excludes_open_ones(self):
        cb = CircuitBreaker(cooldown_seconds=60)
        cb.record_result("arxiv_search", success=False, rate_limited=True)
        available = cb.available_tools(["arxiv_search", "semantic_scholar_search", "crossref_search"])
        assert "arxiv_search" not in available
        assert "semantic_scholar_search" in available
        assert "crossref_search" in available

    def test_fails_open_when_everything_is_down(self):
        """If every tool is currently rate-limited, the agent must still
        be offered SOMETHING -- a throttled tool is better than zero
        options (which would otherwise force premature synthesis)."""
        cb = CircuitBreaker(cooldown_seconds=60)
        all_tools = ["arxiv_search", "semantic_scholar_search", "crossref_search"]
        for name in all_tools:
            cb.record_result(name, success=False, rate_limited=True)

        available = cb.available_tools(all_tools)
        assert available == all_tools


# --------------------------------------------------------------------------
# ToolResult.rate_limited set correctly by all three tools
# --------------------------------------------------------------------------


class TestRateLimitedFlagPerTool:
    @responses.activate
    def test_arxiv_sets_rate_limited_on_429(self):
        responses.add(responses.GET, ARXIV_API_URL, status=429)
        result = ArxivTool().run(query="x", max_results=5)
        assert result.success is False
        assert result.rate_limited is True

    @responses.activate
    def test_arxiv_does_not_set_rate_limited_on_500(self):
        responses.add(responses.GET, ARXIV_API_URL, status=500)
        result = ArxivTool().run(query="x", max_results=5)
        assert result.success is False
        assert result.rate_limited is False

    @responses.activate
    def test_semantic_scholar_sets_rate_limited_on_429(self):
        responses.add(responses.GET, S2_API_URL, status=429)
        result = SemanticScholarTool().run(query="x", max_results=5)
        assert result.success is False
        assert result.rate_limited is True

    @responses.activate
    def test_crossref_sets_rate_limited_on_429(self):
        responses.add(responses.GET, CROSSREF_API_URL, status=429)
        result = CrossrefTool().run(query="x", max_results=5)
        assert result.success is False
        assert result.rate_limited is True

    @responses.activate
    def test_successful_result_has_rate_limited_false(self):
        responses.add(responses.GET, ARXIV_API_URL, body=SAMPLE_ATOM, status=200)
        result = ArxivTool().run(query="x", max_results=5)
        assert result.success is True
        assert result.rate_limited is False

    def test_rate_limit_error_is_shared_across_tools(self):
        """Consolidated into app.tools.base -- not three separate classes
        that happen to have the same name."""
        import app.tools.arxiv_tool as arxiv_mod
        import app.tools.crossref_tool as crossref_mod
        import app.tools.semantic_scholar_tool as s2_mod
        assert arxiv_mod.RateLimitError is RateLimitError
        assert crossref_mod.RateLimitError is RateLimitError
        assert s2_mod.RateLimitError is RateLimitError


# --------------------------------------------------------------------------
# ToolRegistry.schemas_for
# --------------------------------------------------------------------------


class TestSchemasFor:
    def test_returns_only_requested_tool_schemas(self):
        registry = build_default_registry()
        schemas = registry.schemas_for(["arxiv_search"])
        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "arxiv_search"

    def test_empty_list_returns_no_schemas(self):
        registry = build_default_registry()
        assert registry.schemas_for([]) == []


# --------------------------------------------------------------------------
# decide_action / execute_tool_call wiring
# --------------------------------------------------------------------------


class RecordingClient(ReasoningClient):
    """Records exactly which tool schemas it was offered."""

    def __init__(self, next_decision: AgentDecision):
        self._next = next_decision
        self.seen_tool_names: list[str] | None = None

    def decide_next_action(self, state, tool_schemas):
        self.seen_tool_names = [s["function"]["name"] for s in tool_schemas]
        return self._next


class TestDecideActionRespectsCircuitBreaker:
    def test_open_tool_excluded_from_offered_schemas(self):
        cb = CircuitBreaker(cooldown_seconds=60)
        cb.record_result("arxiv_search", success=False, rate_limited=True)

        registry = build_default_registry()
        client = RecordingClient(AgentDecision(action="synthesize", rationale_summary="done"))
        state = create_initial_state("test question")

        decide_action(state, registry, client, circuit_breaker=cb)

        assert "arxiv_search" not in client.seen_tool_names
        assert "semantic_scholar_search" in client.seen_tool_names
        assert "crossref_search" in client.seen_tool_names

    def test_no_circuit_breaker_offers_all_tools_unchanged(self):
        registry = build_default_registry()
        client = RecordingClient(AgentDecision(action="synthesize", rationale_summary="done"))
        state = create_initial_state("test question")

        decide_action(state, registry, client)  # circuit_breaker=None (default)

        assert set(client.seen_tool_names) == set(registry.names())


class TestExecuteToolCallRecordsIntoCircuitBreaker:
    @responses.activate
    def test_429_result_trips_the_breaker(self):
        responses.add(responses.GET, ARXIV_API_URL, status=429)
        cb = CircuitBreaker(cooldown_seconds=60)
        registry = build_default_registry()
        state = create_initial_state("test question")
        decision = AgentDecision(action="call_tool", tool_name="arxiv_search", tool_args={"query": "x"})

        execute_tool_call(state, registry, decision, circuit_breaker=cb)

        assert cb.is_open("arxiv_search") is True

    @responses.activate
    def test_successful_result_does_not_trip_the_breaker(self):
        responses.add(responses.GET, ARXIV_API_URL, body=SAMPLE_ATOM, status=200)
        cb = CircuitBreaker(cooldown_seconds=60)
        registry = build_default_registry()
        state = create_initial_state("test question")
        decision = AgentDecision(action="call_tool", tool_name="arxiv_search", tool_args={"query": "x"})

        execute_tool_call(state, registry, decision, circuit_breaker=cb)

        assert cb.is_open("arxiv_search") is False


# --------------------------------------------------------------------------
# End-to-end: automatic failover through the compiled graph
# --------------------------------------------------------------------------


class TestEndToEndFailover:
    @responses.activate
    def test_rate_limited_tool_excluded_from_agents_very_next_turn(self):
        """This is the core scenario from the real run that motivated this
        feature: tool A returns 429, and the agent's NEXT turn should not
        even be offered tool A as an option -- proving failover doesn't
        need to wait for reflection to notice the pattern."""
        responses.add(responses.GET, ARXIV_API_URL, status=429)
        responses.add(responses.GET, S2_API_URL, body='{"data": []}', status=200)

        registry = build_default_registry()

        class ScriptedClient(ReasoningClient):
            def __init__(self):
                self.n = 0
                self.offered_tools_by_turn: list[list[str]] = []

            def decide_next_action(self, state, tool_schemas):
                self.n += 1
                self.offered_tools_by_turn.append([s["function"]["name"] for s in tool_schemas])
                if self.n == 1:
                    return AgentDecision(action="call_tool", tool_name="arxiv_search", tool_args={"query": "x"})
                return AgentDecision(action="synthesize", rationale_summary="done")

        client = ScriptedClient()
        app = build_research_graph(registry, client)  # circuit_breaker defaults to a live instance

        app.invoke(
            create_initial_state("test question", max_iterations=6),
            config={"recursion_limit": 50},
        )

        # Turn 1: arxiv_search was offered (nothing tripped yet)
        assert "arxiv_search" in client.offered_tools_by_turn[0]
        # Turn 2 (after the 429): arxiv_search must be excluded
        assert "arxiv_search" not in client.offered_tools_by_turn[1]
        assert "semantic_scholar_search" in client.offered_tools_by_turn[1]

    @responses.activate
    def test_circuit_breaker_can_be_explicitly_disabled_via_zero_length_registry_behavior(self):
        """Passing a circuit breaker with an effectively-zero cooldown
        means a tripped tool becomes available again almost immediately
        -- confirms the cooldown is genuinely configurable end-to-end,
        not hardcoded."""
        responses.add(responses.GET, ARXIV_API_URL, status=429)
        responses.add(responses.GET, S2_API_URL, body='{"data": []}', status=200)

        registry = build_default_registry()
        short_cb = CircuitBreaker(cooldown_seconds=0.0)  # expires instantly

        class ScriptedClient(ReasoningClient):
            def __init__(self):
                self.n = 0
                self.offered_tools_by_turn: list[list[str]] = []

            def decide_next_action(self, state, tool_schemas):
                self.n += 1
                self.offered_tools_by_turn.append([s["function"]["name"] for s in tool_schemas])
                if self.n == 1:
                    return AgentDecision(action="call_tool", tool_name="arxiv_search", tool_args={"query": "x"})
                return AgentDecision(action="synthesize", rationale_summary="done")

        client = ScriptedClient()
        app = build_research_graph(registry, client, circuit_breaker=short_cb)

        app.invoke(
            create_initial_state("test question", max_iterations=6),
            config={"recursion_limit": 50},
        )

        # With ~0s cooldown, arxiv_search should already be available again
        # by the very next turn.
        assert "arxiv_search" in client.offered_tools_by_turn[1]
