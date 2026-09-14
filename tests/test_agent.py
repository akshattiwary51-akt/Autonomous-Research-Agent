"""Tests for app/agent.py — the core Reason -> Act -> Observe loop.

Uses a scripted `FakeReasoningClient` (no live LLM calls) and mocked HTTP
for the real ArXiv tool, per Step 19's "mock external APIs" requirement.
Covers Step 19 items 6 (duplicate tool-call detection) and 7 (max
iteration enforcement), plus general agent-step behavior.
"""

from __future__ import annotations

import responses

from app.agent import decide_action, execute_tool_call, is_duplicate_call, run_agent_step
from app.llm.base import AgentDecision, ReasoningClient
from app.state import ResearchState, create_initial_state
from app.tools.arxiv_tool import ARXIV_API_URL
from app.tools.base import ResearchTool, ToolResult, hash_tool_call
from app.tools.registry import build_default_registry

SAMPLE_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.12345v1</id>
    <title>Multimodal RAG for Scientific Documents</title>
    <summary>We study multimodal retrieval augmented generation.</summary>
    <published>2024-01-20T00:00:00Z</published>
    <author><name>Jane Doe</name></author>
  </entry>
</feed>"""

EMPTY_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"></feed>"""


class FakeReasoningClient(ReasoningClient):
    """Test double that returns a pre-scripted sequence of decisions."""

    def __init__(self, decisions: list[AgentDecision]):
        self._decisions = list(decisions)
        self.calls = 0

    def decide_next_action(self, state: ResearchState, tool_schemas: list[dict]) -> AgentDecision:
        self.calls += 1
        if not self._decisions:
            raise AssertionError("FakeReasoningClient ran out of scripted decisions")
        return self._decisions.pop(0)


class AlwaysFailsTool(ResearchTool):
    name = "always_fails"
    description = "A tool that always fails, for testing error handling."

    def parameters_schema(self):
        return {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}

    def run(self, query: str, max_results: int = 5, **kwargs) -> ToolResult:
        return ToolResult(tool_name=self.name, query=query, success=False, error="simulated failure")


class TestDecideAction:
    def test_defers_to_reasoning_client_when_under_iteration_limit(self):
        state = create_initial_state("test question", max_iterations=6)
        registry = build_default_registry()
        expected = AgentDecision(action="call_tool", tool_name="arxiv_search", tool_args={"query": "rag"})
        client = FakeReasoningClient([expected])

        decision = decide_action(state, registry, client)

        assert decision is expected
        assert client.calls == 1

    def test_forces_synthesize_at_max_iterations_without_calling_llm(self):
        state = create_initial_state("test question", max_iterations=2)
        state["iteration"] = 2  # already at the bound
        registry = build_default_registry()
        client = FakeReasoningClient([])  # should never be consulted

        decision = decide_action(state, registry, client)

        assert decision.action == "synthesize"
        assert client.calls == 0


class TestExecuteToolCall:
    @responses.activate
    def test_successful_tool_call_updates_state_correctly(self):
        responses.add(responses.GET, ARXIV_API_URL, body=SAMPLE_ATOM, status=200)
        state = create_initial_state("multimodal RAG limitations")
        registry = build_default_registry()
        decision = AgentDecision(
            action="call_tool", tool_name="arxiv_search",
            tool_args={"query": "multimodal RAG", "max_results": 5},
        )

        update = execute_tool_call(state, registry, decision)

        assert update["iteration"] == 1
        assert update["status"] == "researching"
        assert len(update["search_history"]) == 1
        assert update["search_history"][0]["success"] is True
        assert update["search_history"][0]["result_count"] == 1
        assert update["useful_results_count"] == 1
        assert update["consecutive_zero_yield_count"] == 0
        assert len(update["retrieved_papers"]) == 1
        assert update["retrieved_papers"][0]["title"] == "Multimodal RAG for Scientific Documents"
        assert len(update["tool_call_hashes"]) == 1

    @responses.activate
    def test_zero_results_increments_zero_yield_counter(self):
        responses.add(responses.GET, ARXIV_API_URL, body=EMPTY_ATOM, status=200)
        state = create_initial_state("an extremely obscure topic")
        registry = build_default_registry()
        decision = AgentDecision(
            action="call_tool", tool_name="arxiv_search",
            tool_args={"query": "an extremely obscure topic", "max_results": 5},
        )

        update = execute_tool_call(state, registry, decision)

        assert update["consecutive_zero_yield_count"] == 1
        assert update["failed_queries"] == ["an extremely obscure topic"]

    def test_unknown_tool_is_handled_gracefully_not_raised(self):
        state = create_initial_state("test question")
        registry = build_default_registry()
        decision = AgentDecision(action="call_tool", tool_name="not_a_real_tool", tool_args={"query": "x"})

        update = execute_tool_call(state, registry, decision)

        assert update["status"] == "researching"
        assert update["search_history"][0]["success"] is False
        assert "Unknown tool" in update["search_history"][0]["error"]

    def test_tool_execution_failure_recorded_not_raised(self):
        state = create_initial_state("test question")
        from app.tools.base import ToolRegistry
        registry = ToolRegistry()
        registry.register(AlwaysFailsTool())
        decision = AgentDecision(action="call_tool", tool_name="always_fails", tool_args={"query": "x"})

        update = execute_tool_call(state, registry, decision)

        assert update["search_history"][0]["success"] is False
        assert update["search_history"][0]["error"] == "simulated failure"
        assert update["useful_results_count"] == 0


class TestDuplicateCallDetection:
    def test_is_duplicate_call_false_on_fresh_state(self):
        state = create_initial_state("test question")
        assert is_duplicate_call(state, "arxiv_search", {"query": "rag"}) is False

    def test_is_duplicate_call_true_after_hash_recorded(self):
        state = create_initial_state("test question")
        h = hash_tool_call("arxiv_search", {"query": "rag"})
        state["tool_call_hashes"] = [h]
        assert is_duplicate_call(state, "arxiv_search", {"query": "rag"}) is True
        # normalization still applies
        assert is_duplicate_call(state, "arxiv_search", {"query": "  RAG  "}) is True

    @responses.activate
    def test_duplicate_call_is_skipped_without_network_request(self):
        # Only register ONE response — if execute_tool_call made two real
        # HTTP calls, the second would raise a ConnectionError from
        # `responses` since no matching mock is registered.
        responses.add(responses.GET, ARXIV_API_URL, body=SAMPLE_ATOM, status=200)

        state = create_initial_state("test question")
        registry = build_default_registry()
        decision = AgentDecision(
            action="call_tool", tool_name="arxiv_search",
            tool_args={"query": "multimodal RAG", "max_results": 5},
        )

        first_update = execute_tool_call(state, registry, decision)
        state["iteration"] = first_update["iteration"]
        state["tool_call_hashes"] = first_update["tool_call_hashes"]

        second_update = execute_tool_call(state, registry, decision)

        assert second_update["search_history"][0]["success"] is False
        assert "Duplicate" in second_update["search_history"][0]["error"]


class TestRunAgentStep:
    def test_synthesize_decision_sets_status_and_termination_reason(self):
        state = create_initial_state("test question")
        registry = build_default_registry()
        client = FakeReasoningClient([
            AgentDecision(action="synthesize", rationale_summary="Enough evidence collected.")
        ])

        update = run_agent_step(state, registry, client)

        assert update["status"] == "synthesizing"
        assert update["termination_reason"] == "Enough evidence collected."

    def test_error_decision_sets_status_error(self):
        state = create_initial_state("test question")
        registry = build_default_registry()
        client = FakeReasoningClient([
            AgentDecision(action="error", rationale_summary="LLM call failed: timeout")
        ])

        update = run_agent_step(state, registry, client)

        assert update["status"] == "error"
        assert update["termination_reason"] == "llm_error"
        assert "timeout" in update["error"]

    @responses.activate
    def test_call_tool_decision_executes_and_updates_state(self):
        responses.add(responses.GET, ARXIV_API_URL, body=SAMPLE_ATOM, status=200)
        state = create_initial_state("test question")
        registry = build_default_registry()
        client = FakeReasoningClient([
            AgentDecision(action="call_tool", tool_name="arxiv_search", tool_args={"query": "rag"})
        ])

        update = run_agent_step(state, registry, client)

        assert update["status"] == "researching"
        assert update["iteration"] == 1

    def test_max_iterations_short_circuits_without_consulting_llm(self):
        state = create_initial_state("test question", max_iterations=1)
        state["iteration"] = 1
        registry = build_default_registry()
        client = FakeReasoningClient([])  # would raise if consulted

        update = run_agent_step(state, registry, client)

        assert update["status"] == "synthesizing"
        assert client.calls == 0


class TestMultiStepLoopSimulation:
    """Simulates several ReAct iterations end-to-end to confirm state
    accumulates correctly and the loop terminates (Step 19 item 7)."""

    @responses.activate
    def test_loop_terminates_at_max_iterations_via_repeated_agent_steps(self):
        responses.add(responses.GET, ARXIV_API_URL, body=SAMPLE_ATOM, status=200)
        registry = build_default_registry()
        state = create_initial_state("test question", max_iterations=3)

        # Script: two distinct tool calls, then agent tries to keep going
        # but the max-iterations guard should kick in on the 3rd step
        # regardless of what the LLM would have said.
        client = FakeReasoningClient([
            AgentDecision(action="call_tool", tool_name="arxiv_search", tool_args={"query": "rag one"}),
            AgentDecision(action="call_tool", tool_name="arxiv_search", tool_args={"query": "rag two"}),
        ])

        for _ in range(2):
            update = run_agent_step(state, registry, client)
            state.update(update)  # simulate LangGraph merge for scalar fields
            state["search_history"] = state.get("search_history", []) + update.get("search_history", [])

        assert state["iteration"] == 2
        assert state["status"] == "researching"

        # Third call: iteration (2) < max_iterations (3), so LLM IS consulted,
        # but since we scripted no more decisions, force max_iterations=2
        # to prove the hard bound instead.
        state["max_iterations"] = 2
        final_update = run_agent_step(state, registry, client)
        assert final_update["status"] == "synthesizing"
        assert client.calls == 2  # not called a 3rd time
