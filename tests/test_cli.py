"""Tests for app/__main__.py (Step 21: CLI)."""

from __future__ import annotations

import responses

from app.__main__ import print_update, run_research
from app.config import Settings
from app.evidence.evaluator import (
    NoOpContradictionDetector,
    NoOpEvidenceEvaluator,
)
from app.llm.base import AgentDecision, ReasoningClient
from app.llm.decomposition import NoOpDecomposer
from app.llm.reflection import NoOpReflectionAdvisor
from app.report.gap_discovery import NoOpGapDiscoverer
from app.report.narrative import NoOpNarrativeWriter
from app.tools.arxiv_tool import ARXIV_API_URL
from app.tools.registry import build_default_registry

SAMPLE_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.12345v1</id>
    <title>Test Paper</title>
    <summary>abstract text</summary>
  </entry>
</feed>"""


class ScriptedClient(ReasoningClient):
    def __init__(self):
        self.n = 0

    def decide_next_action(self, state, tool_schemas):
        self.n += 1
        if self.n == 1:
            return AgentDecision(action="call_tool", tool_name="arxiv_search", tool_args={"query": "q1"})
        return AgentDecision(action="synthesize", rationale_summary="done")


def _default_kwargs(registry, client):
    return dict(
        registry=registry,
        reasoning_client=client,
        decomposer=NoOpDecomposer(),
        evidence_evaluator=NoOpEvidenceEvaluator(),
        contradiction_detector=NoOpContradictionDetector(),
        reflection_advisor=NoOpReflectionAdvisor(),
        gap_discoverer=NoOpGapDiscoverer(),
        narrative_writer=NoOpNarrativeWriter(),
    )


class TestPrintUpdate:
    def test_planner_with_sub_questions(self, capsys):
        print_update("planner", {"sub_questions": ["Sub Q1?", "Sub Q2?"]})
        out = capsys.readouterr().out
        assert "Decomposed into 2 sub-question(s)" in out
        assert "Sub Q1?" in out

    def test_planner_without_sub_questions(self, capsys):
        print_update("planner", {"sub_questions": []})
        out = capsys.readouterr().out
        assert "No decomposition available" in out

    def test_tool_update_shows_query_and_result_count(self, capsys):
        print_update("tool", {"search_history": [
            {"tool": "arxiv_search", "query": "rag", "result_count": 3, "success": True},
        ]})
        out = capsys.readouterr().out
        assert "arxiv_search" in out
        assert "3 result(s)" in out

    def test_tool_update_shows_failure_reason(self, capsys):
        print_update("tool", {"search_history": [
            {"tool": "arxiv_search", "query": "rag", "result_count": 0, "success": False, "error": "timeout"},
        ]})
        out = capsys.readouterr().out
        assert "failed (timeout)" in out

    def test_never_prints_raw_reasoning_fields(self, capsys):
        """Sanity check: print_update only reads whitelisted keys — an
        update dict with an unexpected 'raw_chain_of_thought'-style key
        must not leak into output just because it's present in the dict."""
        print_update("agent", {"raw_chain_of_thought": "secret internal reasoning"})
        out = capsys.readouterr().out
        assert "secret internal reasoning" not in out

    def test_unknown_node_uses_generic_label(self, capsys):
        print_update("some_future_node", {})
        out = capsys.readouterr().out
        assert "[some_future_node]" in out


class TestRunResearchNonHitl:
    @responses.activate
    def test_produces_complete_report_and_calls_on_update_per_node(self):
        responses.add(responses.GET, ARXIV_API_URL, body=SAMPLE_ATOM, status=200)
        registry = build_default_registry()
        settings = Settings(checkpoint_backend="memory", enable_hitl=False)

        events = []
        report = run_research(
            "test question", settings,
            on_update=lambda name, update: events.append(name),
            **_default_kwargs(registry, ScriptedClient()),
        )

        assert "# Research Question" in report
        assert "planner" in events
        assert "tool" in events
        assert "synthesis" in events

    @responses.activate
    def test_never_pauses_when_hitl_disabled(self):
        responses.add(responses.GET, ARXIV_API_URL, body=SAMPLE_ATOM, status=200)
        registry = build_default_registry()
        settings = Settings(checkpoint_backend="memory", enable_hitl=False)

        approve_calls = []
        run_research(
            "test question", settings,
            approve_tool_call=lambda pending: approve_calls.append(pending) or True,
            **_default_kwargs(registry, ScriptedClient()),
        )

        assert approve_calls == []  # approval callback never invoked


class TestParseArgs:
    def test_no_args_defaults_to_interactive(self):
        from app.__main__ import parse_args
        args = parse_args([])
        assert args.question is None
        assert args.output is None
        assert args.quiet is False

    def test_positional_question_captured(self):
        from app.__main__ import parse_args
        args = parse_args(["What is LLM used for?"])
        assert args.question == "What is LLM used for?"

    def test_output_and_quiet_flags(self):
        from app.__main__ import parse_args
        args = parse_args(["a question", "-o", "report.md", "-q"])
        assert args.output == "report.md"
        assert args.quiet is True


class TestDefaultApproveToolCallEofSafe:
    def test_returns_false_on_eof_instead_of_raising(self, monkeypatch):
        """Simulates running with ENABLE_HITL=true but no interactive
        stdin (cron, container with no TTY) -- must reject safely, never
        crash with an unhandled EOFError."""
        from app.__main__ import _default_approve_tool_call

        def raise_eof(prompt=""):
            raise EOFError()

        monkeypatch.setattr("builtins.input", raise_eof)
        result = _default_approve_tool_call({"tool_name": "arxiv_search", "tool_args": {"query": "x"}})
        assert result is False


class TestRunResearchHitl:
    @responses.activate
    def test_approval_resumes_and_completes(self):
        responses.add(responses.GET, ARXIV_API_URL, body=SAMPLE_ATOM, status=200)
        registry = build_default_registry()
        settings = Settings(checkpoint_backend="memory", enable_hitl=True)

        approvals = []

        def approve(pending):
            approvals.append(pending)
            return True

        report = run_research(
            "test question", settings,
            approve_tool_call=approve,
            **_default_kwargs(registry, ScriptedClient()),
        )

        assert len(approvals) == 1
        assert approvals[0]["tool_name"] == "arxiv_search"
        assert "# Research Question" in report

    def test_rejection_aborts_without_calling_tool(self):
        registry = build_default_registry()
        settings = Settings(checkpoint_backend="memory", enable_hitl=True)

        with responses.RequestsMock(assert_all_requests_are_fired=False):
            # No mock registered -- if the tool actually ran, this would
            # raise a ConnectionError.
            report = run_research(
                "test question", settings,
                approve_tool_call=lambda pending: False,
                **_default_kwargs(registry, ScriptedClient()),
            )

        assert "not approved" in report
        assert "# Research Question" not in report
