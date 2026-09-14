"""Tests for Step 13 (checkpointing/resume) and Step 14 (human-in-the-loop)."""

from __future__ import annotations

import os
import tempfile

import pytest
import responses

from app.checkpointing import build_checkpointer
from app.config import Settings
from app.graph.build_graph import build_research_graph
from app.llm.base import AgentDecision, ReasoningClient
from app.state import create_initial_state
from app.tools.arxiv_tool import ARXIV_API_URL
from app.tools.registry import build_default_registry

SAMPLE_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.12345v1</id>
    <title>Test Paper</title>
    <summary>abstract</summary>
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


# --------------------------------------------------------------------------
# Checkpointer factory
# --------------------------------------------------------------------------


class TestBuildCheckpointer:
    def test_memory_backend_returns_memory_saver(self):
        from langgraph.checkpoint.memory import MemorySaver
        settings = Settings(checkpoint_backend="memory")
        checkpointer = build_checkpointer(settings)
        assert isinstance(checkpointer, MemorySaver)

    def test_sqlite_backend_returns_sqlite_saver(self):
        from langgraph.checkpoint.sqlite import SqliteSaver
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "cp.sqlite")
            settings = Settings(checkpoint_backend="sqlite", checkpoint_db_path=db_path)
            checkpointer = build_checkpointer(settings)
            assert isinstance(checkpointer, SqliteSaver)

    def test_unknown_backend_raises_clear_error(self):
        settings = Settings.model_construct(checkpoint_backend="postgres", checkpoint_db_path="x")
        with pytest.raises(ValueError, match="Unknown checkpoint backend"):
            build_checkpointer(settings)


# --------------------------------------------------------------------------
# Checkpointing / resume behavior (Step 13)
# --------------------------------------------------------------------------


class TestCheckpointingPersistsAndResumes:
    @responses.activate
    def test_memory_checkpointer_retains_state_history_within_process(self):
        responses.add(responses.GET, ARXIV_API_URL, body=SAMPLE_ATOM, status=200)
        settings = Settings(checkpoint_backend="memory")
        checkpointer = build_checkpointer(settings)
        registry = build_default_registry()
        app = build_research_graph(registry, ScriptedClient(), checkpointer=checkpointer)

        config = {"configurable": {"thread_id": "t1"}, "recursion_limit": 50}
        result = app.invoke(create_initial_state("test question", max_iterations=6), config=config)

        assert result["status"] == "done"
        history = list(app.get_state_history(config))
        assert len(history) > 1  # multiple checkpoints recorded across nodes

    @responses.activate
    def test_sqlite_checkpointer_survives_a_fresh_app_instance(self):
        """Simulates a process restart: a brand-new checkpointer/app object
        pointed at the same DB file must recover the prior run's state."""
        responses.add(responses.GET, ARXIV_API_URL, body=SAMPLE_ATOM, status=200)
        registry = build_default_registry()

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "cp.sqlite")
            settings = Settings(checkpoint_backend="sqlite", checkpoint_db_path=db_path)
            config = {"configurable": {"thread_id": "t2"}, "recursion_limit": 50}

            checkpointer_a = build_checkpointer(settings)
            app_a = build_research_graph(registry, ScriptedClient(), checkpointer=checkpointer_a)
            result = app_a.invoke(create_initial_state("persisted question", max_iterations=6), config=config)
            assert result["status"] == "done"

            # Fresh objects, same file — simulates a new process
            checkpointer_b = build_checkpointer(settings)
            app_b = build_research_graph(registry, ScriptedClient(), checkpointer=checkpointer_b)
            recovered = app_b.get_state(config)

            assert recovered.values.get("research_question") == "persisted question"
            assert recovered.values.get("status") == "done"


# --------------------------------------------------------------------------
# HITL (Step 14)
# --------------------------------------------------------------------------


class TestHumanInTheLoop:
    def test_hitl_without_checkpointer_raises_immediately(self):
        registry = build_default_registry()
        with pytest.raises(ValueError, match="requires a checkpointer"):
            build_research_graph(registry, ScriptedClient(), enable_hitl=True)

    def test_hitl_disabled_by_default(self):
        """Default behavior (no enable_hitl passed) must NOT pause — this
        guards against accidentally changing the default for all existing
        callers/tests."""
        registry = build_default_registry()
        checkpointer = build_checkpointer(Settings(checkpoint_backend="memory"))
        app = build_research_graph(registry, ScriptedClient(), checkpointer=checkpointer)
        # compiled graph with no interrupt configured
        assert not getattr(app, "interrupt_before_nodes", None)

    @responses.activate
    def test_hitl_pauses_before_tool_execution_and_resumes_on_approval(self):
        responses.add(responses.GET, ARXIV_API_URL, body=SAMPLE_ATOM, status=200)
        registry = build_default_registry()
        checkpointer = build_checkpointer(Settings(checkpoint_backend="memory"))
        app = build_research_graph(
            registry, ScriptedClient(), checkpointer=checkpointer, enable_hitl=True,
        )
        config = {"configurable": {"thread_id": "hitl-1"}, "recursion_limit": 50}

        # First invoke: planner -> agent, then PAUSES before the tool node.
        result = app.invoke(create_initial_state("test question", max_iterations=6), config=config)

        assert result.get("final_report") is None  # run did not complete
        state = app.get_state(config)
        assert state.next == ("tool",)
        assert state.values["pending_action"]["action"] == "call_tool"
        assert state.values["pending_action"]["tool_name"] == "arxiv_search"

        # Human approves: resume with no new input, same thread_id.
        resumed = app.invoke(None, config=config)

        assert resumed["status"] == "done"
        assert resumed["final_report"] is not None
        assert len(resumed["retrieved_papers"]) == 1

    @responses.activate
    def test_hitl_run_can_be_abandoned_without_calling_the_tool(self):
        """A human REJECTING the action just means the caller never
        resumes the graph — confirm the tool genuinely wasn't called by
        checking no papers were ever recorded, using a mock that would
        error on any unexpected request."""
        registry = build_default_registry()
        checkpointer = build_checkpointer(Settings(checkpoint_backend="memory"))
        app = build_research_graph(
            registry, ScriptedClient(), checkpointer=checkpointer, enable_hitl=True,
        )
        config = {"configurable": {"thread_id": "hitl-2"}, "recursion_limit": 50}

        with responses.RequestsMock(assert_all_requests_are_fired=False):
            # No mock registered for ARXIV_API_URL — if the tool node ran,
            # this would raise a ConnectionError.
            result = app.invoke(create_initial_state("test question", max_iterations=6), config=config)

        assert result.get("final_report") is None
        assert result.get("retrieved_papers", []) == []
