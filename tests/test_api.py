"""Tests for app/api/* -- the optional HTTP wrapper around run_research_state.

Uses FastAPI's TestClient with fake components injected via a dependency
override, so no live LLM or network call happens in this suite -- same
principle as every other phase's tests.
"""

from __future__ import annotations

import time

import responses
from fastapi.testclient import TestClient

from app.api.jobs import JobStore
from app.api.main import app, get_job_store
from app.config import Settings
from app.llm.base import AgentDecision, ReasoningClient
from app.llm.decomposition import NoOpDecomposer
from app.evidence.evaluator import NoOpContradictionDetector, NoOpEvidenceEvaluator
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


def _fake_components(client=None):
    return dict(
        registry=build_default_registry(),
        reasoning_client=client or ScriptedClient(),
        decomposer=NoOpDecomposer(),
        evidence_evaluator=NoOpEvidenceEvaluator(),
        contradiction_detector=NoOpContradictionDetector(),
        reflection_advisor=NoOpReflectionAdvisor(),
        gap_discoverer=NoOpGapDiscoverer(),
        narrative_writer=NoOpNarrativeWriter(),
    )


class FixedComponentsJobStore(JobStore):
    """Test double: always uses the injected fake components, regardless
    of what settings/components the caller (the HTTP layer) requests --
    so no test ever constructs a real OpenAI-backed client."""

    def __init__(self, components: dict, settings: Settings | None = None):
        super().__init__()
        self._fixed_components = components
        self._fixed_settings = settings or Settings(checkpoint_backend="memory")

    def create(self, question, settings=None, max_iterations=None, components=None):
        effective_settings = self._fixed_settings
        if max_iterations is not None:
            effective_settings = effective_settings.model_copy(update={"max_iterations": max_iterations})
        return super().create(
            question, settings=effective_settings, components=self._fixed_components,
        )


def _wait_for_status(client: TestClient, job_id: str, target_statuses: set[str], timeout: float = 5.0) -> dict:
    """Polls GET /research/{job_id} until status is in target_statuses or
    timeout elapses. Real research runs are fast here since every
    component is a fake/NoOp with no real LLM latency."""
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        resp = client.get(f"/research/{job_id}")
        last = resp.json()
        if last["status"] in target_statuses:
            return last
        time.sleep(0.02)
    raise AssertionError(f"Timed out waiting for status in {target_statuses}; last seen: {last}")


class TestHealthCheck:
    def test_health_endpoint(self):
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestResearchJobLifecycleNonHitl:
    @responses.activate
    def test_full_job_lifecycle_via_polling(self):
        responses.add(responses.GET, ARXIV_API_URL, body=SAMPLE_ATOM, status=200)

        store = FixedComponentsJobStore(_fake_components())
        app.dependency_overrides[get_job_store] = lambda: store
        client = TestClient(app)
        try:
            start_resp = client.post("/research", json={"question": "test question"})
            assert start_resp.status_code == 202
            body = start_resp.json()
            job_id = body["job_id"]
            assert body["status"] in ("queued", "running")
            assert body["question"] == "test question"

            final = _wait_for_status(client, job_id, {"done", "error", "aborted"})

            assert final["status"] == "done"
            assert final["report"] is not None
            assert "# Research Question" in final["report"]
            assert final["papers_retrieved"] == 1
            assert len(final["progress"]) > 0
        finally:
            app.dependency_overrides.clear()

    def test_unknown_job_id_returns_404(self):
        client = TestClient(app)
        resp = client.get("/research/does-not-exist")
        assert resp.status_code == 404

    def test_max_iterations_override_respected(self):
        class NeverConvergesClient(ReasoningClient):
            def decide_next_action(self, state, tool_schemas):
                return AgentDecision(action="call_tool", tool_name="arxiv_search", tool_args={"query": "x"})

        with responses.RequestsMock() as rsps:
            rsps.add(responses.GET, ARXIV_API_URL, json={"error": "unused"}, status=500)

            store = FixedComponentsJobStore(_fake_components(NeverConvergesClient()))
            app.dependency_overrides[get_job_store] = lambda: store
            client = TestClient(app)
            try:
                resp = client.post("/research", json={"question": "test question", "max_iterations": 2})
                job_id = resp.json()["job_id"]
                final = _wait_for_status(client, job_id, {"done", "error", "aborted"})
                assert final["iteration"] == 2
                assert final["max_iterations"] == 2
            finally:
                app.dependency_overrides.clear()

    def test_invalid_request_body_rejected_with_422(self):
        client = TestClient(app)
        resp = client.post("/research", json={"question": ""})  # min_length=1
        assert resp.status_code == 422

    def test_max_iterations_out_of_bounds_rejected_with_422(self):
        client = TestClient(app)
        resp = client.post("/research", json={"question": "x", "max_iterations": 0})
        assert resp.status_code == 422


class TestResearchJobHitl:
    @responses.activate
    def test_approval_flow_resumes_and_completes(self):
        responses.add(responses.GET, ARXIV_API_URL, body=SAMPLE_ATOM, status=200)

        settings = Settings(checkpoint_backend="memory", enable_hitl=True)
        store = FixedComponentsJobStore(_fake_components(), settings=settings)
        app.dependency_overrides[get_job_store] = lambda: store
        client = TestClient(app)
        try:
            start_resp = client.post("/research", json={"question": "test question"})
            job_id = start_resp.json()["job_id"]

            paused = _wait_for_status(client, job_id, {"awaiting_approval"})
            assert paused["pending_approval"]["tool_name"] == "arxiv_search"

            approve_resp = client.post(f"/research/{job_id}/approve", json={"approve": True})
            assert approve_resp.status_code == 200

            final = _wait_for_status(client, job_id, {"done", "error", "aborted"})
            assert final["status"] == "done"
            assert final["report"] is not None
        finally:
            app.dependency_overrides.clear()

    def test_rejection_aborts_without_calling_tool(self):
        settings = Settings(checkpoint_backend="memory", enable_hitl=True)
        store = FixedComponentsJobStore(_fake_components(), settings=settings)
        app.dependency_overrides[get_job_store] = lambda: store
        client = TestClient(app)
        try:
            with responses.RequestsMock(assert_all_requests_are_fired=False):
                # No mock registered -- if the tool actually ran this
                # would raise a ConnectionError inside the worker thread.
                start_resp = client.post("/research", json={"question": "test question"})
                job_id = start_resp.json()["job_id"]

                paused = _wait_for_status(client, job_id, {"awaiting_approval"})
                assert paused["status"] == "awaiting_approval"

                client.post(f"/research/{job_id}/approve", json={"approve": False})

                final = _wait_for_status(client, job_id, {"done", "error", "aborted"})
                assert final["status"] == "aborted"
                assert "not approved" in final["report"]
        finally:
            app.dependency_overrides.clear()

    def test_approving_a_job_not_awaiting_approval_returns_409(self):
        store = FixedComponentsJobStore(_fake_components())  # HITL disabled
        app.dependency_overrides[get_job_store] = lambda: store
        client = TestClient(app)
        try:
            with responses.RequestsMock() as rsps:
                rsps.add(responses.GET, ARXIV_API_URL, body=SAMPLE_ATOM, status=200)
                start_resp = client.post("/research", json={"question": "test question"})
                job_id = start_resp.json()["job_id"]
                _wait_for_status(client, job_id, {"done", "error", "aborted"})

            resp = client.post(f"/research/{job_id}/approve", json={"approve": True})
            assert resp.status_code == 409
        finally:
            app.dependency_overrides.clear()

    def test_approve_unknown_job_id_returns_404(self):
        client = TestClient(app)
        resp = client.post("/research/does-not-exist/approve", json={"approve": True})
        assert resp.status_code == 404
