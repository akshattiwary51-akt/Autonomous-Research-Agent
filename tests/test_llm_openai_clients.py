"""Direct coverage for every OpenAI-backed component's graceful-failure
behavior (Step 17: missing environment variables must never crash), and
for `Settings` validation itself since the hard safety bounds (Step 11)
depend on it.

Complements the component-specific parsing tests already in
test_evidence_openai_parsing.py and test_synthesis.py by covering the
remaining LLM-backed classes: OpenAIReasoningClient, OpenAIQueryDecomposer,
OpenAIReflectionAdvisor, OpenAINarrativeWriter.
"""

from __future__ import annotations

import pytest

from app.config import Settings
from app.llm.openai_client import OpenAIReasoningClient
from app.llm.openai_decomposer import OpenAIQueryDecomposer
from app.llm.openai_reflection import OpenAIReflectionAdvisor
from app.report.openai_narrative import OpenAINarrativeWriter
from app.state import EvidenceItem, create_initial_state
from app.tools.registry import build_default_registry

NO_CREDENTIALS = Settings(openai_api_key="")


class TestOpenAIReasoningClientGracefulFailure:
    def test_returns_error_decision_without_api_key(self):
        client = OpenAIReasoningClient(settings=NO_CREDENTIALS)
        state = create_initial_state("test question")
        registry = build_default_registry()

        decision = client.decide_next_action(state, registry.schemas())

        assert decision.action == "error"
        assert "OPENAI_API_KEY" in decision.rationale_summary


class TestOpenAIQueryDecomposerGracefulFailure:
    def test_returns_empty_list_without_api_key(self):
        decomposer = OpenAIQueryDecomposer(settings=NO_CREDENTIALS)
        result = decomposer.decompose("test research question")
        assert result == []


class TestOpenAIReflectionAdvisorGracefulFailure:
    def test_returns_fallback_guidance_without_api_key(self):
        advisor = OpenAIReflectionAdvisor(settings=NO_CREDENTIALS)
        state = create_initial_state("test question")
        guidance = advisor.reflect(state, ["arxiv_search", "semantic_scholar_search"])

        assert guidance.diagnosis  # non-empty fallback, not a crash
        assert guidance.suggested_strategy_change


class TestOpenAINarrativeWriterGracefulFailure:
    def test_returns_deterministic_fallback_without_api_key(self):
        writer = OpenAINarrativeWriter(settings=NO_CREDENTIALS)
        evidence = [EvidenceItem(
            paper_id="p1", paper_title="T", claim="c", supporting_info="s",
            relevance="high", confidence="strong",
        )]
        sections = writer.write(
            research_question="q", evidence=evidence, contradictions=[], gaps=[],
            iterations_run=1, max_iterations=6,
        )
        # Should match what NoOpNarrativeWriter would produce for the same input
        assert "1 structured" in sections.executive_summary

    def test_returns_fallback_without_llm_call_for_empty_evidence(self):
        writer = OpenAINarrativeWriter(settings=NO_CREDENTIALS)
        sections = writer.write(
            research_question="q", evidence=[], contradictions=[], gaps=[],
            iterations_run=0, max_iterations=6,
        )
        assert "No evidence was collected" in sections.comparative_analysis


class TestSettingsValidation:
    def test_defaults_are_safe_and_sane(self):
        settings = Settings()
        assert settings.has_llm_credentials is False
        assert settings.max_iterations == 6
        assert settings.max_tool_calls == 12
        assert settings.zero_yield_reflection_threshold == 2
        assert settings.max_reflection_attempts == 2
        assert settings.checkpoint_backend == "memory"
        assert settings.enable_hitl is False

    def test_has_llm_credentials_true_when_key_set(self):
        settings = Settings(openai_api_key="sk-fake-for-testing")
        assert settings.has_llm_credentials is True

    def test_max_iterations_out_of_bounds_rejected(self):
        with pytest.raises(Exception):
            Settings(max_iterations=0)
        with pytest.raises(Exception):
            Settings(max_iterations=51)

    def test_invalid_checkpoint_backend_rejected(self):
        with pytest.raises(Exception):
            Settings(checkpoint_backend="not_a_real_backend")

    def test_invalid_log_level_rejected(self):
        with pytest.raises(Exception):
            Settings(log_level="NOT_A_LEVEL")
