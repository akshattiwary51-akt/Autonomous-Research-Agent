"""Interfaces for turning raw retrieved papers into structured evidence
(Step 9), and for detecting contradictions between evidence items.

Mirrors the `ReasoningClient`/`QueryDecomposer` pattern: nodes depend only
on these abstractions, never a specific LLM SDK.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.state import Contradiction, EvidenceItem


class EvidenceEvaluator(ABC):
    """Extracts structured evidence items from raw paper metadata.

    Implementations must ground every extracted claim in the paper's own
    abstract/title — never fabricate findings (Step 16) — and must never
    raise for expected failure modes (LLM error, malformed response);
    return an empty list instead (Step 17).
    """

    @abstractmethod
    def evaluate_papers(
        self,
        papers: list[dict[str, Any]],
        research_question: str,
        sub_questions: list[str],
    ) -> list[EvidenceItem]:
        raise NotImplementedError


class NoOpEvidenceEvaluator(EvidenceEvaluator):
    """Safe default: extracts no structured evidence. Used when no real
    evaluator is configured (tests, or a minimal run without an LLM)."""

    def evaluate_papers(
        self,
        papers: list[dict[str, Any]],
        research_question: str,
        sub_questions: list[str],
    ) -> list[EvidenceItem]:
        return []


class ContradictionDetector(ABC):
    """Detects conflicts between newly extracted evidence and evidence
    already collected, so the agent (and eventually the report) can
    distinguish agreement from disagreement across papers (Step 9)."""

    @abstractmethod
    def detect(
        self,
        new_evidence: list[EvidenceItem],
        existing_evidence: list[EvidenceItem],
    ) -> list[Contradiction]:
        raise NotImplementedError


class NoOpContradictionDetector(ContradictionDetector):
    def detect(
        self,
        new_evidence: list[EvidenceItem],
        existing_evidence: list[EvidenceItem],
    ) -> list[Contradiction]:
        return []
