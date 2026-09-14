"""Interface for research-question decomposition (Step 7).

Mirrors the `ReasoningClient` pattern: the planner node depends only on
this abstraction, never on a specific LLM SDK, so decomposition logic is
fully unit-testable with a fake implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class QueryDecomposer(ABC):
    """Decomposes a broad research question into specific, targeted
    sub-questions (dynamically — never a hardcoded template)."""

    @abstractmethod
    def decompose(self, research_question: str, max_sub_questions: int = 6) -> list[str]:
        """Return a list of sub-questions for `research_question`.

        Implementations must never raise for expected failure modes (LLM
        API error, timeout, malformed response) per Step 17 — return an
        empty list instead, so the agent can safely fall back to
        reasoning over the raw research question with no decomposition.
        """
        raise NotImplementedError


class NoOpDecomposer(QueryDecomposer):
    """Safe default: no decomposition. Used when no real decomposer is
    configured (e.g. tests, or a minimal run without an LLM)."""

    def decompose(self, research_question: str, max_sub_questions: int = 6) -> list[str]:
        return []
