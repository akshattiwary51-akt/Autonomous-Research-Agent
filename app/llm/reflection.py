"""Interface for reflection-driven strategy-change reasoning (Step 11.C).

Mirrors the `ReasoningClient`/`QueryDecomposer` pattern.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Optional

from app.state import ResearchState


@dataclass
class ReflectionGuidance:
    """Safe, structured output of a reflection pass — never raw
    chain-of-thought (Step 6)."""

    diagnosis: str
    suggested_strategy_change: str
    suggested_tool: Optional[str] = None
    suggested_query: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class ReflectionAdvisor(ABC):
    """Diagnoses why recent searches were unproductive and proposes a
    genuinely different next step — not a minor reword of the same query."""

    @abstractmethod
    def reflect(self, state: ResearchState, available_tools: list[str]) -> ReflectionGuidance:
        """Must never raise for expected failure modes (LLM error, malformed
        response) per Step 17 — fall back to generic-but-still-useful
        guidance instead of failing the whole reflection step."""
        raise NotImplementedError


_FALLBACK_GUIDANCE = ReflectionGuidance(
    diagnosis="Automatic fallback: recent searches produced no useful results.",
    suggested_strategy_change=(
        "Try substantially different terminology, broaden or narrow the "
        "scope, or switch to a different research tool entirely — a minor "
        "reword of the same query is unlikely to help."
    ),
)


class NoOpReflectionAdvisor(ReflectionAdvisor):
    """Safe default: generic (but still actionable) guidance, no LLM call."""

    def reflect(self, state: ResearchState, available_tools: list[str]) -> ReflectionGuidance:
        return _FALLBACK_GUIDANCE
