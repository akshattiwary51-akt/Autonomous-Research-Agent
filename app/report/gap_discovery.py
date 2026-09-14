"""Interface for academic gap discovery (Step 10).

Every discovered gap must be traceable to actual collected evidence — no
generic "more research is needed" statements. `ResearchGap.is_inference`
(defined in `app.state`) defaults to `True` and this subsystem never sets
it to `False`: everything a `GapDiscoverer` produces is, by construction,
the agent's own inference from patterns across evidence — never a claim a
paper explicitly made about itself — so it must always be labeled as such
(Step 16).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.state import Contradiction, EvidenceItem, ResearchGap


class GapDiscoverer(ABC):
    @abstractmethod
    def discover_gaps(
        self,
        evidence: list[EvidenceItem],
        contradictions: list[Contradiction],
        research_question: str,
    ) -> list[ResearchGap]:
        """Must never raise for expected failure modes (LLM error,
        malformed response) — return an empty list instead (Step 17)."""
        raise NotImplementedError


class NoOpGapDiscoverer(GapDiscoverer):
    def discover_gaps(
        self,
        evidence: list[EvidenceItem],
        contradictions: list[Contradiction],
        research_question: str,
    ) -> list[ResearchGap]:
        return []
