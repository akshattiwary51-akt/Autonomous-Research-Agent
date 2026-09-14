"""OpenAI-backed GapDiscoverer."""

from __future__ import annotations

from typing import Any

from app.config import Settings, get_settings
from app.llm.client_factory import chat_openai_kwargs
from app.logging_utils import get_logger
from app.report.gap_discovery import GapDiscoverer
from app.state import Contradiction, EvidenceItem, ResearchGap

logger = get_logger(__name__)

PROPOSE_GAPS_TOOL_NAME = "propose_research_gaps"

SYSTEM_PROMPT = """You identify research gaps for an academic research \
agent, grounded STRICTLY in the evidence provided below — never generic \
statements like "more research is needed" that aren't tied to specific \
evidence.

A valid research gap must be connected to one or more of these patterns \
actually present in the evidence:
- A limitation explicitly reported by a paper.
- A contradiction between papers that remains unresolved.
- An underexplored area implied by the set of claims (e.g. all evidence \
covers one narrow setting/dataset/scale).
- A missing evaluation, dataset limitation, methodology limitation, or \
scalability issue mentioned in the evidence.

For each gap:
- State the gap specifically and concretely.
- Reference the exact paper_id(s) whose evidence supports identifying \
this gap. Every id you use MUST exactly match one of the provided ids.
- Explain briefly why it matters (significance).

Do not propose more gaps than the evidence can genuinely support. If the \
evidence is too thin or homogeneous to support any real gap, return an \
empty list rather than inventing one.
"""


def _propose_gaps_tool_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": PROPOSE_GAPS_TOOL_NAME,
            "description": "Propose evidence-grounded research gaps.",
            "parameters": {
                "type": "object",
                "properties": {
                    "gaps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "statement": {"type": "string"},
                                "supporting_evidence_paper_ids": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "significance": {"type": "string"},
                            },
                            "required": ["statement", "supporting_evidence_paper_ids", "significance"],
                        },
                    }
                },
                "required": ["gaps"],
            },
        },
    }


class OpenAIGapDiscoverer(GapDiscoverer):
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            from langchain_openai import ChatOpenAI

            if not self._settings.has_llm_credentials:
                raise RuntimeError(
                    "OPENAI_API_KEY is not configured; cannot make live LLM calls."
                )
            self._llm = ChatOpenAI(**chat_openai_kwargs(self._settings))
        return self._llm

    def discover_gaps(
        self,
        evidence: list[EvidenceItem],
        contradictions: list[Contradiction],
        research_question: str,
    ) -> list[ResearchGap]:
        if not evidence:
            return []

        valid_ids = {e.paper_id for e in evidence}

        try:
            llm = self._get_llm()
            bound = llm.bind_tools(
                [_propose_gaps_tool_schema()], tool_choice=PROPOSE_GAPS_TOOL_NAME
            )
            human = self._render_prompt(evidence, contradictions, research_question)
            response = bound.invoke([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": human},
            ])
        except Exception as exc:  # noqa: BLE001 - Step 17
            logger.warning("Gap discovery failed, returning no gaps: %s", exc)
            return []

        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            return []

        raw_gaps = tool_calls[0].get("args", {}).get("gaps", []) or []
        return self._parse_gaps(raw_gaps, valid_ids)

    @staticmethod
    def _render_prompt(
        evidence: list[EvidenceItem], contradictions: list[Contradiction], research_question: str
    ) -> str:
        lines = [f"Research question: {research_question}", "\nEvidence:"]
        for e in evidence:
            lines.append(
                f"  - [paper_id={e.paper_id}] ({e.relevance}/{e.confidence}) "
                f"{e.paper_title}: {e.claim}"
            )
        if contradictions:
            lines.append("\nContradictions:")
            for c in contradictions:
                lines.append(
                    f"  - {c.description} ({c.evidence_a_paper_id} vs "
                    f"{c.evidence_b_paper_id}): {c.explanation}"
                )
        return "\n".join(lines)

    @staticmethod
    def _parse_gaps(raw_gaps: list[dict[str, Any]], valid_ids: set[str]) -> list[ResearchGap]:
        result: list[ResearchGap] = []
        for item in raw_gaps:
            statement = (item.get("statement") or "").strip()
            significance = (item.get("significance") or "").strip()
            supporting_ids_raw = item.get("supporting_evidence_paper_ids") or []
            supporting_ids = [pid for pid in supporting_ids_raw if pid in valid_ids]

            if not statement or not significance:
                continue
            if not supporting_ids:
                logger.warning(
                    "Dropping proposed gap with no valid supporting evidence "
                    "(all referenced paper_ids were unrecognized): %r", statement,
                )
                continue

            result.append(ResearchGap(
                statement=statement,
                supporting_evidence_paper_ids=supporting_ids,
                significance=significance,
                is_inference=True,
            ))
        return result
