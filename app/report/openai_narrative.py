"""OpenAI-backed grounded NarrativeWriter.

The model is given exactly the evidence/contradictions/gaps already
collected and instructed to write prose using only that material. As a
concrete enforcement mechanism (not just a prompt instruction), every
generated section is scanned for arXiv-ID-like and DOI-like substrings;
if any identifier appears that isn't one of the actually-known paper_ids,
that ENTIRE section is discarded and replaced with the deterministic
`NoOpNarrativeWriter`'s version instead — this is the concrete grounding
guard referenced in Step 16 ("never fabricate citations").
"""

from __future__ import annotations

import re
from typing import Any

from app.config import Settings, get_settings
from app.llm.client_factory import chat_openai_kwargs
from app.logging_utils import get_logger
from app.report.narrative import NarrativeSections, NarrativeWriter, NoOpNarrativeWriter
from app.state import Contradiction, EvidenceItem, ResearchGap

logger = get_logger(__name__)

WRITE_TOOL_NAME = "write_narrative_sections"

# arXiv ids look like 2401.12345 or 2401.12345v2; DOIs look like 10.1234/...
_ARXIV_ID_RE = re.compile(r"\b\d{4}\.\d{4,5}(v\d+)?\b")
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b")

SYSTEM_PROMPT = """You write the narrative sections of an academic \
research report for a research agent. You will be given the EXACT set of \
evidence, contradictions, and research gaps already collected — this is \
the ONLY material you may draw on.

STRICT RULES:
- Refer to papers only by the titles given to you. Never invent a paper \
title, author, DOI, arXiv id, year, or finding that isn't in the \
provided material.
- Do not state any specific fact, statistic, or claim that isn't directly \
present in the evidence given.
- If the evidence is sparse, write a shorter, more modest section rather \
than padding with generic filler or invented specifics.
- Write in clear, professional academic-report prose.
"""


def _write_tool_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": WRITE_TOOL_NAME,
            "description": "Write the four narrative sections of the research report.",
            "parameters": {
                "type": "object",
                "properties": {
                    "executive_summary": {"type": "string"},
                    "comparative_analysis": {"type": "string"},
                    "limitations": {"type": "string"},
                    "conclusion": {"type": "string"},
                },
                "required": [
                    "executive_summary", "comparative_analysis",
                    "limitations", "conclusion",
                ],
            },
        },
    }


def guard_section(text: str, known_ids: set[str], section_name: str, fallback: str) -> str:
    """Concrete anti-fabrication check: reject the whole section if it
    contains an identifier-shaped substring not in `known_ids`."""
    for pattern in (_ARXIV_ID_RE, _DOI_RE):
        for match in pattern.finditer(text):
            candidate = match.group(0)
            if candidate not in known_ids:
                logger.warning(
                    "Narrative section %r contained an unrecognized identifier "
                    "%r not among the known paper ids; discarding LLM text and "
                    "using the deterministic fallback for this section.",
                    section_name, candidate,
                )
                return fallback
    return text


class OpenAINarrativeWriter(NarrativeWriter):
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._llm = None
        self._fallback = NoOpNarrativeWriter()

    def _get_llm(self):
        if self._llm is None:
            from langchain_openai import ChatOpenAI

            if not self._settings.has_llm_credentials:
                raise RuntimeError(
                    "OPENAI_API_KEY is not configured; cannot make live LLM calls."
                )
            self._llm = ChatOpenAI(**chat_openai_kwargs(self._settings))
        return self._llm

    def write(
        self,
        research_question: str,
        evidence: list[EvidenceItem],
        contradictions: list[Contradiction],
        gaps: list[ResearchGap],
        iterations_run: int,
        max_iterations: int,
    ) -> NarrativeSections:
        fallback = self._fallback.write(
            research_question, evidence, contradictions, gaps, iterations_run, max_iterations,
        )

        if not evidence:
            return fallback  # nothing to write about; avoid an unnecessary LLM call

        known_ids = {e.paper_id for e in evidence}

        try:
            llm = self._get_llm()
            bound = llm.bind_tools([_write_tool_schema()], tool_choice=WRITE_TOOL_NAME)
            human = self._render_prompt(research_question, evidence, contradictions, gaps)
            response = bound.invoke([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": human},
            ])
        except Exception as exc:  # noqa: BLE001 - Step 17
            logger.warning("Narrative writing failed, using deterministic fallback: %s", exc)
            return fallback

        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            logger.warning("Narrative writer did not return a tool call; using fallback.")
            return fallback

        args = tool_calls[0].get("args", {})

        return NarrativeSections(
            executive_summary=guard_section(
                args.get("executive_summary") or fallback.executive_summary,
                known_ids, "executive_summary", fallback.executive_summary,
            ),
            comparative_analysis=guard_section(
                args.get("comparative_analysis") or fallback.comparative_analysis,
                known_ids, "comparative_analysis", fallback.comparative_analysis,
            ),
            limitations=guard_section(
                args.get("limitations") or fallback.limitations,
                known_ids, "limitations", fallback.limitations,
            ),
            conclusion=guard_section(
                args.get("conclusion") or fallback.conclusion,
                known_ids, "conclusion", fallback.conclusion,
            ),
        )

    @staticmethod
    def _render_prompt(
        research_question: str,
        evidence: list[EvidenceItem],
        contradictions: list[Contradiction],
        gaps: list[ResearchGap],
    ) -> str:
        lines = [f"Research question: {research_question}", "\nEvidence:"]
        for e in evidence:
            lines.append(f"  - \"{e.paper_title}\" ({e.relevance}/{e.confidence}): {e.claim}")
        if contradictions:
            lines.append("\nContradictions:")
            for c in contradictions:
                lines.append(f"  - {c.description}: {c.explanation}")
        if gaps:
            lines.append("\nIdentified research gaps:")
            for g in gaps:
                lines.append(f"  - {g.statement} ({g.significance})")
        return "\n".join(lines)
