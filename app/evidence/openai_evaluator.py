"""OpenAI-backed evidence evaluation and contradiction detection.

Both use forced structured tool-calling so output is validated against a
schema rather than parsed from free text, and both defensively filter out
anything that doesn't ground back to a real, provided `paper_id` — this is
the concrete mechanism behind Step 16's "never fabricate citations" rule:
even if the model hallucinates a paper_id, it gets dropped before ever
reaching state.
"""

from __future__ import annotations

from typing import Any

from app.config import Settings, get_settings
from app.llm.client_factory import chat_openai_kwargs
from app.evidence.evaluator import ContradictionDetector, EvidenceEvaluator
from app.logging_utils import get_logger
from app.state import Contradiction, EvidenceItem

logger = get_logger(__name__)

EXTRACT_TOOL_NAME = "extract_evidence"
CONTRADICTION_TOOL_NAME = "detect_contradictions"

_VALID_RELEVANCE = {"high", "medium", "low", "irrelevant"}
_VALID_CONFIDENCE = {"strong", "moderate", "weak"}

EVIDENCE_SYSTEM_PROMPT = """You extract structured evidence for an academic \
research agent. You will be given a research question, optional \
sub-questions, and a list of papers (id, title, and either the paper's \
full text or, when full text isn't available, its abstract).

STRICT GROUNDING RULES:
- Only extract claims that are explicitly supported by the given text. \
Never infer or fabricate findings not present in the text.
- Every evidence item's "paper_id" MUST exactly match one of the provided \
paper ids.
- If a paper's text is missing, empty, or irrelevant to the research \
question, extract nothing for it — do not force an evidence item.
- A single paper may yield zero, one, or multiple evidence items if it \
supports multiple distinct claims.
- When full text is provided (not just an abstract), prefer claims from \
the paper's stated methodology, results, and limitations/discussion \
sections over introductory or motivational framing — those sections are \
where you're grounded in the paper's actual findings rather than its pitch.
- Rate "relevance" (high/medium/low/irrelevant) based on how directly the \
claim addresses the research question or a sub-question.
- Rate "confidence" (strong/moderate/weak) based on how explicit and \
unambiguous the source text's statement of the claim is — a hedged or \
vague statement should be "weak", a direct empirical result should be \
"strong".
"""

CONTRADICTION_SYSTEM_PROMPT = """You detect genuine contradictions between \
pieces of academic evidence for a research agent. You will be given a set \
of "new" claims and a set of "existing" claims, each tagged with a paper \
id. Identify pairs where a new claim and an existing claim make \
incompatible assertions about the same specific point (not just different \
topics, or different emphasis). Only flag REAL, substantive \
disagreements — if in doubt, do not flag it. Every paper_id you reference \
MUST exactly match one of the provided ids.
"""


def _extract_tool_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": EXTRACT_TOOL_NAME,
            "description": "Extract structured, grounded evidence items from the given papers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "evidence_items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "paper_id": {"type": "string"},
                                "claim": {"type": "string"},
                                "supporting_info": {"type": "string"},
                                "relevance": {
                                    "type": "string",
                                    "enum": sorted(_VALID_RELEVANCE),
                                },
                                "confidence": {
                                    "type": "string",
                                    "enum": sorted(_VALID_CONFIDENCE),
                                },
                                "related_sub_question": {"type": "string"},
                            },
                            "required": [
                                "paper_id", "claim", "supporting_info",
                                "relevance", "confidence",
                            ],
                        },
                    }
                },
                "required": ["evidence_items"],
            },
        },
    }


def _contradiction_tool_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": CONTRADICTION_TOOL_NAME,
            "description": "Report genuine contradictions between new and existing evidence claims.",
            "parameters": {
                "type": "object",
                "properties": {
                    "contradictions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "description": {"type": "string"},
                                "evidence_a_paper_id": {"type": "string"},
                                "evidence_b_paper_id": {"type": "string"},
                                "explanation": {"type": "string"},
                            },
                            "required": [
                                "description", "evidence_a_paper_id",
                                "evidence_b_paper_id", "explanation",
                            ],
                        },
                    }
                },
                "required": ["contradictions"],
            },
        },
    }


class OpenAIEvidenceEvaluator(EvidenceEvaluator):
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

    def evaluate_papers(
        self,
        papers: list[dict[str, Any]],
        research_question: str,
        sub_questions: list[str],
    ) -> list[EvidenceItem]:
        if not papers:
            return []

        valid_ids = {p.get("paper_id") for p in papers if p.get("paper_id")}
        papers_by_id = {p["paper_id"]: p for p in papers if p.get("paper_id")}

        try:
            llm = self._get_llm()
            bound = llm.bind_tools([_extract_tool_schema()], tool_choice=EXTRACT_TOOL_NAME)

            human = self._render_papers_prompt(papers, research_question, sub_questions)
            response = bound.invoke([
                {"role": "system", "content": EVIDENCE_SYSTEM_PROMPT},
                {"role": "user", "content": human},
            ])
        except Exception as exc:  # noqa: BLE001 - Step 17
            logger.warning("Evidence extraction failed, returning no evidence: %s", exc)
            return []

        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            logger.warning("Evidence evaluator did not return a tool call.")
            return []

        raw_items = tool_calls[0].get("args", {}).get("evidence_items", []) or []
        return self._parse_evidence_items(raw_items, valid_ids, papers_by_id)

    @staticmethod
    def _render_papers_prompt(
        papers: list[dict[str, Any]], research_question: str, sub_questions: list[str]
    ) -> str:
        lines = [f"Research question: {research_question}"]
        if sub_questions:
            lines.append("Sub-questions:")
            for sq in sub_questions:
                lines.append(f"  - {sq}")
        lines.append("\nPapers:")
        for p in papers:
            lines.append(f"\n[paper_id={p.get('paper_id')}]")
            lines.append(f"Title: {p.get('title', '')}")
            full_text = p.get("full_text")
            if full_text:
                note = " (truncated)" if p.get("full_text_truncated") else ""
                lines.append(f"Full text{note}:\n{full_text}")
            else:
                lines.append(f"Abstract: {p.get('abstract') or '(no abstract available)'}")
        return "\n".join(lines)

    @staticmethod
    def _parse_evidence_items(
        raw_items: list[dict[str, Any]],
        valid_ids: set[str],
        papers_by_id: dict[str, dict[str, Any]],
    ) -> list[EvidenceItem]:
        result: list[EvidenceItem] = []
        for item in raw_items:
            paper_id = item.get("paper_id")
            claim = (item.get("claim") or "").strip()
            supporting_info = (item.get("supporting_info") or "").strip()

            if not paper_id or paper_id not in valid_ids:
                logger.warning(
                    "Dropping evidence item with unrecognized paper_id=%r "
                    "(not fabricating a citation).", paper_id,
                )
                continue
            if not claim:
                continue

            relevance = item.get("relevance")
            if relevance not in _VALID_RELEVANCE:
                logger.warning("Invalid relevance %r, defaulting to 'low'.", relevance)
                relevance = "low"

            confidence = item.get("confidence")
            if confidence not in _VALID_CONFIDENCE:
                logger.warning("Invalid confidence %r, defaulting to 'weak'.", confidence)
                confidence = "weak"

            paper = papers_by_id[paper_id]
            result.append(EvidenceItem(
                paper_id=paper_id,
                paper_title=paper.get("title", ""),
                claim=claim,
                supporting_info=supporting_info,
                relevance=relevance,
                confidence=confidence,
                related_sub_question=item.get("related_sub_question") or None,
                source=paper.get("source"),
                url=paper.get("url"),
            ))
        return result


class OpenAIContradictionDetector(ContradictionDetector):
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

    def detect(
        self,
        new_evidence: list[EvidenceItem],
        existing_evidence: list[EvidenceItem],
    ) -> list[Contradiction]:
        if not new_evidence or not existing_evidence:
            return []

        valid_ids = {e.paper_id for e in new_evidence} | {e.paper_id for e in existing_evidence}

        try:
            llm = self._get_llm()
            bound = llm.bind_tools(
                [_contradiction_tool_schema()], tool_choice=CONTRADICTION_TOOL_NAME
            )
            human = self._render_claims_prompt(new_evidence, existing_evidence)
            response = bound.invoke([
                {"role": "system", "content": CONTRADICTION_SYSTEM_PROMPT},
                {"role": "user", "content": human},
            ])
        except Exception as exc:  # noqa: BLE001 - Step 17
            logger.warning("Contradiction detection failed, skipping: %s", exc)
            return []

        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            return []

        raw = tool_calls[0].get("args", {}).get("contradictions", []) or []
        return self._parse_contradictions(raw, valid_ids)

    @staticmethod
    def _render_claims_prompt(
        new_evidence: list[EvidenceItem], existing_evidence: list[EvidenceItem]
    ) -> str:
        lines = ["New claims:"]
        for e in new_evidence:
            lines.append(f"  - [paper_id={e.paper_id}] {e.claim}")
        lines.append("\nExisting claims:")
        for e in existing_evidence:
            lines.append(f"  - [paper_id={e.paper_id}] {e.claim}")
        return "\n".join(lines)

    @staticmethod
    def _parse_contradictions(
        raw: list[dict[str, Any]], valid_ids: set[str]
    ) -> list[Contradiction]:
        result: list[Contradiction] = []
        for item in raw:
            a = item.get("evidence_a_paper_id")
            b = item.get("evidence_b_paper_id")
            description = (item.get("description") or "").strip()
            explanation = (item.get("explanation") or "").strip()
            if not a or not b or a not in valid_ids or b not in valid_ids:
                logger.warning("Dropping contradiction with unrecognized paper_id(s): %r, %r", a, b)
                continue
            if not description or not explanation:
                continue
            result.append(Contradiction(
                description=description,
                evidence_a_paper_id=a,
                evidence_b_paper_id=b,
                explanation=explanation,
            ))
        return result
