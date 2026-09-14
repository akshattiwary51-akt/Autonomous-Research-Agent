"""OpenAI-backed QueryDecomposer.

Uses LangChain's `ChatOpenAI` + a single forced tool call to get a
structured list of sub-questions back from the model, rather than parsing
free-text output.
"""

from __future__ import annotations

from typing import Any

from app.config import Settings, get_settings
from app.llm.client_factory import chat_openai_kwargs
from app.llm.decomposition import QueryDecomposer
from app.logging_utils import get_logger

logger = get_logger(__name__)

PROPOSE_TOOL_NAME = "propose_sub_questions"

SYSTEM_PROMPT = """You are the research-planning component of an autonomous \
academic research agent. Given a broad research question, decompose it \
into a set of specific, targeted sub-questions that together would let \
someone thoroughly investigate and answer the main question.

Guidelines:
- Each sub-question should be narrow enough to search for directly on \
academic databases (e.g. "What multimodal RAG approaches have been \
proposed for scientific documents?" not just "multimodal RAG").
- Cover different angles: existing approaches/methods, how they work, \
reported limitations/failure modes, datasets/benchmarks used, and open \
problems — but only include angles that are actually relevant to the \
specific question asked; do not force a fixed template onto every topic.
- Do not include a sub-question that just repeats the main question \
verbatim.
- Produce between 3 and {max_sub_questions} sub-questions.
"""


def _propose_tool_schema(max_sub_questions: int) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": PROPOSE_TOOL_NAME,
            "description": (
                "Propose a dynamic set of specific, targeted sub-questions "
                "for investigating the given research question."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sub_questions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            f"3 to {max_sub_questions} specific sub-questions, "
                            "each narrow enough to search for directly."
                        ),
                    }
                },
                "required": ["sub_questions"],
            },
        },
    }


class OpenAIQueryDecomposer(QueryDecomposer):
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            from langchain_openai import ChatOpenAI

            if not self._settings.has_llm_credentials:
                raise RuntimeError(
                    "OPENAI_API_KEY is not configured; cannot make live LLM calls. "
                    "Set it in your environment or .env file."
                )
            self._llm = ChatOpenAI(**chat_openai_kwargs(self._settings))
        return self._llm

    def decompose(self, research_question: str, max_sub_questions: int = 6) -> list[str]:
        try:
            llm = self._get_llm()
            schema = _propose_tool_schema(max_sub_questions)
            bound = llm.bind_tools([schema], tool_choice=PROPOSE_TOOL_NAME)

            response = bound.invoke(
                [
                    {"role": "system", "content": SYSTEM_PROMPT.format(max_sub_questions=max_sub_questions)},
                    {"role": "user", "content": f"Research question: {research_question}"},
                ]
            )
        except Exception as exc:  # noqa: BLE001 - deliberate broad catch per Step 17
            logger.warning("Query decomposition failed, falling back to no decomposition: %s", exc)
            return []

        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            logger.warning("Decomposer did not return a tool call; falling back to no decomposition.")
            return []

        raw = tool_calls[0].get("args", {}).get("sub_questions") or []
        sub_questions = [str(q).strip() for q in raw if str(q).strip()]
        return sub_questions[:max_sub_questions]
