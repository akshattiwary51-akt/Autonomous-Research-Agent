"""Planner node — decomposes the research question into sub-questions
(Step 7).

Depends only on the `QueryDecomposer` interface, so it's testable without
a live LLM (see `NoOpDecomposer` and the fake used in tests). Never
hardcodes a fixed decomposition template — the actual sub-questions come
from whatever `QueryDecomposer` implementation is injected.
"""

from __future__ import annotations

from typing import Any, Callable

from app.llm.decomposition import QueryDecomposer
from app.logging_utils import get_logger
from app.state import ResearchState

logger = get_logger(__name__)

DEFAULT_MAX_SUB_QUESTIONS = 6


def build_planner_node(
    decomposer: QueryDecomposer,
    max_sub_questions: int = DEFAULT_MAX_SUB_QUESTIONS,
) -> Callable[[ResearchState], dict[str, Any]]:
    def planner_node(state: ResearchState) -> dict[str, Any]:
        research_question = state.get("research_question", "")
        sub_questions = decomposer.decompose(research_question, max_sub_questions)

        if sub_questions:
            logger.info("Planner decomposed question into %d sub-questions.", len(sub_questions))
            note = f"Decomposed into {len(sub_questions)} sub-questions."
        else:
            logger.info("Planner produced no decomposition; agent will reason over the raw question.")
            note = "No decomposition available; proceeding with the raw research question."

        return {
            "sub_questions": sub_questions,
            "status": "researching",
            "scratchpad": [{"stage": "planner", "note": note}],
        }

    return planner_node
