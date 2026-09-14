"""OpenAI-backed ReflectionAdvisor."""

from __future__ import annotations

from typing import Any

from app.config import Settings, get_settings
from app.llm.client_factory import chat_openai_kwargs
from app.llm.prompts import render_state_summary
from app.llm.reflection import ReflectionAdvisor, ReflectionGuidance, _FALLBACK_GUIDANCE
from app.logging_utils import get_logger
from app.state import ResearchState

logger = get_logger(__name__)

PROPOSE_TOOL_NAME = "propose_strategy_change"

SYSTEM_PROMPT = """You are the reflection component of an autonomous academic \
research agent. The agent's recent searches have produced little or no \
useful evidence. Your job is to:

1. Briefly diagnose WHY the recent searches likely failed (e.g. terms too \
narrow/broad/jargon-heavy, wrong tool for this kind of question, query too \
close to the raw research question rather than a specific angle).
2. Propose a genuinely DIFFERENT next step — not a minor reword. This \
could mean substantially different terminology, a different tool, a \
narrower or broader scope, or targeting a different sub-question entirely.

Ground your diagnosis only in the search history provided below. Do not \
fabricate reasons not evidenced by the actual queries/results shown.

IMPORTANT: any suggested_query you propose must be plain keywords or a \
short natural-language phrase. Most of these academic search APIs do \
simple keyword/phrase matching, NOT boolean search — a query like \
"(A OR B) AND (C OR D)" will usually be matched as literal text and \
return zero results. If broader coverage is needed, prefer proposing \
several separate simpler queries (or switching sub-question/tool) over \
one complex boolean expression.
"""


def _propose_tool_schema(available_tools: list[str]) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "diagnosis": {
            "type": "string",
            "description": "Brief diagnosis of why recent searches were unproductive.",
        },
        "suggested_strategy_change": {
            "type": "string",
            "description": "A concrete, substantively different approach to try next.",
        },
        "suggested_query": {
            "type": "string",
            "description": "A specific example query embodying the new strategy, if applicable.",
        },
    }
    if available_tools:
        properties["suggested_tool"] = {
            "type": "string",
            "enum": available_tools,
            "description": "Which tool to try next, if switching tools is part of the suggestion.",
        }
    return {
        "type": "function",
        "function": {
            "name": PROPOSE_TOOL_NAME,
            "description": (
                "Diagnose why recent searches failed and propose a "
                "genuinely different next step."
            ),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": ["diagnosis", "suggested_strategy_change"],
            },
        },
    }


class OpenAIReflectionAdvisor(ReflectionAdvisor):
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

    def reflect(self, state: ResearchState, available_tools: list[str]) -> ReflectionGuidance:
        try:
            llm = self._get_llm()
            schema = _propose_tool_schema(available_tools)
            bound = llm.bind_tools([schema], tool_choice=PROPOSE_TOOL_NAME)

            human = render_state_summary(state)
            response = bound.invoke([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": human},
            ])
        except Exception as exc:  # noqa: BLE001 - Step 17
            logger.warning("Reflection LLM call failed, using fallback guidance: %s", exc)
            return _FALLBACK_GUIDANCE

        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            logger.warning("Reflection advisor did not return a tool call; using fallback guidance.")
            return _FALLBACK_GUIDANCE

        args = tool_calls[0].get("args", {})
        diagnosis = (args.get("diagnosis") or "").strip()
        strategy = (args.get("suggested_strategy_change") or "").strip()
        if not diagnosis or not strategy:
            logger.warning("Reflection advisor returned incomplete guidance; using fallback guidance.")
            return _FALLBACK_GUIDANCE

        suggested_tool = args.get("suggested_tool")
        if suggested_tool and suggested_tool not in available_tools:
            logger.warning("Reflection advisor suggested unknown tool %r; dropping.", suggested_tool)
            suggested_tool = None

        return ReflectionGuidance(
            diagnosis=diagnosis,
            suggested_strategy_change=strategy,
            suggested_tool=suggested_tool,
            suggested_query=args.get("suggested_query") or None,
        )
