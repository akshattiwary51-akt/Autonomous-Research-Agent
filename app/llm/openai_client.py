"""OpenAI-backed ReasoningClient.

Uses LangChain's `ChatOpenAI` + `bind_tools` so the model chooses a tool
(or explicitly signals it's done) via structured function-calling rather
than free-text parsing.
"""

from __future__ import annotations

from typing import Any

from app.config import Settings, get_settings
from app.llm.client_factory import chat_openai_kwargs
from app.llm.base import AgentDecision, ReasoningClient
from app.llm.prompts import render_state_summary
from app.logging_utils import get_logger
from app.state import ResearchState

logger = get_logger(__name__)

FINISH_TOOL_NAME = "finish_research"

SYSTEM_PROMPT = """You are the reasoning component of an autonomous academic \
research agent. Your job each turn is to decide ONE next action:

1. Call exactly one research tool to gather more evidence, OR
2. Call the "{finish_tool}" tool if the evidence collected so far is \
sufficient to answer the research question thoroughly.

Rules:
- Do not repeat a search you have already run with the same or a trivially \
reworded query — if prior searches on a topic failed, try a substantively \
different query or a different tool.
- Prefer decomposing broad questions into specific, targeted queries \
rather than searching the raw research question verbatim every time.
- If several consecutive searches produced no useful results, change \
strategy significantly (different terminology, different tool, narrower \
or broader scope) rather than repeating the same approach.
- If "Reflection guidance" appears in the state summary below, it is the \
output of a dedicated strategy-diagnosis step — follow its suggested \
strategy change on this turn rather than repeating your previous approach.
- Only call "{finish_tool}" once you have enough varied, relevant evidence \
to support a real literature-review-style report — not just one paper.
- Never fabricate paper titles, authors, or findings; you may only reason \
about evidence already present in the state summary below.
"""


def _finish_tool_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": FINISH_TOOL_NAME,
            "description": (
                "Call this when the evidence collected so far is sufficient "
                "to synthesize a thorough research report. Do not call this "
                "prematurely with little or no evidence."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "rationale": {
                        "type": "string",
                        "description": (
                            "One short sentence on why the evidence is now "
                            "sufficient (e.g. 'Collected 6 relevant papers "
                            "covering methods, limitations, and benchmarks')."
                        ),
                    }
                },
                "required": ["rationale"],
            },
        },
    }


class OpenAIReasoningClient(ReasoningClient):
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._llm = None  # lazily constructed; avoids import cost/failure
        # when no API key is configured (e.g. during tests / Phase 5 wiring
        # before credentials exist).

    def _get_llm(self):
        if self._llm is None:
            # Imported lazily so importing this module never requires
            # langchain-openai to be installed unless actually used.
            from langchain_openai import ChatOpenAI

            if not self._settings.has_llm_credentials:
                raise RuntimeError(
                    "OPENAI_API_KEY is not configured; cannot make live LLM calls. "
                    "Set it in your environment or .env file."
                )
            self._llm = ChatOpenAI(**chat_openai_kwargs(self._settings))
        return self._llm

    def decide_next_action(
        self,
        state: ResearchState,
        tool_schemas: list[dict[str, Any]],
    ) -> AgentDecision:
        all_schemas = list(tool_schemas) + [_finish_tool_schema()]

        try:
            llm = self._get_llm()
            bound = llm.bind_tools(all_schemas)

            system = SYSTEM_PROMPT.format(finish_tool=FINISH_TOOL_NAME)
            human = render_state_summary(state)

            response = bound.invoke(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": human},
                ]
            )
        except Exception as exc:  # noqa: BLE001 - deliberate broad catch per Step 17
            logger.warning("LLM reasoning call failed: %s", exc)
            return AgentDecision(
                action="error",
                rationale_summary=f"LLM call failed: {exc}",
            )

        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            # Model responded with plain text instead of a tool call — treat
            # as a soft failure rather than crashing the loop.
            logger.warning("LLM did not return a tool call; treating as error.")
            return AgentDecision(
                action="error",
                rationale_summary="LLM did not select a tool or finish action.",
            )

        call = tool_calls[0]
        name = call.get("name")
        args = dict(call.get("args") or {})

        if name == FINISH_TOOL_NAME:
            return AgentDecision(
                action="synthesize",
                rationale_summary=args.get("rationale"),
            )

        sub_question = args.pop("sub_question", None)
        rationale = args.pop("rationale", None)

        return AgentDecision(
            action="call_tool",
            tool_name=name,
            tool_args=args,
            sub_question=sub_question,
            rationale_summary=rationale,
        )
