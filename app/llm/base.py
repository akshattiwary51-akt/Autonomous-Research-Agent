"""LLM abstraction layer for the ReAct agent.

The agent loop (app/agent.py) never talks to an LLM SDK directly — it only
depends on the `ReasoningClient` interface below. This keeps the core
reasoning logic testable without live API calls (see FakeReasoningClient
in tests) and keeps the real provider swappable (Step: "keep external API
integrations replaceable").
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal, Optional

from app.state import ResearchState

AgentAction = Literal["call_tool", "synthesize", "error"]


@dataclass
class AgentDecision:
    """The agent's next move, as safe structured metadata only.

    Never carries raw chain-of-thought — `rationale_summary` is a short,
    user-safe description of *what* was decided and *why* at a high level
    (e.g. "Need evidence on failure modes"), suitable for logging per
    Step 6 / Step 18.
    """

    action: AgentAction
    tool_name: Optional[str] = None
    tool_args: Optional[dict[str, Any]] = None
    sub_question: Optional[str] = None
    rationale_summary: Optional[str] = None


class ReasoningClient(ABC):
    """Interface every reasoning backend (OpenAI, a test fake, etc.) must
    implement."""

    @abstractmethod
    def decide_next_action(
        self,
        state: ResearchState,
        tool_schemas: list[dict[str, Any]],
    ) -> AgentDecision:
        """Given current research state and the available tool schemas,
        decide the next action.

        Implementations must never raise for expected failure modes (LLM
        API error, timeout, malformed tool-call response) — those should
        be captured as `AgentDecision(action="error", rationale_summary=...)`
        per Step 17. Only truly unexpected exceptions may propagate.
        """
        raise NotImplementedError
