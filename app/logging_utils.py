"""Structured, CoT-free observability logging for the research agent.

IMPORTANT: nothing that resembles the model's private chain-of-thought
should ever be passed to these helpers. Only structured *metadata* about
what action was taken (tool, query, result counts, decisions) is logged.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from app.config import get_settings


def get_logger(name: str = "research_agent") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        settings = get_settings()
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(
            logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, settings.log_level, logging.INFO))
    return logger


@dataclass
class IterationLog:
    """Safe, structured record of a single ReAct iteration.

    This is what Step 18 (Observability) requires — no raw model reasoning,
    only the resulting decisions and outcomes.
    """

    iteration: int
    tool: Optional[str] = None
    query: Optional[str] = None
    result_count: int = 0
    useful_result_count: int = 0
    reflection_triggered: bool = False
    status: str = "in_progress"
    termination_reason: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def render(self) -> str:
        lines = [f"ITERATION {self.iteration}"]
        if self.tool:
            lines.append(f"Tool: {self.tool}")
        if self.query:
            lines.append(f'Query: "{self.query}"')
        lines.append(f"Results: {self.result_count}")
        lines.append(f"Useful: {self.useful_result_count}")
        if self.reflection_triggered:
            lines.append("Reflection: triggered")
        lines.append(f"Status: {self.status}")
        if self.termination_reason:
            lines.append(f"Termination reason: {self.termination_reason}")
        return "\n".join(lines)


def log_iteration(log: IterationLog, logger: Optional[logging.Logger] = None) -> None:
    (logger or get_logger()).info("\n" + log.render())
