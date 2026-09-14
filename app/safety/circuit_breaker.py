"""Circuit breaker for research tool availability (Step 11 extension).

When a tool returns an explicit rate-limit signal (HTTP 429), repeatedly
offering it to the agent as an option just burns iterations waiting for
reflection to notice the pattern and suggest a different tool. This
module makes that failover automatic and immediate: a rate-limited tool
is temporarily excluded from the schemas offered to the LLM on the very
next turn, so the agent picks among currently-healthy tools without
needing to rediscover "this one is throttled" itself.

Deliberately dynamic, not a hardcoded fallback chain (e.g. "always try
Crossref after Semantic Scholar") -- which tool the agent reaches for next
is still its own choice among whatever remains healthy, preserving Step
5's "the LLM dynamically selects tools" principle rather than hardcoding
an order.

Only HTTP 429 (`ToolResult.rate_limited=True`) trips the breaker -- other
failures (timeout, malformed response, 5xx) don't, since those aren't
necessarily resolved by switching providers and the existing zero-yield
reflection path already handles those more general cases.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class _ToolCircuitState:
    open_until: float = 0.0  # monotonic timestamp; tool is "open" (excluded) until this time


class CircuitBreaker:
    """Thread-unsafe by design -- one instance per research run (owned by
    the graph closure, like the tool registry), never shared across
    concurrent runs. Not part of `ResearchState`: this is operational
    health tracking, not research data, and doesn't need checkpointing."""

    def __init__(self, cooldown_seconds: float = 60.0) -> None:
        self._cooldown = cooldown_seconds
        self._state: dict[str, _ToolCircuitState] = {}

    def record_result(self, tool_name: str, success: bool, rate_limited: bool) -> None:
        if rate_limited:
            self._state[tool_name] = _ToolCircuitState(
                open_until=time.monotonic() + self._cooldown
            )
        elif success:
            # A healthy result is an immediate "all clear" signal, even if
            # the cooldown timer hasn't technically elapsed yet.
            self._state.pop(tool_name, None)

    def is_open(self, tool_name: str) -> bool:
        state = self._state.get(tool_name)
        if state is None:
            return False
        if time.monotonic() >= state.open_until:
            del self._state[tool_name]
            return False
        return True

    def available_tools(self, all_tool_names: list[str]) -> list[str]:
        """Returns `all_tool_names` minus any currently-open (rate-limited)
        ones. Fails open (returns the full original list) if that would
        otherwise leave zero options -- a temporarily-throttled tool is
        still better than no tools at all."""
        healthy = [name for name in all_tool_names if not self.is_open(name)]
        return healthy if healthy else list(all_tool_names)
