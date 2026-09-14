"""Interface for full-text paper retrieval.

Evidence extraction (Phase 8) only ever saw title + abstract, which is a
real accuracy ceiling — an abstract is a marketing summary of a paper, not
its actual methodology, results, or stated limitations. This subsystem
optionally fetches and extracts the full text of the underlying PDF (when
an open-access one is resolvable) so evidence can be grounded in much more
than a couple of sentences.

Mirrors every other LLM/IO-backed subsystem in this project: an ABC, a
safe `NoOp` default that changes nothing about existing behavior, and a
real implementation (`HttpPdfFullTextFetcher`) that never raises for
expected failure modes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class FullTextResult:
    """Outcome of one full-text fetch attempt.

    `text` is None whenever full text isn't available for any reason
    (no PDF URL, network failure, unparseable/scanned PDF, etc.) — callers
    must fall back to the paper's abstract in that case, never fabricate
    content.
    """

    text: Optional[str] = None
    truncated: bool = False
    error: Optional[str] = None


class FullTextFetcher(ABC):
    @abstractmethod
    def fetch(self, paper: dict[str, Any]) -> FullTextResult:
        """Must never raise for expected failure modes (missing URL,
        timeout, malformed/encrypted/scanned PDF) — return
        `FullTextResult(text=None, error=...)` instead (Step 17)."""
        raise NotImplementedError


class NoOpFullTextFetcher(FullTextFetcher):
    """Safe default: full-text fetching disabled, evidence extraction
    falls back to abstract-only exactly as it did before this feature
    existed. This is the default everywhere unless explicitly enabled."""

    def fetch(self, paper: dict[str, Any]) -> FullTextResult:
        return FullTextResult(text=None, error="Full-text fetching is disabled.")
