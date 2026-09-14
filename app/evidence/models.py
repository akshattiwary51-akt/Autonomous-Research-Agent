"""Evidence data model helpers.

`EvidenceItem` and `Contradiction` are defined in `app.state` (they're
core parts of `ResearchState`) — this module re-exports them for a clean
`app.evidence.*` import path and adds helpers for reconstructing the
dataclasses from the plain dicts stored in state (state must stay
JSON-serializable for checkpointing, so dataclasses are always stored as
dicts there).
"""

from __future__ import annotations

from typing import Any

from app.state import Contradiction, EvidenceItem

__all__ = ["EvidenceItem", "Contradiction", "evidence_items_from_dicts"]

_VALID_RELEVANCE = {"high", "medium", "low", "irrelevant"}
_VALID_CONFIDENCE = {"strong", "moderate", "weak"}


def evidence_items_from_dicts(items: list[dict[str, Any]]) -> list[EvidenceItem]:
    """Reconstruct `EvidenceItem` dataclasses from state's plain-dict form."""
    result = []
    for d in items:
        try:
            result.append(EvidenceItem(**d))
        except TypeError:
            # Defensive: skip malformed entries rather than crash (Step 17).
            continue
    return result
