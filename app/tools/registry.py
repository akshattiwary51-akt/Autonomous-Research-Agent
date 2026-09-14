"""Factory for a fully-populated ToolRegistry.

Centralizing tool registration here (rather than scattering
`registry.register(...)` calls) makes it trivial to see, at a glance,
every research tool the agent has access to — and to add a new one later
without touching the graph/agent code.
"""

from __future__ import annotations

from app.tools.arxiv_tool import ArxivTool
from app.tools.base import ToolRegistry
from app.tools.crossref_tool import CrossrefTool
from app.tools.semantic_scholar_tool import SemanticScholarTool


def build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(ArxivTool())
    registry.register(SemanticScholarTool())
    registry.register(CrossrefTool())
    return registry
