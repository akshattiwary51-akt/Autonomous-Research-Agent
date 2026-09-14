"""Tool interface + registry for the research agent.

Design goal (Step 5 of the spec): the LLM must be able to dynamically pick
a tool from a registry using structured tool-call definitions — never via
hardcoded `if "arxiv" in question` string matching.

Every concrete tool (ArxivTool, SemanticScholarTool, CrossrefTool, ...)
subclasses `ResearchTool` and registers itself in a shared `ToolRegistry`
instance. The registry exposes an OpenAI/LangChain-compatible JSON schema
for each tool so it can be handed directly to the LLM's function-calling
API.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


# --------------------------------------------------------------------------
# Structured paper / result schema (Step 4)
# --------------------------------------------------------------------------


@dataclass
class Paper:
    """Structured metadata for a single retrieved paper.

    Every field beyond `title` and `source` is optional because different
    academic APIs expose different levels of metadata — callers must not
    assume completeness.
    """

    title: str
    source: str  # "arxiv" | "semantic_scholar" | "crossref"
    paper_id: str  # stable identifier: arxiv id, S2 paper id, DOI, etc.
    url: Optional[str] = None
    authors: list[str] = field(default_factory=list)
    abstract: Optional[str] = None
    published: Optional[str] = None  # ISO date string if available
    # Direct link to a freely-downloadable full-text PDF, if this source's
    # API can resolve one (Step: full-text grounding). None means no known
    # open-access PDF — callers must fall back to abstract-only evidence
    # extraction, never fabricate a URL.
    open_access_pdf_url: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "source": self.source,
            "paper_id": self.paper_id,
            "url": self.url,
            "authors": self.authors,
            "abstract": self.abstract,
            "published": self.published,
            "open_access_pdf_url": self.open_access_pdf_url,
        }


@dataclass
class ToolResult:
    """The outcome of a single tool invocation."""

    tool_name: str
    query: str
    success: bool
    papers: list[Paper] = field(default_factory=list)
    error: Optional[str] = None
    # True specifically when the failure was an explicit rate-limit signal
    # (HTTP 429) from the provider -- distinguished from other failures
    # (timeout, malformed response, 5xx) because it's the one signal that
    # means "this exact provider is throttling us right now", which is
    # what the circuit breaker (app/safety/circuit_breaker.py) acts on.
    rate_limited: bool = False

    @property
    def result_count(self) -> int:
        return len(self.papers)


class RateLimitError(Exception):
    """Raised when an academic API explicitly signals rate limiting (429).
    Shared across tools so the circuit breaker's trip condition is
    consistent regardless of which provider raised it."""


# --------------------------------------------------------------------------
# Tool interface
# --------------------------------------------------------------------------


class ResearchTool(ABC):
    """Base class every research tool must implement."""

    name: str
    description: str

    @abstractmethod
    def parameters_schema(self) -> dict[str, Any]:
        """Return a JSON-schema `parameters` block (OpenAI function-calling
        compatible) describing this tool's accepted arguments."""
        raise NotImplementedError

    @abstractmethod
    def run(self, query: str, max_results: int = 5, **kwargs: Any) -> ToolResult:
        """Execute the tool. Must never raise for expected failure modes
        (timeout, HTTP error, malformed response, empty results) — those
        must be captured into `ToolResult(success=False, error=...)`.
        Unexpected exceptions are still allowed to propagate so the agent
        node's own try/except can log them, per Step 17.
        """
        raise NotImplementedError

    def openai_tool_schema(self) -> dict[str, Any]:
        """Render this tool as an OpenAI/LangChain function-calling schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema(),
            },
        }


# --------------------------------------------------------------------------
# Tool-call hashing (Step 11.B) — lives here since it's tightly coupled to
# the tool call shape, though the loop-prevention *policy* lives in
# app/safety/loop_guard.py
# --------------------------------------------------------------------------


def normalize_args(args: dict[str, Any]) -> dict[str, Any]:
    """Normalize tool arguments so trivially-different calls hash the same.

    - Strings are lowercased and whitespace-collapsed.
    - Keys are sorted implicitly by json.dumps(sort_keys=True).
    """
    normalized: dict[str, Any] = {}
    for key, value in args.items():
        if isinstance(value, str):
            normalized[key] = " ".join(value.lower().split())
        else:
            normalized[key] = value
    return normalized


def hash_tool_call(tool_name: str, args: dict[str, Any]) -> str:
    """Deterministic hash of (tool name, normalized args) for dedup."""
    payload = json.dumps(
        {"tool": tool_name, "args": normalize_args(args)},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def with_reasoning_fields(schema: dict[str, Any]) -> dict[str, Any]:
    """Augment a tool's JSON parameter schema with optional reasoning
    metadata fields (Step 6, Step 8).

    `sub_question` lets the agent tag which research sub-question this
    call targets. `rationale` lets the agent record *why* this specific
    query was chosen — especially important when reformulating a
    previously unproductive query, so the reformulation reasoning is
    traceable in state/logs rather than only living in the LLM's own
    (unlogged) reasoning.

    Both are optional so a minimal tool call still validates.
    """
    schema = dict(schema)
    properties = dict(schema.get("properties", {}))
    properties.setdefault(
        "sub_question",
        {
            "type": "string",
            "description": "Which research sub-question this search targets, if applicable.",
        },
    )
    properties.setdefault(
        "rationale",
        {
            "type": "string",
            "description": (
                "One short phrase explaining why this specific query was "
                "chosen. If this reformulates a previous unproductive "
                "query, briefly say why the previous one failed and what "
                "changed."
            ),
        },
    )
    schema["properties"] = properties
    return schema


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


class ToolRegistry:
    """Maps tool name -> ResearchTool instance.

    The agent node uses `.schemas()` to hand the LLM every available tool,
    and `.get(name).run(...)` to execute whichever one the LLM selects.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ResearchTool] = {}

    def register(self, tool: ResearchTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered.")
        self._tools[tool.name] = tool

    def get(self, name: str) -> ResearchTool:
        if name not in self._tools:
            raise KeyError(
                f"Unknown tool '{name}'. Registered tools: {list(self._tools)}"
            )
        return self._tools[name]

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def schemas(self) -> list[dict[str, Any]]:
        """OpenAI/LangChain-compatible tool schemas for every registered tool."""
        return [tool.openai_tool_schema() for tool in self._tools.values()]

    def schemas_for(self, names: list[str]) -> list[dict[str, Any]]:
        """Schemas for only the given tool names, in registry order —
        used by the circuit breaker to offer the LLM only currently-healthy
        tools without needing to rebuild the registry itself."""
        name_set = set(names)
        return [
            tool.openai_tool_schema()
            for tool in self._tools.values()
            if tool.name in name_set
        ]
