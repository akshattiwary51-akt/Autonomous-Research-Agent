"""Semantic Scholar search tool.

Uses the public Semantic Scholar Graph API. Works without an API key
(shared rate-limit pool); an optional `SEMANTIC_SCHOLAR_API_KEY` raises
the rate limit if provided.
"""

from __future__ import annotations

from typing import Any

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.logging_utils import get_logger
from app.tools.base import Paper, ResearchTool, ToolResult, with_reasoning_fields

logger = get_logger(__name__)

S2_API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_FIELDS = "title,abstract,authors,year,externalIds,url,publicationDate,openAccessPdf"


class SemanticScholarTool(ResearchTool):
    name = "semantic_scholar_search"
    description = (
        "Search Semantic Scholar for peer-reviewed and preprint academic "
        "papers across all fields. Good for citation-aware discovery and "
        "papers with richer metadata than arXiv alone. Returns title, "
        "authors, abstract, publication date, and a URL for each paper."
    )

    def parameters_schema(self) -> dict[str, Any]:
        return with_reasoning_fields({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (keywords or phrase describing the research topic).",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of papers to return.",
                    "default": 5,
                },
            },
            "required": ["query"],
        })

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(requests.exceptions.RequestException),
    )
    def _fetch(
        self, query: str, max_results: int, timeout: int, headers: dict[str, str]
    ) -> requests.Response:
        params = {
            "query": query,
            "limit": max_results,
            "fields": S2_FIELDS,
        }
        response = requests.get(
            S2_API_URL, params=params, headers=headers, timeout=timeout
        )
        # 429 is a rate limit, not a transient network error — handle
        # explicitly rather than retrying blindly against tenacity's
        # RequestException filter.
        if response.status_code == 429:
            raise RateLimitError("Semantic Scholar rate limit exceeded (429).")
        response.raise_for_status()
        return response

    def run(self, query: str, max_results: int = 5, **kwargs: Any) -> ToolResult:
        settings = get_settings()
        timeout = settings.request_timeout
        max_results = min(max_results, settings.max_results_per_tool_call)

        if not query or not query.strip():
            return ToolResult(
                tool_name=self.name, query=query, success=False,
                error="Empty query provided.",
            )

        headers: dict[str, str] = {}
        if settings.semantic_scholar_api_key:
            headers["x-api-key"] = settings.semantic_scholar_api_key

        try:
            response = self._fetch(
                query=query, max_results=max_results, timeout=timeout, headers=headers
            )
        except RateLimitError as exc:
            logger.warning("Semantic Scholar rate limited for query=%r", query)
            return ToolResult(
                tool_name=self.name, query=query, success=False, error=str(exc),
            )
        except requests.exceptions.Timeout:
            logger.warning("Semantic Scholar request timed out for query=%r", query)
            return ToolResult(
                tool_name=self.name, query=query, success=False,
                error="Semantic Scholar request timed out.",
            )
        except requests.exceptions.RequestException as exc:
            logger.warning("Semantic Scholar request failed for query=%r: %s", query, exc)
            return ToolResult(
                tool_name=self.name, query=query, success=False,
                error=f"Semantic Scholar request failed: {exc}",
            )

        try:
            payload = response.json()
        except ValueError as exc:
            logger.warning("Semantic Scholar returned malformed JSON: %s", exc)
            return ToolResult(
                tool_name=self.name, query=query, success=False,
                error=f"Malformed Semantic Scholar response: {exc}",
            )

        try:
            papers = self._parse_payload(payload)
        except (KeyError, TypeError, AttributeError) as exc:
            logger.warning("Unexpected Semantic Scholar payload shape: %s", exc)
            return ToolResult(
                tool_name=self.name, query=query, success=False,
                error=f"Unexpected Semantic Scholar response shape: {exc}",
            )
        return ToolResult(tool_name=self.name, query=query, success=True, papers=papers)

    @staticmethod
    def _parse_payload(payload: dict[str, Any]) -> list[Paper]:
        papers: list[Paper] = []
        for item in payload.get("data", []) or []:
            title = (item.get("title") or "").strip()
            paper_id = item.get("paperId")
            if not title or not paper_id:
                # Skip incomplete entries rather than fabricating identifiers.
                continue

            authors = [
                a.get("name", "").strip()
                for a in (item.get("authors") or [])
                if a.get("name")
            ]

            external_ids = item.get("externalIds") or {}
            url = item.get("url") or (
                f"https://doi.org/{external_ids['DOI']}"
                if external_ids.get("DOI")
                else None
            )
            open_access_pdf = (item.get("openAccessPdf") or {}).get("url")

            papers.append(
                Paper(
                    title=title,
                    source="semantic_scholar",
                    paper_id=paper_id,
                    url=url,
                    authors=authors,
                    abstract=item.get("abstract"),
                    published=item.get("publicationDate") or (
                        str(item["year"]) if item.get("year") else None
                    ),
                    open_access_pdf_url=open_access_pdf,
                    extra={"externalIds": external_ids},
                )
            )
        return papers


class RateLimitError(Exception):
    """Raised when an academic API explicitly signals rate limiting."""
