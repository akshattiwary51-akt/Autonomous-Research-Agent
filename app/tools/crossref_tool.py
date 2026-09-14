"""Crossref search tool.

Uses the public Crossref REST API (no auth required) for citation/DOI
metadata discovery. Useful for finding formally published, peer-reviewed
works that complement arXiv preprints and Semantic Scholar coverage.
"""

from __future__ import annotations

from typing import Any

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.logging_utils import get_logger
from app.tools.base import Paper, ResearchTool, ToolResult, with_reasoning_fields

logger = get_logger(__name__)

CROSSREF_API_URL = "https://api.crossref.org/works"


class CrossrefTool(ResearchTool):
    name = "crossref_search"
    description = (
        "Search Crossref for formally published, peer-reviewed academic "
        "works with DOIs. Best for citation metadata and confirming "
        "publication venue/year. Returns title, authors, publication date, "
        "DOI URL, and source for each work; abstracts are often unavailable."
    )

    def parameters_schema(self) -> dict[str, Any]:
        return with_reasoning_fields({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (keywords, title fragment, or author).",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of works to return.",
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
    def _fetch(self, query: str, max_results: int, timeout: int) -> requests.Response:
        params = {
            "query": query,
            "rows": max_results,
            # Polite pool: identify ourselves per Crossref etiquette guidelines.
            "mailto": "research-agent@example.com",
        }
        response = requests.get(CROSSREF_API_URL, params=params, timeout=timeout)
        if response.status_code == 429:
            raise RateLimitError("Crossref rate limit exceeded (429).")
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

        try:
            response = self._fetch(query=query, max_results=max_results, timeout=timeout)
        except RateLimitError as exc:
            logger.warning("Crossref rate limited for query=%r", query)
            return ToolResult(
                tool_name=self.name, query=query, success=False, error=str(exc),
            )
        except requests.exceptions.Timeout:
            logger.warning("Crossref request timed out for query=%r", query)
            return ToolResult(
                tool_name=self.name, query=query, success=False,
                error="Crossref request timed out.",
            )
        except requests.exceptions.RequestException as exc:
            logger.warning("Crossref request failed for query=%r: %s", query, exc)
            return ToolResult(
                tool_name=self.name, query=query, success=False,
                error=f"Crossref request failed: {exc}",
            )

        try:
            payload = response.json()
        except ValueError as exc:
            logger.warning("Crossref returned malformed JSON: %s", exc)
            return ToolResult(
                tool_name=self.name, query=query, success=False,
                error=f"Malformed Crossref response: {exc}",
            )

        try:
            papers = self._parse_payload(payload)
        except (KeyError, TypeError, AttributeError) as exc:
            logger.warning("Unexpected Crossref payload shape: %s", exc)
            return ToolResult(
                tool_name=self.name, query=query, success=False,
                error=f"Unexpected Crossref response shape: {exc}",
            )

        return ToolResult(tool_name=self.name, query=query, success=True, papers=papers)

    @staticmethod
    def _parse_payload(payload: dict[str, Any]) -> list[Paper]:
        items = (payload.get("message") or {}).get("items", []) or []
        papers: list[Paper] = []

        for item in items:
            title_list = item.get("title") or []
            title = title_list[0].strip() if title_list else ""
            doi = item.get("DOI")
            if not title or not doi:
                continue

            authors = [
                " ".join(filter(None, [a.get("given"), a.get("family")])).strip()
                for a in (item.get("author") or [])
            ]
            authors = [a for a in authors if a]

            date_parts = (
                (item.get("published") or item.get("issued") or {}).get("date-parts")
            )
            published = None
            if date_parts and date_parts[0]:
                published = "-".join(str(p) for p in date_parts[0])

            open_access_pdf = None
            for link in item.get("link") or []:
                if isinstance(link, dict) and link.get("content-type") == "application/pdf":
                    open_access_pdf = link.get("URL")
                    break

            papers.append(
                Paper(
                    title=title,
                    source="crossref",
                    paper_id=doi,
                    url=f"https://doi.org/{doi}",
                    authors=authors,
                    abstract=item.get("abstract"),  # often absent from Crossref
                    published=published,
                    open_access_pdf_url=open_access_pdf,
                    extra={"container_title": (item.get("container-title") or [None])[0]},
                )
            )

        return papers


class RateLimitError(Exception):
    """Raised when Crossref explicitly signals rate limiting."""
