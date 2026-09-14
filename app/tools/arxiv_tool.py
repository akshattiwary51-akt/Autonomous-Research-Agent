"""ArXiv search tool.

Uses the public ArXiv Atom API (http://export.arxiv.org/api/query) which
requires no authentication. Implemented with `requests` + XML parsing from
the standard library (avoids pulling in the third-party `arxiv` package
just for this) to keep the dependency surface small.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.logging_utils import get_logger
from app.tools.base import Paper, RateLimitError, ResearchTool, ToolResult, with_reasoning_fields

logger = get_logger(__name__)

ARXIV_API_URL = "http://export.arxiv.org/api/query"
_ATOM_NS = "{http://www.w3.org/2005/Atom}"


class ArxivTool(ResearchTool):
    name = "arxiv_search"
    description = (
        "Search arXiv.org for preprint papers by topic/keywords. Best for "
        "recent ML/AI/physics/math/CS research, including papers not yet "
        "peer-reviewed. Returns title, authors, abstract, publication date, "
        "and a URL for each paper."
    )

    def parameters_schema(self) -> dict[str, Any]:
        return with_reasoning_fields({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Search query. Plain keywords or a short phrase work best "
                        "(e.g. 'multimodal RAG scientific documents'). ArXiv's API "
                        "does support basic boolean operators (AND/OR/ANDNOT) if "
                        "needed, but simple keyword phrases are more reliable."
                    ),
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
    def _fetch(self, query: str, max_results: int, timeout: int) -> requests.Response:
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        response = requests.get(ARXIV_API_URL, params=params, timeout=timeout)
        if response.status_code == 429:
            raise RateLimitError("ArXiv rate limit exceeded (429).")
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
            logger.warning("ArXiv rate limited for query=%r", query)
            return ToolResult(
                tool_name=self.name, query=query, success=False,
                error=str(exc), rate_limited=True,
            )
        except requests.exceptions.Timeout:
            logger.warning("ArXiv request timed out for query=%r", query)
            return ToolResult(
                tool_name=self.name, query=query, success=False,
                error="ArXiv request timed out.",
            )
        except requests.exceptions.RequestException as exc:
            logger.warning("ArXiv request failed for query=%r: %s", query, exc)
            return ToolResult(
                tool_name=self.name, query=query, success=False,
                error=f"ArXiv request failed: {exc}",
            )

        try:
            papers = self._parse_atom_feed(response.text)
        except ET.ParseError as exc:
            logger.warning("ArXiv returned malformed XML for query=%r: %s", query, exc)
            return ToolResult(
                tool_name=self.name, query=query, success=False,
                error=f"Malformed ArXiv response: {exc}",
            )

        if not papers:
            return ToolResult(tool_name=self.name, query=query, success=True, papers=[])

        return ToolResult(tool_name=self.name, query=query, success=True, papers=papers)

    @staticmethod
    def _parse_atom_feed(xml_text: str) -> list[Paper]:
        root = ET.fromstring(xml_text)
        papers: list[Paper] = []

        for entry in root.findall(f"{_ATOM_NS}entry"):
            title_el = entry.find(f"{_ATOM_NS}title")
            id_el = entry.find(f"{_ATOM_NS}id")
            summary_el = entry.find(f"{_ATOM_NS}summary")
            published_el = entry.find(f"{_ATOM_NS}published")

            title = (title_el.text or "").strip() if title_el is not None else ""
            arxiv_url = (id_el.text or "").strip() if id_el is not None else ""
            abstract = (summary_el.text or "").strip() if summary_el is not None else None
            published = published_el.text.strip() if published_el is not None else None

            if not title or not arxiv_url:
                # Malformed / incomplete entry — skip rather than fabricate.
                continue

            authors = [
                (name_el.text or "").strip()
                for author_el in entry.findall(f"{_ATOM_NS}author")
                for name_el in author_el.findall(f"{_ATOM_NS}name")
                if name_el.text
            ]

            paper_id = arxiv_url.rsplit("/", 1)[-1]

            papers.append(
                Paper(
                    title=title,
                    source="arxiv",
                    paper_id=paper_id,
                    url=arxiv_url,
                    authors=authors,
                    abstract=abstract,
                    published=published,
                    open_access_pdf_url=f"https://arxiv.org/pdf/{paper_id}.pdf",
                )
            )

        return papers
