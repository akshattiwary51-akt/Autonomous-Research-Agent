"""Tests for Step 19 items 1-5:
1. Tool registration.
2. ArXiv tool parsing.
3. Semantic Scholar parsing.
4. Empty search results.
5. Tool execution failure.

All external HTTP is mocked via `responses` — no live network calls.
"""

from __future__ import annotations

import pytest
import responses

from app.tools.arxiv_tool import ARXIV_API_URL, ArxivTool
from app.tools.base import ToolRegistry, hash_tool_call
from app.tools.crossref_tool import CROSSREF_API_URL, CrossrefTool
from app.tools.registry import build_default_registry
from app.tools.semantic_scholar_tool import S2_API_URL, SemanticScholarTool

SAMPLE_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.12345v1</id>
    <title>Multimodal RAG for Scientific Documents</title>
    <summary>We study multimodal retrieval augmented generation.</summary>
    <published>2024-01-20T00:00:00Z</published>
    <author><name>Jane Doe</name></author>
  </entry>
</feed>"""

EMPTY_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"></feed>"""

MALFORMED_XML = "<feed><entry><title>broken"


# --------------------------------------------------------------------------
# 1. Tool registration
# --------------------------------------------------------------------------


class TestToolRegistration:
    def test_default_registry_has_three_tools(self):
        registry = build_default_registry()
        assert set(registry.names()) == {
            "arxiv_search",
            "semantic_scholar_search",
            "crossref_search",
        }

    def test_duplicate_registration_raises(self):
        registry = ToolRegistry()
        registry.register(ArxivTool())
        with pytest.raises(ValueError):
            registry.register(ArxivTool())

    def test_unknown_tool_lookup_raises(self):
        registry = build_default_registry()
        with pytest.raises(KeyError):
            registry.get("not_a_real_tool")

    def test_schemas_are_openai_compatible(self):
        registry = build_default_registry()
        schemas = registry.schemas()
        assert len(schemas) == 3
        for schema in schemas:
            assert schema["type"] == "function"
            assert "name" in schema["function"]
            assert "parameters" in schema["function"]
            assert schema["function"]["parameters"]["type"] == "object"

    def test_tool_call_hash_is_deterministic_and_order_independent(self):
        h1 = hash_tool_call("arxiv_search", {"query": "rag", "max_results": 5})
        h2 = hash_tool_call("arxiv_search", {"max_results": 5, "query": "rag"})
        assert h1 == h2

    def test_tool_call_hash_normalizes_whitespace_and_case(self):
        h1 = hash_tool_call("arxiv_search", {"query": "Multimodal RAG"})
        h2 = hash_tool_call("arxiv_search", {"query": "  multimodal   rag "})
        assert h1 == h2

    def test_tool_call_hash_differs_for_different_args(self):
        h1 = hash_tool_call("arxiv_search", {"query": "rag"})
        h2 = hash_tool_call("arxiv_search", {"query": "multimodal rag"})
        assert h1 != h2


# --------------------------------------------------------------------------
# 2. ArXiv tool parsing
# --------------------------------------------------------------------------


class TestArxivTool:
    @responses.activate
    def test_successful_search_returns_structured_papers(self):
        responses.add(responses.GET, ARXIV_API_URL, body=SAMPLE_ATOM, status=200)
        tool = ArxivTool()
        result = tool.run(query="multimodal RAG", max_results=5)

        assert result.success is True
        assert result.result_count == 1
        paper = result.papers[0]
        assert paper.title == "Multimodal RAG for Scientific Documents"
        assert paper.source == "arxiv"
        assert paper.paper_id == "2401.12345v1"
        assert paper.authors == ["Jane Doe"]
        assert paper.published == "2024-01-20T00:00:00Z"

    def test_empty_query_fails_fast_without_network_call(self):
        tool = ArxivTool()
        result = tool.run(query="   ", max_results=5)
        assert result.success is False
        assert "empty" in result.error.lower()

    @responses.activate
    def test_malformed_xml_handled_gracefully(self):
        responses.add(responses.GET, ARXIV_API_URL, body=MALFORMED_XML, status=200)
        tool = ArxivTool()
        result = tool.run(query="anything", max_results=5)
        assert result.success is False
        assert result.error is not None


# --------------------------------------------------------------------------
# 3. Semantic Scholar parsing
# --------------------------------------------------------------------------


class TestSemanticScholarTool:
    @responses.activate
    def test_successful_search_returns_structured_papers(self):
        payload = {
            "data": [
                {
                    "paperId": "abc123",
                    "title": "Vision-Language Models for Scientific PDFs",
                    "abstract": "We propose...",
                    "authors": [{"name": "A. Researcher"}],
                    "year": 2023,
                    "publicationDate": "2023-05-01",
                    "url": "https://www.semanticscholar.org/paper/abc123",
                    "externalIds": {"DOI": "10.1234/xyz"},
                }
            ]
        }
        responses.add(responses.GET, S2_API_URL, json=payload, status=200)
        tool = SemanticScholarTool()
        result = tool.run(query="vision language models", max_results=5)

        assert result.success is True
        assert result.result_count == 1
        paper = result.papers[0]
        assert paper.paper_id == "abc123"
        assert paper.source == "semantic_scholar"
        assert paper.authors == ["A. Researcher"]

    @responses.activate
    def test_incomplete_entries_are_skipped_not_fabricated(self):
        payload = {"data": [{"title": "No paper id here"}]}  # missing paperId
        responses.add(responses.GET, S2_API_URL, json=payload, status=200)
        tool = SemanticScholarTool()
        result = tool.run(query="x", max_results=5)
        assert result.success is True
        assert result.result_count == 0


class TestCrossrefTool:
    @responses.activate
    def test_successful_search_returns_structured_papers(self):
        payload = {
            "message": {
                "items": [
                    {
                        "DOI": "10.1234/example.2024",
                        "title": ["Scalable Retrieval for Scientific PDFs"],
                        "author": [{"given": "Jane", "family": "Doe"}, {"given": "John", "family": "Smith"}],
                        "published": {"date-parts": [[2024, 3, 15]]},
                        "container-title": ["Journal of Example Research"],
                    }
                ]
            }
        }
        responses.add(responses.GET, CROSSREF_API_URL, json=payload, status=200)
        tool = CrossrefTool()
        result = tool.run(query="scalable retrieval", max_results=5)

        assert result.success is True
        assert result.result_count == 1
        paper = result.papers[0]
        assert paper.title == "Scalable Retrieval for Scientific PDFs"
        assert paper.source == "crossref"
        assert paper.paper_id == "10.1234/example.2024"
        assert paper.url == "https://doi.org/10.1234/example.2024"
        assert paper.authors == ["Jane Doe", "John Smith"]
        assert paper.published == "2024-3-15"

    @responses.activate
    def test_incomplete_entries_are_skipped_not_fabricated(self):
        # Missing DOI — must not fabricate a paper_id.
        payload = {"message": {"items": [{"title": ["No DOI here"]}]}}
        responses.add(responses.GET, CROSSREF_API_URL, json=payload, status=200)
        tool = CrossrefTool()
        result = tool.run(query="x", max_results=5)
        assert result.success is True
        assert result.result_count == 0

    @responses.activate
    def test_unexpected_payload_shape_handled_gracefully(self):
        responses.add(responses.GET, CROSSREF_API_URL, json={"message": {"items": "not-a-list"}}, status=200)
        tool = CrossrefTool()
        result = tool.run(query="x", max_results=5)
        assert result.success is False


# --------------------------------------------------------------------------
# 4. Empty search results
# --------------------------------------------------------------------------


class TestEmptyResults:
    @responses.activate
    def test_arxiv_empty_feed_returns_success_with_zero_papers(self):
        responses.add(responses.GET, ARXIV_API_URL, body=EMPTY_ATOM, status=200)
        result = ArxivTool().run(query="a very obscure query", max_results=5)
        assert result.success is True
        assert result.result_count == 0

    @responses.activate
    def test_crossref_empty_results_returns_success_with_zero_papers(self):
        responses.add(
            responses.GET, CROSSREF_API_URL,
            json={"message": {"items": []}}, status=200,
        )
        result = CrossrefTool().run(query="a very obscure query", max_results=5)
        assert result.success is True
        assert result.result_count == 0


# --------------------------------------------------------------------------
# 5. Tool execution failure
# --------------------------------------------------------------------------


class TestToolExecutionFailure:
    @responses.activate
    def test_arxiv_http_500_reported_as_failure_not_crash(self):
        responses.add(responses.GET, ARXIV_API_URL, status=500)
        result = ArxivTool().run(query="anything", max_results=5)
        assert result.success is False
        assert result.error is not None

    @responses.activate
    def test_semantic_scholar_429_reported_as_rate_limit_failure(self):
        responses.add(responses.GET, S2_API_URL, status=429)
        result = SemanticScholarTool().run(query="anything", max_results=5)
        assert result.success is False
        assert "rate limit" in result.error.lower()

    @responses.activate
    def test_crossref_malformed_json_handled_gracefully(self):
        responses.add(
            responses.GET, CROSSREF_API_URL, body="not json at all", status=200,
        )
        result = CrossrefTool().run(query="anything", max_results=5)
        assert result.success is False
        assert result.error is not None

    def test_crossref_empty_query_fails_fast(self):
        result = CrossrefTool().run(query="", max_results=5)
        assert result.success is False
