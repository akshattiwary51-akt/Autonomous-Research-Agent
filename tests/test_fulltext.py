"""Tests for app/fulltext/* and its integration into evidence_node and
OpenAIEvidenceEvaluator's prompt rendering — the full-text accuracy
upgrade over abstract-only grounding.
"""

from __future__ import annotations

import io

import responses

from app.evidence.evaluator import EvidenceEvaluator, NoOpContradictionDetector
from app.evidence.openai_evaluator import OpenAIEvidenceEvaluator
from app.fulltext.fetcher import FullTextResult, NoOpFullTextFetcher
from app.fulltext.pdf_fetcher import HttpPdfFullTextFetcher
from app.graph.evidence_node import build_evidence_node
from app.state import create_initial_state
from app.tools.arxiv_tool import ArxivTool
from app.tools.crossref_tool import CrossrefTool
from app.tools.semantic_scholar_tool import SemanticScholarTool

SAMPLE_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.12345v1</id>
    <title>Multimodal RAG for Scientific Documents</title>
    <summary>abstract text</summary>
  </entry>
</feed>"""


def _build_test_pdf_bytes(lines: list[str]) -> bytes:
    """Generates a REAL, valid PDF with extractable text using reportlab,
    so pdf_fetcher's extraction logic is tested against genuine PDF bytes,
    not mocked-around."""
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    y = 750
    for line in lines:
        c.drawString(100, y, line)
        y -= 20
    c.save()
    return buf.getvalue()


# --------------------------------------------------------------------------
# Per-source open_access_pdf_url resolution
# --------------------------------------------------------------------------


class TestOpenAccessPdfUrlResolution:
    def test_arxiv_always_resolves_pdf_url_from_paper_id(self):
        papers = ArxivTool._parse_atom_feed(SAMPLE_ATOM)
        assert papers[0].open_access_pdf_url == "https://arxiv.org/pdf/2401.12345v1.pdf"

    def test_semantic_scholar_uses_open_access_pdf_field_when_present(self):
        payload = {"data": [{
            "paperId": "abc123", "title": "T",
            "openAccessPdf": {"url": "https://example.org/paper.pdf"},
        }]}
        papers = SemanticScholarTool._parse_payload(payload)
        assert papers[0].open_access_pdf_url == "https://example.org/paper.pdf"

    def test_semantic_scholar_none_when_no_open_access_pdf(self):
        payload = {"data": [{"paperId": "abc123", "title": "T"}]}
        papers = SemanticScholarTool._parse_payload(payload)
        assert papers[0].open_access_pdf_url is None

    def test_crossref_finds_pdf_link_by_content_type(self):
        payload = {"message": {"items": [{
            "DOI": "10.1234/x", "title": ["T"],
            "link": [
                {"URL": "https://example.org/paper.xml", "content-type": "text/xml"},
                {"URL": "https://example.org/paper.pdf", "content-type": "application/pdf"},
            ],
        }]}}
        papers = CrossrefTool._parse_payload(payload)
        assert papers[0].open_access_pdf_url == "https://example.org/paper.pdf"

    def test_crossref_none_when_no_pdf_link(self):
        payload = {"message": {"items": [{
            "DOI": "10.1234/x", "title": ["T"],
            "link": [{"URL": "https://example.org/paper.xml", "content-type": "text/xml"}],
        }]}}
        papers = CrossrefTool._parse_payload(payload)
        assert papers[0].open_access_pdf_url is None

    def test_crossref_none_when_no_link_field_at_all(self):
        payload = {"message": {"items": [{"DOI": "10.1234/x", "title": ["T"]}]}}
        papers = CrossrefTool._parse_payload(payload)
        assert papers[0].open_access_pdf_url is None


# --------------------------------------------------------------------------
# NoOpFullTextFetcher
# --------------------------------------------------------------------------


class TestNoOpFullTextFetcher:
    def test_always_returns_no_text(self):
        result = NoOpFullTextFetcher().fetch({"open_access_pdf_url": "https://x.pdf"})
        assert result.text is None
        assert result.error


# --------------------------------------------------------------------------
# HttpPdfFullTextFetcher — real PDF bytes, real extraction
# --------------------------------------------------------------------------


class TestHttpPdfFullTextFetcher:
    def test_no_url_returns_error_without_network_call(self):
        fetcher = HttpPdfFullTextFetcher()
        result = fetcher.fetch({"title": "No PDF here"})
        assert result.text is None
        assert "No open-access PDF URL" in result.error

    @responses.activate
    def test_successful_download_and_extraction_of_real_pdf(self):
        pdf_bytes = _build_test_pdf_bytes([
            "This is a test paper about multimodal RAG limitations.",
            "We find that figure extraction remains a key bottleneck.",
        ])
        responses.add(responses.GET, "https://example.org/paper.pdf", body=pdf_bytes, status=200,
                       content_type="application/pdf")

        fetcher = HttpPdfFullTextFetcher()
        result = fetcher.fetch({"open_access_pdf_url": "https://example.org/paper.pdf"})

        assert result.error is None
        assert "multimodal RAG limitations" in result.text
        assert "figure extraction" in result.text
        assert result.truncated is False

    @responses.activate
    def test_truncates_long_text_to_configured_max_chars(self):
        from app.config import Settings

        pdf_bytes = _build_test_pdf_bytes(["word " * 500])  # long single line
        responses.add(responses.GET, "https://example.org/paper.pdf", body=pdf_bytes, status=200)

        fetcher = HttpPdfFullTextFetcher(settings=Settings(fulltext_max_chars=1000))
        result = fetcher.fetch({"open_access_pdf_url": "https://example.org/paper.pdf"})

        assert result.truncated is True
        assert len(result.text) == 1000

    @responses.activate
    def test_download_failure_handled_gracefully(self):
        responses.add(responses.GET, "https://example.org/paper.pdf", status=500)
        fetcher = HttpPdfFullTextFetcher()
        result = fetcher.fetch({"open_access_pdf_url": "https://example.org/paper.pdf"})
        assert result.text is None
        assert result.error is not None

    @responses.activate
    def test_malformed_pdf_bytes_handled_gracefully_not_raised(self):
        responses.add(responses.GET, "https://example.org/paper.pdf", body=b"not a real pdf at all", status=200)
        fetcher = HttpPdfFullTextFetcher()
        result = fetcher.fetch({"open_access_pdf_url": "https://example.org/paper.pdf"})
        assert result.text is None
        assert "Failed to parse PDF" in result.error

    @responses.activate
    def test_pdf_with_no_extractable_text_reported_as_scanned(self):
        # A blank-page PDF (no drawString calls) has zero extractable text
        # -- simulates a scanned/image-only paper.
        from reportlab.pdfgen import canvas
        buf = io.BytesIO()
        canvas.Canvas(buf).save()
        responses.add(responses.GET, "https://example.org/paper.pdf", body=buf.getvalue(), status=200)

        fetcher = HttpPdfFullTextFetcher()
        result = fetcher.fetch({"open_access_pdf_url": "https://example.org/paper.pdf"})
        assert result.text is None
        assert "no extractable text" in result.error


# --------------------------------------------------------------------------
# evidence_node full-text wiring
# --------------------------------------------------------------------------


class FakeFullTextFetcher:
    def __init__(self, text_by_paper_id: dict[str, str]):
        self._text = text_by_paper_id

    def fetch(self, paper):
        text = self._text.get(paper.get("paper_id"))
        if text:
            return FullTextResult(text=text, truncated=False)
        return FullTextResult(text=None, error="not available")


class RecordingEvidenceEvaluator(EvidenceEvaluator):
    """Records exactly what papers it was called with, so we can confirm
    full_text was actually attached before evaluation."""

    def __init__(self):
        self.received_papers = None

    def evaluate_papers(self, papers, research_question, sub_questions):
        self.received_papers = papers
        return []


class TestEvidenceNodeFullTextWiring:
    def test_papers_augmented_with_full_text_before_evaluation(self):
        evaluator = RecordingEvidenceEvaluator()
        fetcher = FakeFullTextFetcher({"p1": "Full paper text about limitations."})
        node = build_evidence_node(evaluator, NoOpContradictionDetector(), fetcher)

        state = create_initial_state("test question")
        state["last_tool_papers"] = [{"paper_id": "p1", "title": "T", "abstract": "short abstract"}]

        node(state)

        assert evaluator.received_papers[0]["full_text"] == "Full paper text about limitations."

    def test_papers_without_fulltext_left_unaugmented(self):
        evaluator = RecordingEvidenceEvaluator()
        fetcher = FakeFullTextFetcher({})  # nothing available
        node = build_evidence_node(evaluator, NoOpContradictionDetector(), fetcher)

        state = create_initial_state("test question")
        state["last_tool_papers"] = [{"paper_id": "p1", "title": "T", "abstract": "short abstract"}]

        node(state)

        assert "full_text" not in evaluator.received_papers[0]

    def test_default_fetcher_is_noop_preserves_old_behavior(self):
        evaluator = RecordingEvidenceEvaluator()
        node = build_evidence_node(evaluator, NoOpContradictionDetector())  # no fetcher passed

        state = create_initial_state("test question")
        state["last_tool_papers"] = [{"paper_id": "p1", "title": "T", "abstract": "short abstract"}]

        node(state)

        assert "full_text" not in evaluator.received_papers[0]

    def test_scratchpad_records_fulltext_fetch_count(self):
        evaluator = RecordingEvidenceEvaluator()
        fetcher = FakeFullTextFetcher({"p1": "text"})
        node = build_evidence_node(evaluator, NoOpContradictionDetector(), fetcher)

        state = create_initial_state("test question")
        state["last_tool_papers"] = [
            {"paper_id": "p1", "title": "T1"},
            {"paper_id": "p2", "title": "T2"},  # no full text available
        ]

        update = node(state)

        note = update["scratchpad"][0]
        assert note["full_text_fetched"] == 1
        assert note["papers_evaluated"] == 2


# --------------------------------------------------------------------------
# OpenAIEvidenceEvaluator prompt rendering prefers full text
# --------------------------------------------------------------------------


class TestEvidencePromptPrefersFullText:
    def test_uses_full_text_when_present(self):
        papers = [{"paper_id": "p1", "title": "T", "abstract": "short", "full_text": "much longer real text"}]
        prompt = OpenAIEvidenceEvaluator._render_papers_prompt(papers, "q", [])
        assert "much longer real text" in prompt
        assert "Full text" in prompt

    def test_falls_back_to_abstract_when_no_full_text(self):
        papers = [{"paper_id": "p1", "title": "T", "abstract": "short abstract only"}]
        prompt = OpenAIEvidenceEvaluator._render_papers_prompt(papers, "q", [])
        assert "short abstract only" in prompt
        assert "Abstract:" in prompt

    def test_marks_truncated_full_text(self):
        papers = [{"paper_id": "p1", "title": "T", "full_text": "text", "full_text_truncated": True}]
        prompt = OpenAIEvidenceEvaluator._render_papers_prompt(papers, "q", [])
        assert "(truncated)" in prompt
