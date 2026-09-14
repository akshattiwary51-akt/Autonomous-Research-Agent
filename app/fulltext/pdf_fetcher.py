"""HTTP + PDF-based FullTextFetcher.

Downloads the `open_access_pdf_url` a tool already resolved (per-source
parsing in app/tools/*.py) and extracts text via `pypdf`. Every failure
mode is caught and reported as a structured `FullTextResult(text=None,
error=...)`, never a crash: missing URL, network timeout/error,
encrypted/malformed PDF, and scanned (image-only, no extractable text)
PDFs are all handled explicitly.
"""

from __future__ import annotations

from typing import Any

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import Settings, get_settings
from app.fulltext.fetcher import FullTextFetcher, FullTextResult
from app.logging_utils import get_logger

logger = get_logger(__name__)


class HttpPdfFullTextFetcher(FullTextFetcher):
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(requests.exceptions.RequestException),
    )
    def _download(self, url: str, timeout: int) -> requests.Response:
        response = requests.get(url, timeout=timeout, headers={"Accept": "application/pdf"})
        response.raise_for_status()
        return response

    def fetch(self, paper: dict[str, Any]) -> FullTextResult:
        url = paper.get("open_access_pdf_url")
        if not url:
            return FullTextResult(
                text=None,
                error="No open-access PDF URL available for this paper; falling back to abstract.",
            )

        timeout = self._settings.fulltext_fetch_timeout

        try:
            response = self._download(url, timeout)
        except requests.exceptions.Timeout:
            logger.warning("Full-text fetch timed out for %r", url)
            return FullTextResult(text=None, error="Full-text PDF download timed out.")
        except requests.exceptions.RequestException as exc:
            logger.warning("Full-text fetch failed for %r: %s", url, exc)
            return FullTextResult(text=None, error=f"Full-text PDF download failed: {exc}")

        try:
            text = self._extract_text(response.content)
        except Exception as exc:  # noqa: BLE001 - pypdf can raise many exception
            # types on malformed/encrypted PDFs; treat all of them as a
            # clean extraction failure rather than crashing (Step 17).
            logger.warning("Failed to parse PDF from %r: %s", url, exc)
            return FullTextResult(text=None, error=f"Failed to parse PDF: {exc}")

        if not text or not text.strip():
            return FullTextResult(
                text=None,
                error="PDF contained no extractable text (likely scanned/image-based).",
            )

        max_chars = self._settings.fulltext_max_chars
        truncated = len(text) > max_chars
        return FullTextResult(text=text[:max_chars], truncated=truncated)

    @staticmethod
    def _extract_text(pdf_bytes: bytes) -> str:
        import io

        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(pdf_bytes))
        if reader.is_encrypted:
            # Try an empty password (some PDFs are "encrypted" only to
            # restrict editing, not to require a real password).
            reader.decrypt("")
        parts = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            if page_text:
                parts.append(page_text)
        return "\n".join(parts)
