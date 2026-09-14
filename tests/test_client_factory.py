"""Tests for app/llm/client_factory.py — the shared helper that lets every
OpenAI-backed component point at an alternate OpenAI-compatible endpoint
(Groq, OpenRouter, etc.) via a single settings field."""

from __future__ import annotations

from app.config import Settings
from app.llm.client_factory import chat_openai_kwargs


class TestChatOpenAIKwargs:
    def test_omits_base_url_when_not_configured(self):
        settings = Settings(openai_api_key="sk-test")
        kwargs = chat_openai_kwargs(settings)
        assert "base_url" not in kwargs
        assert kwargs["model"] == settings.model_name
        assert kwargs["api_key"] == "sk-test"
        assert kwargs["timeout"] == settings.request_timeout

    def test_includes_base_url_when_configured(self):
        settings = Settings(
            openai_api_key="gsk-test",
            openai_base_url="https://api.groq.com/openai/v1",
        )
        kwargs = chat_openai_kwargs(settings)
        assert kwargs["base_url"] == "https://api.groq.com/openai/v1"

    def test_default_base_url_is_empty(self):
        settings = Settings()
        assert settings.openai_base_url == ""
