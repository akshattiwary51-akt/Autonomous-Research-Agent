"""Shared ChatOpenAI construction helper.

Every OpenAI-backed component (reasoning client, decomposer, reflection
advisor, evidence evaluator, gap discoverer, narrative writer) builds its
`ChatOpenAI` instance through this one function, so pointing the whole
project at a different OpenAI-compatible provider (Groq, OpenRouter,
etc.) via `OPENAI_BASE_URL` only has to be handled correctly in one place.
"""

from __future__ import annotations

from typing import Any

from app.config import Settings


def chat_openai_kwargs(settings: Settings) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": settings.model_name,
        "api_key": settings.openai_api_key,
        "timeout": settings.request_timeout,
    }
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
    return kwargs
