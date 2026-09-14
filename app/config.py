"""Centralized, typed configuration for the research agent.

All runtime configuration is sourced from environment variables (see
`.env.example` for the full list). Nothing here should ever hold a real
secret value — defaults are safe placeholders only.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- LLM ---
    openai_api_key: str = Field(default="", description="Required for live LLM calls.")
    model_name: str = Field(default="gpt-4o-mini")
    openai_base_url: str = Field(
        default="",
        description=(
            "Optional. Point at any OpenAI-compatible endpoint (e.g. Groq's "
            "https://api.groq.com/openai/v1, or OpenRouter's "
            "https://openrouter.ai/api/v1) to use a different provider "
            "without code changes. Leave blank to use OpenAI's default endpoint."
        ),
    )

    # --- Academic APIs ---
    semantic_scholar_api_key: str = Field(default="")

    # --- Safety / loop prevention ---
    max_iterations: int = Field(default=6, ge=1, le=50)
    max_tool_calls: int = Field(default=12, ge=1, le=200)
    zero_yield_reflection_threshold: int = Field(default=2, ge=1, le=10)
    max_reflection_attempts: int = Field(default=2, ge=1, le=10)

    # --- Networking ---
    request_timeout: int = Field(default=15, ge=1, le=120)
    max_results_per_tool_call: int = Field(default=5, ge=1, le=50)

    # --- Full-text retrieval (optional accuracy upgrade over abstract-only) ---
    enable_fulltext_fetch: bool = Field(default=False)
    fulltext_fetch_timeout: int = Field(default=30, ge=1, le=180)
    # Rough character budget per paper passed to the LLM as grounding text
    # — controls cost/latency; full papers are truncated to this length.
    fulltext_max_chars: int = Field(default=15000, ge=1000, le=200000)

    # --- HITL ---
    enable_hitl: bool = Field(default=False)

    # --- Checkpointing ---
    checkpoint_backend: Literal["memory", "sqlite"] = Field(default="memory")
    checkpoint_db_path: str = Field(default="./checkpoints.sqlite")

    # --- Logging ---
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")

    @field_validator("openai_api_key")
    @classmethod
    def _warn_if_missing_key(cls, v: str) -> str:
        # We intentionally do NOT raise here: importing config should never
        # crash (e.g. during test collection). Callers that need a live LLM
        # must check `settings.has_llm_credentials` and fail loudly there.
        return v

    @property
    def has_llm_credentials(self) -> bool:
        return bool(self.openai_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance.

    Cached so the .env file is parsed once per process; tests that need to
    override env vars should call `get_settings.cache_clear()` first.
    """
    return Settings()
