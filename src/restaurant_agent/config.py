"""Centralised, validated configuration.

All runtime knobs live here and are loaded from environment variables / a local
``.env`` file via pydantic-settings. Nothing else in the codebase should read
``os.environ`` directly — inject ``Settings`` instead. This keeps configuration
testable (you can construct ``Settings(**overrides)`` in a test) and makes the
full set of tunables discoverable in one place.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings sourced from the environment / ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Required secrets ---
    openrouter_api_key: str = Field(..., description="OpenRouter API key.")
    tavily_api_key: str = Field(..., description="Tavily search API key.")

    # --- LLM (OpenRouter) ---
    openrouter_model: str = "nvidia/nemotron-3-super-120b-a12b:free"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 4096
    llm_max_retries: int = 2

    # --- Search / agent behaviour ---
    tavily_project: str = "restaurant_recommender_agent"
    num_restaurants: int = Field(5, ge=1, le=10)
    scout_max_results: int = Field(8, ge=1, le=20)
    scout_review_results: int = Field(4, ge=1, le=10)

    # --- Logging (self-hosted) ---
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["json", "text"] = "text"
    log_file: str | None = None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached, validated ``Settings`` instance.

    Cached so the ``.env`` file is parsed once per process. Tests that need
    different values should construct ``Settings(...)`` directly rather than
    calling this function.
    """
    return Settings()  # type: ignore[call-arg]
