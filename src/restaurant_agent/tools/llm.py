"""Thin, observable wrapper around the OpenRouter chat model.

Centralises model construction and adds:
  * structured timing/usage logging on every call (our self-hosted substitute
    for a tracing backend),
  * a tenacity retry layer for transient network/5xx errors on top of the SDK's
    own ``max_retries``.

Nothing else in the codebase constructs ``ChatOpenRouter`` directly.
"""

from __future__ import annotations

import os
import time

from langchain_core.messages import BaseMessage
from langchain_openrouter import ChatOpenRouter
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import Settings
from ..logging_config import get_logger, log_event

log = get_logger(__name__)


class LLMClient:
    """Wraps a configured ``ChatOpenRouter`` with logging + retries."""

    def __init__(self, settings: Settings):
        self._settings = settings
        # The SDK reads the key from the environment; set it from validated config
        # so callers never touch os.environ themselves.
        os.environ.setdefault("OPENROUTER_API_KEY", settings.openrouter_api_key)
        self._model = ChatOpenRouter(
            model=settings.openrouter_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            max_retries=settings.llm_max_retries,
            streaming=False,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def invoke(self, messages: list[BaseMessage], *, step: str = "llm") -> str:
        """Invoke the model and return its text content.

        ``step`` is a short label (e.g. ``scout.extract``) used in logs so each
        call site is distinguishable.
        """
        start = time.perf_counter()
        try:
            response = self._model.invoke(messages)
        except Exception as exc:
            log_event(log, "error", "llm.error", step=step, error=str(exc))
            raise

        latency_ms = round((time.perf_counter() - start) * 1000)
        content = response.content if isinstance(response.content, str) else str(response.content)

        # Token usage is surfaced by the SDK when the provider returns it.
        usage = getattr(response, "usage_metadata", None) or {}
        log_event(
            log,
            "info",
            "llm.invoke",
            step=step,
            model=self._settings.openrouter_model,
            latency_ms=latency_ms,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            chars=len(content),
        )
        return content
