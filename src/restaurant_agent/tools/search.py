"""Thin, observable wrapper around the Tavily web search client.

Normalises Tavily's raw response into a small list of ``SearchResult`` records,
adds retry/backoff for transient failures, logs timing, and degrades gracefully
to an empty list instead of raising into the agent logic.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from tavily import TavilyClient
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import Settings
from ..logging_config import get_logger, log_event

log = get_logger(__name__)


@dataclass(frozen=True)
class SearchResult:
    """A single normalised search hit."""

    title: str
    content: str
    url: str


class SearchClient:
    """Wraps ``TavilyClient`` with normalisation, retries and logging."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._client = TavilyClient(api_key=settings.tavily_api_key)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    def _raw_search(self, query: str, search_depth: str, max_results: int) -> dict:
        return self._client.search(
            query=query,
            search_depth=search_depth,
            max_results=max_results,
        )

    def search(
        self,
        query: str,
        *,
        search_depth: str = "basic",
        max_results: int = 5,
        step: str = "search",
    ) -> list[SearchResult]:
        """Run a search and return normalised results (empty list on failure)."""
        start = time.perf_counter()
        try:
            raw = self._raw_search(query, search_depth, max_results)
        except Exception as exc:
            # Search is best-effort: log and return nothing rather than crash the run.
            log_event(log, "error", "search.error", step=step, query=query, error=str(exc))
            return []

        results = [
            SearchResult(
                title=str(item.get("title", "")),
                content=str(item.get("content", "")),
                url=str(item.get("url", "")),
            )
            for item in raw.get("results", [])
        ]
        latency_ms = round((time.perf_counter() - start) * 1000)
        log_event(
            log,
            "info",
            "search.query",
            step=step,
            query=query,
            depth=search_depth,
            results=len(results),
            latency_ms=latency_ms,
        )
        return results
