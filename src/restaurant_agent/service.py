"""High-level entry point that ties the graph to validated I/O.

This is the single reusable function the CLI uses today and the FastAPI layer
will use in Phase 5 — keeping the orchestration in one place.
"""

from __future__ import annotations

import time

from .agents.graph import build_app
from .config import Settings, get_settings
from .logging_config import get_logger, log_event
from .schemas import RecommendRequest, RecommendResponse, Restaurant

log = get_logger(__name__)


def recommend(
    request: RecommendRequest,
    settings: Settings | None = None,
) -> RecommendResponse:
    """Run the full Scout -> Critic pipeline for a validated request."""
    settings = settings or get_settings()
    app = build_app(settings)

    log_event(log, "info", "run.start", food=request.food, location=request.location)
    start = time.perf_counter()

    final = app.invoke(
        {
            "food": request.food,
            "location": request.location,
            "restaurants": [],
            "comparison": "",
        }
    )

    restaurants = [
        r if isinstance(r, Restaurant) else Restaurant(**r)
        for r in final.get("restaurants", [])
    ]
    elapsed_ms = round((time.perf_counter() - start) * 1000)
    log_event(
        log,
        "info",
        "run.done",
        restaurants=len(restaurants),
        elapsed_ms=elapsed_ms,
    )

    return RecommendResponse(
        food=request.food,
        location=request.location,
        restaurants=restaurants,
        comparison=final.get("comparison", ""),
    )
