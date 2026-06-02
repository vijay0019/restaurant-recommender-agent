"""Restaurant recommendation agent.

A LangGraph multi-agent pipeline (Scout -> Critic) that finds restaurants,
gathers review evidence, and produces an evidence-grounded comparison.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .schemas import RecommendRequest, RecommendResponse, Restaurant
from .service import recommend

__all__ = [
    "RecommendRequest",
    "RecommendResponse",
    "Restaurant",
    "recommend",
    "__version__",
]
