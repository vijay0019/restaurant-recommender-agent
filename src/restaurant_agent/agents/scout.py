"""Scout agent: discover candidate restaurants and gather review evidence.

Exposed as a *node factory* (``make_scout_node``) so its dependencies — the LLM
client, the search client and settings — are injected rather than referenced as
module globals. This keeps the node pure-ish and trivially testable with fakes.
"""

from __future__ import annotations

from collections.abc import Callable

from langchain_core.messages import HumanMessage, SystemMessage

from ..config import Settings
from ..logging_config import get_logger, log_event
from ..parsing import parse_restaurants
from ..schemas import AgentState, Restaurant
from ..tools.llm import LLMClient
from ..tools.search import SearchClient

log = get_logger(__name__)

_EXTRACT_SYSTEM = (
    """You extract a clean list of restaurants from raw web search results. 
    Pick the most relevant, real restaurants for the user's request. 
    Respond with ONLY a JSON array of {name, short_description} objects. 
    Do not wrap it in markdown or add any prose."""
)


def make_scout_node(
    llm: LLMClient,
    search: SearchClient,
    settings: Settings,
) -> Callable[[AgentState], dict]:
    """Build the scout node bound to its dependencies."""

    def scout_agent(state: AgentState) -> dict:
        food, location = state["food"], state["location"]
        log_event(log, "info", "scout.start", food=food, location=location)

        # Step 1 — discover candidates.
        discovery_query = f"best {food} restaurants in {location} with reviews"
        hits = search.search(
            discovery_query,
            search_depth="advanced",
            max_results=settings.scout_max_results,
            step="scout.discover",
        )
        context = "\n".join(
            f"- {h.title}: {h.content[:400]} (source: {h.url})" for h in hits
        )

        prompt = [
            SystemMessage(content=_EXTRACT_SYSTEM),
            HumanMessage(
                content=(
                    f"User wants: {food} in {location}.\n\n"
                    f"Search results:\n{context}\n\n"
                    f"Pick the {settings.num_restaurants} most relevant, real restaurants."
                )
            ),
        ]
        raw = llm.invoke(prompt, step="scout.extract")
        restaurants: list[Restaurant] = parse_restaurants(raw, settings.num_restaurants)
        log_event(log, "info", "scout.candidates", count=len(restaurants))

        # Step 2 — gather review evidence for each candidate.
        for r in restaurants:
            review_hits = search.search(
                f"{r.name} {location} customer reviews ratings",
                search_depth="basic",
                max_results=settings.scout_review_results,
                step="scout.reviews",
            )
            r.reviews = (
                "\n".join(h.content[:500] for h in review_hits if h.content)
                or "No review data found."
            )
            r.sources = [h.url for h in review_hits if h.url]
            log_event(log, "info", "scout.review_gathered", restaurant=r.name, sources=len(r.sources))

        return {"restaurants": restaurants}

    return scout_agent
