"""Critic agent: synthesise a fair, evidence-grounded comparison.

Like the scout, exposed as a node factory so the LLM client is injected.
"""

from __future__ import annotations

from collections.abc import Callable

from langchain_core.messages import HumanMessage, SystemMessage

from ..config import Settings
from ..logging_config import get_logger, log_event
from ..schemas import AgentState
from ..tools.llm import LLMClient

log = get_logger(__name__)

_CRITIC_SYSTEM = (
    """You are a fair, unbiased food critic. Compare the restaurants using "
    ONLY the review evidence provided — never invent ratings, prices, or details 
    not supported by the text.\n
    For EACH restaurant, write these four labeled sections:\n
    • **What reviews are saying** — the recurring themes in the reviews, both 
    positive and negative.\n
    • **Perceived value for money** — what the reviews suggest about price vs. 
    quality. If reviews don't mention price/value, say so plainly.\n
    • **Service & atmosphere** — what reviews say about staff, service speed, and 
    ambiance.\n
    • **Best for** — the type of diner or occasion this place suits, based on the 
    evidence.\n
    After covering all restaurants, end with a short **Note on evidence** that 
    flags any restaurants where the review data was thin or sparse, so the reader 
    knows where the comparison is less reliable."""
)


def make_critic_node(llm: LLMClient, settings: Settings) -> Callable[[AgentState], dict]:
    """Build the critic node bound to its dependencies."""

    def critic_agent(state: AgentState) -> dict:
        restaurants = state["restaurants"]
        log_event(log, "info", "critic.start", count=len(restaurants))

        if not restaurants:
            return {
                "comparison": (
                    "No restaurants could be found for this request. Try a broader "
                    "cuisine or a larger location."
                )
            }

        review_block = "\n\n".join(
            f"### {r.name}\n{r.reviews}" for r in restaurants
        )
        prompt = [
            SystemMessage(content=_CRITIC_SYSTEM),
            HumanMessage(
                content=(
                    f"User wanted: {state['food']} in {state['location']}.\n\n"
                    f"Review evidence:\n{review_block}\n\n"
                    "Produce a clear, fair and structured comparison as instructed."
                )
            ),
        ]
        comparison = llm.invoke(prompt, step="critic.compare")
        log_event(log, "info", "critic.done", chars=len(comparison))
        return {"comparison": comparison}

    return critic_agent
