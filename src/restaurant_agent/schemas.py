"""Typed data models for the agent pipeline.

The LangGraph *state* is a ``TypedDict`` (LangGraph's native state container,
which is merged shallowly between nodes), while the domain objects flowing
through that state are Pydantic models so they are validated at the boundary
where we parse messy LLM / web output.
"""

from __future__ import annotations

from typing import TypedDict

from pydantic import BaseModel, Field


class Restaurant(BaseModel):
    """A single candidate restaurant and its gathered review evidence."""

    name: str
    short_description: str = ""
    reviews: str = "No review data found."
    sources: list[str] = Field(default_factory=list)


class AgentState(TypedDict):
    """Shared state passed between graph nodes.

    Note: ``restaurants`` holds validated ``Restaurant`` instances rather than
    raw dicts, so every downstream node can rely on the shape.
    """

    food: str
    location: str
    restaurants: list[Restaurant]
    comparison: str


class RecommendRequest(BaseModel):
    """Validated user input for a recommendation run (used by CLI and, later, the API)."""

    food: str = Field(..., min_length=1, max_length=200)
    location: str = Field(..., min_length=1, max_length=200)


class RecommendResponse(BaseModel):
    """The final result of a pipeline run."""

    food: str
    location: str
    restaurants: list[Restaurant]
    comparison: str
