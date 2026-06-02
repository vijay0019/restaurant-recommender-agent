"""Integration test of the Scout -> Critic graph with fakes (no network)."""

from restaurant_agent.agents.critic import make_critic_node
from restaurant_agent.agents.scout import make_scout_node
from restaurant_agent.config import Settings
from restaurant_agent.schemas import AgentState
from restaurant_agent.tools.search import SearchResult


def _settings() -> Settings:
    return Settings(
        openrouter_api_key="test",
        tavily_api_key="test",
        num_restaurants=2,
        scout_max_results=3,
        scout_review_results=2,
    )


class FakeLLM:
    """Returns canned responses keyed by the call ``step``."""

    def __init__(self):
        self.calls = []

    def invoke(self, messages, *, step="llm"):
        self.calls.append(step)
        if step == "scout.extract":
            return '[{"name":"Alpha","short_description":"cosy"},{"name":"Beta"}]'
        if step == "critic.compare":
            return "## Comparison\nAlpha vs Beta, grounded in evidence."
        return ""


class FakeSearch:
    def search(self, query, *, search_depth="basic", max_results=5, step="search"):
        return [
            SearchResult(title=f"{query} result", content="Great food, slow service.", url="http://x")
        ]


def test_pipeline_runs_end_to_end():
    settings = _settings()
    llm, search = FakeLLM(), FakeSearch()

    scout = make_scout_node(llm, search, settings)
    critic = make_critic_node(llm, settings)

    state: AgentState = {"food": "pasta", "location": "Rome", "restaurants": [], "comparison": ""}
    state = {**state, **scout(state)}

    assert [r.name for r in state["restaurants"]] == ["Alpha", "Beta"]
    assert all(r.reviews and r.sources for r in state["restaurants"])

    state = {**state, **critic(state)}
    assert "Comparison" in state["comparison"]
    assert llm.calls == ["scout.extract", "critic.compare"]


def test_critic_handles_no_restaurants():
    settings = _settings()
    critic = make_critic_node(FakeLLM(), settings)
    state: AgentState = {"food": "x", "location": "y", "restaurants": [], "comparison": ""}
    out = critic(state)
    assert "No restaurants" in out["comparison"]
