"""Assemble the Scout -> Critic LangGraph pipeline.

``build_app`` wires the tool clients and node factories together and returns a
compiled graph. Dependencies are constructed once here (composition root) and
injected into the nodes.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from ..config import Settings, get_settings
from ..schemas import AgentState
from ..tools.llm import LLMClient
from ..tools.search import SearchClient
from .critic import make_critic_node
from .scout import make_scout_node


def build_app(settings: Settings | None = None):
    """Build and compile the agent graph.

    Pass ``settings`` to override configuration (e.g. in tests); otherwise the
    cached environment-derived settings are used.
    """
    settings = settings or get_settings()

    llm = LLMClient(settings)
    search = SearchClient(settings)

    graph = StateGraph(AgentState)
    graph.add_node("scout", make_scout_node(llm, search, settings))
    graph.add_node("critic", make_critic_node(llm, settings))

    graph.add_edge(START, "scout")
    graph.add_edge("scout", "critic")
    graph.add_edge("critic", END)

    return graph.compile()
