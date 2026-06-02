# Restaurant Recommendation Agent

A LangGraph multi-agent system that finds restaurants and produces an
**evidence-grounded** comparison from real web reviews.

```
START → scout → critic → END
        │        │
        │        └─ synthesises a fair, citation-bounded comparison
        └─ discovers candidates (Tavily) + gathers per-restaurant reviews
```

- **Scout agent** — searches the web using Tavily, extracts a clean list of real
  restaurants from messy results, then gathers review evidence for each.
- **Critic agent** — writes a structured, unbiased comparison using **only** the
  gathered evidence, flagging where review data is thin.

## Setup

```bash
pip install -e ".[dev]"
cp .env.example .env   # then fill in OPENROUTER_API_KEY and TAVILY_API_KEY
```

## Run

```bash
# positional args
python -m restaurant_agent.cli "vegetarian pasta" "Birmingham"

# JSON output
python -m restaurant_agent.cli --json "ramen" "Seattle"

# interactive (prompts if args omitted)
python -m restaurant_agent.cli

# installed console script
restaurant-agent "tacos" "Austin"
```

Programmatic use:

```python
from restaurant_agent import RecommendRequest, recommend

result = recommend(RecommendRequest(food="ramen", location="Seattle"))
print(result.comparison)
```

## Logging

Set in `.env`:

- `LOG_LEVEL` — `DEBUG | INFO | WARNING | ERROR`
- `LOG_FORMAT` — `text` (human) or `json` (machine/aggregator-friendly)
- `LOG_FILE` — optional path to also append logs to a file

Each agent step emits a structured event (`scout.discover`, `llm.invoke`,
`search.query`, …) with timing and, where available, token usage.

## Tests

```bash
pytest
```

## Project layout

```
src/restaurant_agent/
├── config.py          # pydantic-settings configuration
├── schemas.py         # Pydantic models + LangGraph state
├── logging_config.py  # self-hosted structured logging
├── parsing.py         # robust JSON-array extraction (no tool calling needed)
├── service.py         # recommend(): reusable entry point
├── cli.py             # command-line interface
├── tools/
│   ├── llm.py         # OpenRouter wrapper (retries + logging)
│   └── search.py      # Tavily wrapper (retries + logging)
└── agents/
    ├── scout.py       # discovery + review gathering (node factory)
    ├── critic.py      # evidence-grounded comparison (node factory)
    └── graph.py       # composition root: builds & compiles the graph
```
