"""Command-line entry point.

Examples
--------
    python -m restaurant_agent.cli "vegetarian pasta" "Birmingham"
    python -m restaurant_agent.cli --json "ramen" "Seattle"

If food/location are omitted, the user is prompted interactively (preserving the
original notebook's behaviour).
"""

from __future__ import annotations

import argparse
import sys

from pydantic import ValidationError

from .config import get_settings
from .logging_config import configure_logging, get_logger, log_event
from .schemas import RecommendRequest
from .service import recommend

log = get_logger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="restaurant-agent",
        description="Find restaurants and produce an evidence-grounded comparison.",
    )
    parser.add_argument("food", nargs="?", help="Cuisine or dish, e.g. 'vegetarian pasta'.")
    parser.add_argument("location", nargs="?", help="City or area, e.g. 'Birmingham'.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the full result as JSON instead of formatted text.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    settings = get_settings()
    configure_logging(
        level=settings.log_level,
        fmt=settings.log_format,
        log_file=settings.log_file,
    )

    food = args.food or input("What would you like to eat? (cuisine/dish): ").strip()
    location = args.location or input("Where? (city/area): ").strip()

    try:
        request = RecommendRequest(food=food, location=location)
    except ValidationError as exc:
        print(f"Invalid input: {exc}", file=sys.stderr)
        return 2

    try:
        result = recommend(request)
    except Exception as exc:  # surface a clean message; full trace is logged
        log_event(log, "error", "cli.failed", error=str(exc))
        print(f"\nSomething went wrong: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(result.model_dump_json(indent=2))
    else:
        print("\n" + "=" * 70)
        print("RESTAURANT COMPARISON")
        print("=" * 70 + "\n")
        print(result.comparison)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
