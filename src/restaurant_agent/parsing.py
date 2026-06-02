"""Robust extraction of a JSON array of restaurants from free-form LLM text.

The free OpenRouter model does not support ``with_structured_output`` / tool
calling, so we cannot rely on the SDK to hand us structured data. Instead we
defensively parse the model's text response. The original notebook used a single
``re.search(r"\\[.*\\]")`` which breaks on common LLM quirks (markdown code
fences, prose around the JSON, trailing commas, nested brackets). This module
layers several strategies and validates the result into ``Restaurant`` models,
skipping anything malformed rather than crashing the whole run.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .logging_config import get_logger, log_event
from .schemas import Restaurant

log = get_logger(__name__)

# Matches ```json ... ``` or ``` ... ``` fenced blocks.
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _strip_code_fences(text: str) -> str:
    match = _FENCE_RE.search(text)
    return match.group(1).strip() if match else text.strip()


def _remove_trailing_commas(text: str) -> str:
    """Remove trailing commas before ``]`` or ``}`` (a very common LLM JSON error)."""
    return re.sub(r",(\s*[\]}])", r"\1", text)


def _find_balanced_array(text: str) -> str | None:
    """Return the first balanced ``[...]`` substring, respecting strings/escapes.

    A naive ``\\[.*\\]`` is greedy and brittle; this scans bracket depth while
    ignoring brackets that appear inside JSON string literals.
    """
    start = text.find("[")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _attempt_loads(text: str) -> Any | None:
    for candidate in (text, _remove_trailing_commas(text)):
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
    return None


def extract_json_array(text: str) -> list[dict[str, Any]]:
    """Extract a list of dicts from LLM text using layered fallbacks.

    Returns an empty list if nothing usable is found.
    """
    if not text:
        return []

    cleaned = _strip_code_fences(text)

    # Strategy 1: the whole (de-fenced) response is valid JSON.
    parsed = _attempt_loads(cleaned)

    # Strategy 2: locate the first balanced JSON array within the text.
    if not isinstance(parsed, list):
        array_str = _find_balanced_array(cleaned)
        if array_str is not None:
            parsed = _attempt_loads(array_str)

    if not isinstance(parsed, list):
        log_event(log, "warning", "parsing.no_array", preview=cleaned[:160])
        return []

    return [item for item in parsed if isinstance(item, dict)]


def parse_restaurants(text: str, limit: int) -> list[Restaurant]:
    """Parse and validate up to ``limit`` restaurants from LLM text.

    Malformed entries are skipped (and logged) so one bad item doesn't sink the
    whole batch. An entry must at least have a usable ``name``.
    """
    raw_items = extract_json_array(text)
    restaurants: list[Restaurant] = []

    for item in raw_items:
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        try:
            restaurants.append(
                Restaurant(
                    name=name,
                    short_description=str(item.get("short_description", "")).strip(),
                )
            )
        except Exception as exc:  # pragma: no cover - defensive
            log_event(log, "warning", "parsing.invalid_item", error=str(exc), item=item)
        if len(restaurants) >= limit:
            break

    log_event(
        log,
        "info",
        "parsing.restaurants",
        found=len(restaurants),
        raw=len(raw_items),
    )
    return restaurants
