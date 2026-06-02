"""Unit tests for the robust JSON parser (no network, no API keys)."""

from restaurant_agent.parsing import extract_json_array, parse_restaurants


def test_plain_array():
    text = '[{"name":"A","short_description":"x"},{"name":"B"}]'
    assert [d["name"] for d in extract_json_array(text)] == ["A", "B"]


def test_fenced_block():
    text = '```json\n[{"name":"C"}]\n```'
    assert [d["name"] for d in extract_json_array(text)] == ["C"]


def test_prose_around_json():
    text = 'Sure!\n[{"name":"D","short_description":"nice"}]\nHope that helps.'
    assert [d["name"] for d in extract_json_array(text)] == ["D"]


def test_trailing_comma_recovered():
    text = '[{"name":"E"},]'
    assert [d["name"] for d in extract_json_array(text)] == ["E"]


def test_brackets_inside_strings():
    text = '[{"name":"F [special]","short_description":"has [x]"}]'
    assert [d["name"] for d in extract_json_array(text)] == ["F [special]"]


def test_object_not_array_returns_empty():
    assert extract_json_array('{"name":"Z"}') == []


def test_garbage_and_empty():
    assert extract_json_array("no json here") == []
    assert extract_json_array("") == []


def test_parse_restaurants_validates_and_limits():
    text = '[{"name":"A"},{"name":""},{"name":"B"},{"name":"C"}]'
    out = parse_restaurants(text, limit=2)
    assert [r.name for r in out] == ["A", "B"]  # blank name skipped, capped at 2
    assert out[0].reviews == "No review data found."
