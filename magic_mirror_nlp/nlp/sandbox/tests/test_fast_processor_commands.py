import time

import pytest

from sandbox.extraction.fast_processor import parse_simple_command


@pytest.mark.parametrize("text,verb", [
    ("show me an apple", "show me"),
    ("create a red apple", "create"),
    ("move apple on my hand", "move"),
    ("animate apple to rotate", "animate"),
    ("remove apple", "remove"),
])
def test_explicit_lifecycle_commands_are_parsed(text, verb):
    result = parse_simple_command(text)
    assert result is not None
    assert result["semantic_graph"]["entities"][0]["name"] == "apple"


def test_modifier_is_separate_from_concept():
    result = parse_simple_command("show me a red apple")
    entity = result["semantic_graph"]["entities"][0]
    assert entity["name"] == "apple"
    assert result["semantic_graph"]["relationships"] == [
        {"subject": "apple", "predicate": "color", "object": "red", "type": "property", "confidence": 1.0}
    ]


def test_supported_spatial_anchor_is_preserved():
    result = parse_simple_command("show me a red apple on my hand")
    assert result["semantic_graph"]["relationships"][-1]["object"] == "hand"
    assert "anchored on hand" in result["explicit_information"][-1]


@pytest.mark.parametrize("text", [
    "move it left", "show me an apple and a pear", "do not remove apple",
    "put apple behind the box", "make it rotate", "show me apple, maybe",
    "update apple", "move apple", "animate apple",
])
def test_coreference_negation_relations_and_ambiguity_fall_back(text):
    assert parse_simple_command(text) is None


def test_deterministic_parser_latency_under_100ms():
    samples = []
    for _ in range(1000):
        start = time.perf_counter_ns()
        parse_simple_command("show me a red apple on my hand")
        samples.append((time.perf_counter_ns() - start) / 1_000_000)
    assert max(samples) < 100
