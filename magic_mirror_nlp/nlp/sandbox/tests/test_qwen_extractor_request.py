import json
from unittest.mock import patch

from sandbox.extraction.qwen_extractor import QwenExtractor


def test_request_is_compact_bounded_and_keeps_model_resident():
    extractor = QwenExtractor()
    response = {"message": {"content": json.dumps({"coreference_resolutions": [{"text_reference": "apple", "resolved_entity": "apple"}]})}, "eval_count": 3}
    with patch("sandbox.extraction.qwen_extractor.ollama.chat", return_value=response) as chat:
        result, _ = extractor.extract("Create a red apple above my right palm and make it blue", "", False)

    request = chat.call_args.kwargs
    assert request["format"] == extractor.schema_dict
    assert request["format"]["type"] == "object"
    assert request["keep_alive"] == "30m"
    assert request["options"]["temperature"] == 0.0
    assert request["options"]["num_predict"] == 384
    assert "modifiers" in request["messages"][0]["content"]
    assert "semantic_graph" in request["messages"][0]["content"]
    assert "do not repeat stale values" in request["messages"][0]["content"]
    assert "coreference" in request["messages"][0]["content"].lower()
    assert "red apple" in request["messages"][1]["content"]
    assert "right palm" in request["messages"][1]["content"]
    assert isinstance(result, dict)
    assert result["coreference_resolutions"][0]["text_reference"] == "it"


def test_context_compaction_keeps_recent_graph_and_coreference_state():
    state = {
        "text_segment": "Make it blue",
        "semantic_graph": {
            "entities": [{"id": f"e{i}", "name": f"object{i}"} for i in range(20)],
            "relationships": [{"subject": "e19", "predicate": "on", "object": "hand"}],
            "quantities": [{"entity": "e19", "amount": 2}],
        },
        "coreference_resolutions": [{"text_reference": "it", "resolved_entity": "e19"}],
        "explanation_state": {"active_concept": "object19"},
        "unused_large_field": "x" * 5000,
    }
    compact = QwenExtractor._compact_context(json.dumps(state))
    parsed = json.loads(compact)
    assert len(parsed["semantic_graph"]["entities"]) == 12
    assert parsed["semantic_graph"]["entities"][-1]["id"] == "e19"
    assert parsed["coreference_resolutions"][0]["resolved_entity"] == "e19"
    assert parsed["explanation_state"]["active_concept"] == "object19"
    assert "unused_large_field" not in compact


def test_update_coreference_targets_active_visual_and_replaces_old_values():
    previous = {
        "semantic_graph": {"entities": [
            {"id": "apple", "name": "apple"},
            {"id": "right_palm", "name": "right palm"},
        ]},
        "explanation_state": {"active_concept": "apple"},
    }
    result = {
        "semantic_graph": {"relationships": [
            {"subject": "apple", "predicate": "color", "object": "red", "type": "property"},
            {"subject": "apple", "predicate": "color", "object": "blue", "type": "property"},
            {"subject": "apple", "predicate": "location", "object": "right_palm", "type": "property"},
            {"subject": "apple", "predicate": "location", "object": "left_hand", "type": "property"},
        ]},
        "coreference_resolutions": [{"text_reference": "apple", "resolved_entity": "right_palm"}],
    }
    QwenExtractor._normalize_references_and_updates(
        result, json.dumps(previous), "Move it to my left hand and make it blue"
    )
    assert result["coreference_resolutions"] == [{"text_reference": "it", "resolved_entity": "apple"}]
    assert {(r["predicate"], r["object"]) for r in result["semantic_graph"]["relationships"]} == {
        ("color", "blue"), ("location", "left_hand")
    }


def test_red_apple_keeps_base_concept_and_explicit_color_modifier():
    result = {"semantic_graph": {
        "entities": [{"id": "apple", "name": "apple", "class": "object", "status": "new_explicit"}],
        "relationships": [],
    }}
    QwenExtractor._ensure_color_modifiers(result, "Create a red apple above my right palm")
    assert any(e["id"] == "apple" and e["name"] == "apple" for e in result["semantic_graph"]["entities"])
    assert any(e["id"] == "red" and e["name"] == "red" for e in result["semantic_graph"]["entities"])
    assert {"subject": "apple", "predicate": "color", "object": "red", "type": "property"} in result["semantic_graph"]["relationships"]
