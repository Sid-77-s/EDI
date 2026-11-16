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
