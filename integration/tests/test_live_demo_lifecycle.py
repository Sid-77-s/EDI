"""Hardware-free speech-to-render lifecycle through the application runtime."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'MagicMirror_Visualisation'))
sys.path.insert(0, str(ROOT / 'magic_mirror_nlp' / 'nlp' / 'sandbox'))

from integration.generation.engine import GenerationEngine
from integration.generation.renderer3d import SoftwareRenderer3D
from integration.runtime.multimodal_runtime import MultimodalRuntime
from integration_live_demo import LiveState
from extraction.fast_processor import parse_simple_command
from MagicMirror_Visualisation.src.perception.assets import ConceptAssetStore, render_box_and_asset

FIXTURE = "integration/tests/fixtures/glb_real_world/BoxTextured.glb"


def _nlp_json(text, *, entity=True, intent="demonstrating", relationships=None):
    return json.dumps({
        "timestamp": 12345.0, "is_partial": False, "text_segment": text,
        "speaker_activity": [{"intent": intent, "confidence": 1.0}],
        "semantic_graph": {
            "entities": ([{"id": "fixture_box", "name": "fixture box", "class": "object", "status": "new_explicit"}]
                         if entity else []),
            "relationships": relationships or [], "quantities": [],
        },
        "explicit_information": [], "inferred_information": [],
        "coreference_resolutions": [],
        "explanation_state": {"current_topic": "fixture box", "active_concept": "fixture box", "recent_entities": ["fixture box"]},
    })


def _command_payload(text):
    payload = parse_simple_command(text)
    assert payload is not None, f"Expected deterministic FastProcessor command for: {text}"
    return payload


def _vision_scene():
    return SimpleNamespace(
        timestamp=12345.0, frame_id=1,
        activity=SimpleNamespace(
            left_hand_gesture=SimpleNamespace(value="none"),
            right_hand_gesture=SimpleNamespace(value="none"), motion="none"),
        relationships=[],
        visual_opportunity=SimpleNamespace(
            available=True, type=SimpleNamespace(value="on_hand"),
            bbox=(0.55, 0.45, 0.8, 0.75), center=(0.675, 0.6)),
    )


def test_spoken_lifecycle_local_glb_update_move_remove_and_fallback(tmp_path):
    pack = tmp_path / "pack"
    pack.mkdir()
    glb = pack / "fixture-box.glb"
    glb.write_bytes(open(FIXTURE, "rb").read())
    (pack / "fixture-box.json").write_text(json.dumps({
        "name": "fixture box", "aliases": ["box fixture"], "license": "CC0",
        "source": "bundled test fixture", "author": "test", "attribution": "fixture",
    }), encoding="utf-8")

    store = ConceptAssetStore()
    generation = GenerationEngine(asset_store=store, asset_dir=tmp_path / "generated", asset_pack_dir=pack)
    runtime = MultimodalRuntime()
    live = LiveState(generation_engine=generation)
    try:
        # CREATE: deterministic finalized command -> normalized runtime -> plan -> generation/store/render.
        create = _command_payload("show me fixture box")
        create_plan = runtime.process_nlp_update(json.dumps(create))
        assert create_plan.action.value == "CREATE"
        assert create_plan.visual_id == "fixture_box"
        live.update_plan(create_plan)
        asset = store.get_asset("fixture box")
        assert asset and asset.is_3d
        model = asset.model_3d
        assert model.visual_id == create_plan.visual_id
        assert model.metadata["license"] == "CC0"
        assert generation.get_job_status("fixture box").provider == "instant_3d"
        assert model.metadata["asset_source"].endswith("fixture-box.glb")
        frame = np.zeros((320, 480, 3), dtype=np.uint8)
        render_box_and_asset(frame, (0.2, 0.2, 0.8, 0.85), "fixture box", asset)
        assert np.count_nonzero(frame) > 0
        render_frame = np.zeros((256, 256, 3), dtype=np.uint8)
        assert SoftwareRenderer3D().render(render_frame, model, bbox=(0, 0, 1, 1), t=0.0)

        # UPDATE: deterministic finalized extractor result for an utterance that FastProcessor
        # routes to its local Qwen fallback; runtime/planner/state/generation remain real.
        update_plan = runtime.process_nlp_update(_nlp_json(
            "Change fixture box to red.", relationships=[{
                "subject": "fixture_box", "predicate": "color", "object": "red",
                "type": "property", "confidence": 1.0,
            }]))
        assert update_plan.action.value == "UPDATE"
        assert update_plan.visual_id == create_plan.visual_id
        live.update_plan(update_plan)
        assert store.get_asset("fixture box").model_3d is model
        assert model.metadata.get("asset_source", "").endswith("fixture-box.glb")
        meshes = [mesh for mesh, _ in model.get_all_meshes_with_transforms()]
        assert meshes and all(mesh.material.diffuse_color == (0, 0, 220) for mesh in meshes)
        assert generation.get_job_status("fixture box").result.model_3d is model

        # MOVE: finalized deterministic command plus an existing vision anchor moves the same model.
        runtime.process_vision_update(_vision_scene())
        move_plan = runtime.process_nlp_update(json.dumps(_command_payload("move fixture box on my hand")))
        assert move_plan.action.value == "MOVE"
        assert move_plan.visual_id == create_plan.visual_id
        assert move_plan.spatial.anchored and move_plan.spatial.position is not None
        prior_version = model.version
        live.update_plan(move_plan)
        assert store.get_asset("fixture box").model_3d is model
        assert model.visual_id == create_plan.visual_id
        assert np.allclose(model.root.position, move_plan.spatial.position)
        assert model.version == prior_version + 1
        assert generation.get_job_status("fixture box").result.model_3d is model

        # REMOVE: command plan removes both active representation and renderable store entry.
        remove_plan = runtime.process_nlp_update(json.dumps(_command_payload("remove fixture box")))
        assert remove_plan.action.value == "REMOVE"
        assert remove_plan.visual_id == create_plan.visual_id
        live.update_plan(remove_plan)
        assert live.active_concept is None and live.active_visual_id is None
        assert generation.instant_engine._active_visuals.get(create_plan.visual_id) is None
        assert store.get_asset("fixture box").is_3d is False

        # An unverified concept reports unavailable instead of showing the generic fallback shape.
        fallback_plan = runtime.process_nlp_update(json.dumps(_command_payload("show me missing widget")))
        assert fallback_plan.action.value == "CREATE"
        live.update_plan(fallback_plan)
        unavailable = generation.get_job_status("missing widget")
        assert unavailable.status.value == "failed"
        assert unavailable.result.metadata["representation_fidelity"] == "unavailable"
    finally:
        generation.shutdown(wait=True)
        generation.instant_engine.shutdown()




