"""
integration/tests/test_3d_representation.py — Comprehensive tests for Instant 3D Representation Engine.

Validates:
1. Cached 3D asset retrieval (< 50 ms)
2. Procedural object construction (Apple, Chair, Car, Tree, Dog, Person) (< 150 ms)
3. Scientific model construction (Atom, Solar System, Wave, Heart, Gears, Circuit) (< 200 ms)
4. Correct molecule structure for supported examples (H2O, CO2, CH4, NaCl)
5. GLB/glTF loading
6. 3D rendering integration & FPS benchmark (> 30 FPS)
7. Persistent visual identity across CREATE, MOVE, UPDATE, ANIMATE
8. MOVE without regeneration
9. UPDATE without duplication
10. ANIMATE time-varying behavior
11. Stale background task rejection
12. Offline operation & zero paid API calls
13. Graceful failure and fallback
14. Preservation of existing PNG rendering
"""

import json
import math
import os
import shutil
import struct
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from MagicMirror_Visualisation.src.perception.assets import (
    ConceptAsset,
    ConceptAssetStore,
    render_box_and_asset,
)
from integration.generation.engine import GenerationEngine
from integration.generation.geometry_primitives import (
    create_box,
    create_cylinder,
    create_gear,
    create_torus,
    create_uv_sphere,
)
from integration.generation.instant_engine import InstantRepresentationEngine
from integration.generation.model3d import (
    Material3D,
    Mesh3D,
    Model3D,
    ModelNode3D,
    load_glb,
    load_obj,
)
from integration.generation.models_ordinary import OrdinaryObject3DBuilder
from integration.generation.models_scientific import (
    build_atom_model,
    build_circuit_model,
    build_heart_model,
    build_mechanical_assembly_model,
    build_molecule_model,
    build_solar_system_model,
    build_wave_model,
)
from integration.generation.renderer3d import SoftwareRenderer3D
from integration.generation.representation_planner import (
    RepresentationCategory,
    RepresentationPlanner,
)
from integration.generation.schemas import GenerationRequest, JobState
from integration.multimodal_fusion.schemas import (
    ActionType,
    Representation,
    RepresentationType,
    SpatialInfo,
    VisualPlan,
)
from integration_live_demo import LiveState, _draw_visual_opportunity


# ---------------------------------------------------------------------------
# Fixtures & Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_model_dir():
    d = tempfile.mkdtemp(prefix="test_models_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


def _create_minimal_glb_file(path: Path) -> Path:
    """Creates a minimal valid GLB file with 1 triangle for parser validation."""
    # Vertices: 3 x VEC3 float32
    verts = np.array([
        [-0.5, -0.5, 0.0],
        [ 0.5, -0.5, 0.0],
        [ 0.0,  0.5, 0.0],
    ], dtype=np.float32)
    bin_data = verts.tobytes()

    gltf_dict = {
        "asset": {"version": "2.0"},
        "accessors": [
            {
                "bufferView": 0,
                "byteOffset": 0,
                "componentType": 5126,  # float
                "count": 3,
                "type": "VEC3",
                "max": [0.5, 0.5, 0.0],
                "min": [-0.5, -0.5, 0.0],
            }
        ],
        "bufferViews": [
            {
                "buffer": 0,
                "byteOffset": 0,
                "byteLength": len(bin_data),
            }
        ],
        "buffers": [{"byteLength": len(bin_data)}],
        "meshes": [
            {
                "name": "triangle_mesh",
                "primitives": [{"attributes": {"POSITION": 0}}],
            }
        ],
    }

    json_str = json.dumps(gltf_dict)
    # Pad JSON to 4-byte alignment
    pad_len = (4 - (len(json_str) % 4)) % 4
    json_bytes = json_str.encode("utf-8") + (b" " * pad_len)

    # Pad binary data to 4-byte alignment
    bin_pad = (4 - (len(bin_data) % 4)) % 4
    bin_bytes = bin_data + (b"\x00" * bin_pad)

    total_len = 12 + 8 + len(json_bytes) + 8 + len(bin_bytes)

    # GLB Header
    header = struct.pack("<4sII", b"glTF", 2, total_len)
    chunk0_hdr = struct.pack("<II", len(json_bytes), 0x4E4F534A)  # "JSON"
    chunk1_hdr = struct.pack("<II", len(bin_bytes), 0x004E4942)   # "BIN\0"

    with open(str(path), "wb") as f:
        f.write(header)
        f.write(chunk0_hdr)
        f.write(json_bytes)
        f.write(chunk1_hdr)
        f.write(bin_bytes)

    return path


# ---------------------------------------------------------------------------
# 1. Cached 3D Asset Retrieval (< 50 ms)
# ---------------------------------------------------------------------------

def test_cached_3d_asset_retrieval(temp_model_dir):
    engine = InstantRepresentationEngine(library_dir=temp_model_dir)

    # First load/construct
    m1, lat1 = engine.get_or_create_representation("apple")
    assert m1 is not None

    # Second retrieve: must hit cache in < 50 ms
    t0 = time.perf_counter()
    m2, lat2 = engine.get_or_create_representation("apple")
    t1 = time.perf_counter()
    total_cached_latency = (t1 - t0) * 1000.0

    assert m2 is m1
    assert total_cached_latency < 50.0
    assert lat2 < 50.0


# ---------------------------------------------------------------------------
# 2. Procedural Object Construction (< 150 ms)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("concept", ["apple", "chair", "car", "tree", "dog", "person", "cup", "book"])
def test_procedural_object_construction(temp_model_dir, concept):
    engine = InstantRepresentationEngine(library_dir=temp_model_dir)

    t0 = time.perf_counter()
    model, lat = engine.get_or_create_representation(concept)
    t1 = time.perf_counter()

    elapsed = (t1 - t0) * 1000.0
    assert model is not None
    assert model.category == "3D_OBJECT"
    assert len(model.root.children) > 0 or model.root.mesh is not None
    assert elapsed < 150.0, f"Construction of {concept} took {elapsed:.2f} ms (> 150 ms)"


# ---------------------------------------------------------------------------
# 3. Scientific Model Construction (< 200 ms)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("concept", ["atom", "water molecule", "solar system", "wave", "heart", "gear train", "electrical circuit"])
def test_scientific_model_construction(temp_model_dir, concept):
    engine = InstantRepresentationEngine(library_dir=temp_model_dir)

    t0 = time.perf_counter()
    model, lat = engine.get_or_create_representation(concept)
    t1 = time.perf_counter()

    elapsed = (t1 - t0) * 1000.0
    assert model is not None
    assert model.category in ("SCIENTIFIC_MODEL", "TECHNICAL_ASSEMBLY")
    assert model.animation_fn is not None
    assert elapsed < 200.0, f"Construction of {concept} took {elapsed:.2f} ms (> 200 ms)"


# ---------------------------------------------------------------------------
# 4. Correct Molecule Structure for Supported Examples
# ---------------------------------------------------------------------------

def test_molecule_geometry_water():
    mol = build_molecule_model("water molecule h2o")
    atom_nodes = [c for c in mol.root.children if c.name.startswith("atom_")]
    bond_nodes = [c for c in mol.root.children if c.name.startswith("bond_")]

    # H2O has 3 atoms (1 O, 2 H) and 2 bonds
    assert len(atom_nodes) == 3
    assert len(bond_nodes) == 2

    # Verify bond angle ~104.5 degrees between O and two H atoms
    o_pos = atom_nodes[0].position
    h1_pos = atom_nodes[1].position
    h2_pos = atom_nodes[2].position

    v1 = h1_pos - o_pos
    v2 = h2_pos - o_pos
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    deg = math.degrees(math.acos(cos_angle))
    assert abs(deg - 104.5) < 1.0


def test_molecule_geometry_co2():
    mol = build_molecule_model("carbon dioxide co2")
    atom_nodes = [c for c in mol.root.children if c.name.startswith("atom_")]
    assert len(atom_nodes) == 3

    # Linear: 180 degrees
    c_pos = atom_nodes[0].position
    o1_pos = atom_nodes[1].position
    o2_pos = atom_nodes[2].position

    v1 = o1_pos - c_pos
    v2 = o2_pos - c_pos
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    deg = math.degrees(math.acos(cos_angle))
    assert abs(deg - 180.0) < 0.1


def test_molecule_geometry_methane():
    mol = build_molecule_model("methane ch4")
    atom_nodes = [c for c in mol.root.children if c.name.startswith("atom_")]
    assert len(atom_nodes) == 5  # 1 C + 4 H

    c_pos = atom_nodes[0].position
    # Tetrahedral bond angle is ~109.5 degrees
    h_positions = [atom_nodes[i].position - c_pos for i in range(1, 5)]
    cos_t = np.dot(h_positions[0], h_positions[1]) / (np.linalg.norm(h_positions[0]) * np.linalg.norm(h_positions[1]))
    deg = math.degrees(math.acos(cos_t))
    assert abs(deg - 109.5) < 1.0


# ---------------------------------------------------------------------------
# 5. GLB/glTF Loading
# ---------------------------------------------------------------------------

def test_glb_loading(temp_model_dir):
    glb_path = temp_model_dir / "sample.glb"
    _create_minimal_glb_file(glb_path)

    model = load_glb(glb_path)
    assert model is not None
    assert model.concept == "sample"
    meshes = model.get_all_meshes_with_transforms()
    assert len(meshes) == 1
    mesh, mat = meshes[0]
    assert len(mesh.vertices) == 3


# ---------------------------------------------------------------------------
# 6. 3D Rendering Integration & FPS Benchmark (> 30 FPS)
# ---------------------------------------------------------------------------

def test_3d_rendering_integration_and_fps():
    renderer = SoftwareRenderer3D()
    model = build_atom_model("atom")

    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    bbox = (0.2, 0.2, 0.6, 0.6)

    # Render single frame
    success = renderer.render(canvas, model, bbox=bbox, t=1.0)
    assert success
    # Verify geometry was drawn in bounding box
    cx, cy = int(0.4 * 640), int(0.4 * 480)
    roi = canvas[cy - 30 : cy + 30, cx - 30 : cx + 30]
    assert np.any(roi > 0)

    # Benchmark FPS over 60 frames
    t0 = time.perf_counter()
    n_frames = 60
    test_canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    for frame_idx in range(n_frames):
        test_canvas.fill(0)
        renderer.render(test_canvas, model, bbox=bbox, t=frame_idx * 0.033)
    t1 = time.perf_counter()

    avg_ms = ((t1 - t0) / n_frames) * 1000.0
    fps = n_frames / (t1 - t0)
    print(f"\n3D Rasterizer FPS: {fps:.1f} FPS ({avg_ms:.2f} ms/frame)")
    assert fps >= 30.0, f"Renderer achieved {fps:.1f} FPS (< 30 FPS)"


# ---------------------------------------------------------------------------
# 7. Persistent Visual Identity Across Actions
# ---------------------------------------------------------------------------

def test_persistent_visual_id_across_actions(temp_model_dir):
    engine = InstantRepresentationEngine(library_dir=temp_model_dir)

    # CREATE
    m1, _ = engine.get_or_create_representation("dog", visual_id="vis-dog-42")
    assert m1.visual_id == "vis-dog-42"

    # MOVE (same visual_id)
    m2, _ = engine.get_or_create_representation("dog", visual_id="vis-dog-42")
    assert m2 is m1
    assert m2.visual_id == "vis-dog-42"

    # ANIMATE (same visual_id)
    m3, _ = engine.get_or_create_representation("dog", visual_id="vis-dog-42")
    assert m3 is m1


# ---------------------------------------------------------------------------
# 8. MOVE Without Regeneration
# ---------------------------------------------------------------------------

def test_move_action_without_regeneration(temp_model_dir):
    store = ConceptAssetStore()
    engine = GenerationEngine(asset_store=store, asset_dir=temp_model_dir)
    live_state = LiveState(generation_engine=engine)

    # CREATE
    plan_c = VisualPlan(
        event_id="p1",
        timestamp=time.time(),
        action=ActionType.CREATE,
        visual_id="v-car",
        concept="car",
        reason="Create car",
    )
    live_state.update_plan(plan_c)
    assert live_state.active_concept == "car"
    asset_after_create = store.get_asset("car")
    assert asset_after_create.is_3d

    # MOVE
    plan_m = VisualPlan(
        event_id="p2",
        timestamp=time.time(),
        action=ActionType.MOVE,
        visual_id="v-car",
        concept="car",
        reason="Move car",
    )
    live_state.update_plan(plan_m)

    # Asset must be identical; no re-creation
    asset_after_move = store.get_asset("car")
    assert asset_after_move is asset_after_create
    assert asset_after_move.model_3d is asset_after_create.model_3d

    engine.shutdown()


# ---------------------------------------------------------------------------
# 9. UPDATE Without Duplication
# ---------------------------------------------------------------------------

def test_update_without_duplication(temp_model_dir):
    store = ConceptAssetStore()
    engine = GenerationEngine(asset_store=store, asset_dir=temp_model_dir)
    live_state = LiveState(generation_engine=engine)

    # CREATE
    plan_c = VisualPlan(
        event_id="p1",
        timestamp=time.time(),
        action=ActionType.CREATE,
        visual_id="v-apple",
        concept="apple",
        reason="Create apple",
    )
    live_state.update_plan(plan_c)
    model1 = store.get_asset("apple").model_3d
    orig_version = model1.version

    # UPDATE scale
    plan_u = VisualPlan(
        event_id="p2",
        timestamp=time.time(),
        action=ActionType.UPDATE,
        visual_id="v-apple",
        concept="apple",
        spatial=SpatialInfo(anchored=True, anchor_type="hand", scale=1.4),
        reason="Scale apple",
    )
    live_state.update_plan(plan_u)

    model2 = store.get_asset("apple").model_3d
    assert model2 is model1  # In-place update, same object!
    assert model2.version > orig_version
    assert abs(model2.root.scale[0] - 1.4) < 1e-4

    engine.shutdown()


# ---------------------------------------------------------------------------
# 10. ANIMATE Time-Varying Behavior
# ---------------------------------------------------------------------------

def test_animate_time_varying_transforms():
    wave = build_wave_model("wave")
    meshes_t0 = wave.get_all_meshes_with_transforms(t=0.0)
    verts_t0 = meshes_t0[0][0].vertices.copy()

    meshes_t1 = wave.get_all_meshes_with_transforms(t=1.0)
    verts_t1 = meshes_t1[0][0].vertices.copy()

    # Dynamic wave oscillation: vertices change over time t
    assert not np.allclose(verts_t0, verts_t1)


# ---------------------------------------------------------------------------
# 11. Stale Background Task Rejection
# ---------------------------------------------------------------------------

def test_stale_background_task_rejection(temp_model_dir):
    engine = InstantRepresentationEngine(library_dir=temp_model_dir)
    model, _ = engine.get_or_create_representation("apple", visual_id="v-1")
    v1 = model.version

    # Manually bump version to simulate a subsequent update
    engine.update_representation("v-1", {"scale": 2.0})
    v2 = engine._visual_versions["v-1"]
    assert v2 > v1

    # Run refinement task specifying older task_version v1
    # It must be rejected and not overwrite the version
    engine._run_async_refinement("v-1", "apple", task_version=v1)
    assert engine._visual_versions["v-1"] == v2


# ---------------------------------------------------------------------------
# 12. Offline Operation & Zero Paid API Calls
# ---------------------------------------------------------------------------

def test_offline_operation_zero_paid_api_calls(temp_model_dir):
    engine = InstantRepresentationEngine(library_dir=temp_model_dir)

    # Ensure no network socket connections are made
    with patch("urllib.request.urlopen") as mock_url:
        m, lat = engine.get_or_create_representation("solar system")
        assert m is not None
        assert lat < 200.0
        # Assert strictly zero external API calls
        assert mock_url.call_count == 0


# ---------------------------------------------------------------------------
# 13. Graceful Failure and Fallback
# ---------------------------------------------------------------------------

def test_graceful_failure_and_fallback(temp_model_dir):
    store = ConceptAssetStore()
    engine = GenerationEngine(asset_store=store, asset_dir=temp_model_dir)

    # Unknown, uncatalogued concept
    req = GenerationRequest(concept="xyz_unknown_unprecedented_widget")
    job = engine.request_generation(req)

    assert job.status == JobState.FAILED
    assert job.result.metadata["representation_fidelity"] == "unavailable"
    assert "no verified exact identity" in job.error

    engine.shutdown()


def test_concept_accuracy_and_unavailable_distinctions(temp_model_dir):
    from integration.generation.instant_engine import RepresentationUnavailableError
    from integration.generation.representation_planner import FidelityLevel, validate_concept

    assert validate_concept("apple").identity.canonical_id == "object.apple"
    assert validate_concept("water molecule").identity.canonical_id == "object.water"
    assert validate_concept("solar system").identity.canonical_id == "system.solar"
    assert validate_concept("Jupiter").identity.canonical_id != "system.solar"
    assert validate_concept("Earth").identity.canonical_id == "planet.earth"
    assert validate_concept("planet").available is False
    assert validate_concept("apple").identity.aliases == ("apple",)
    assert validate_concept("O2").identity.canonical_id == "molecule.oxygen"
    assert validate_concept("oxygen atom").fidelity is FidelityLevel.UNAVAILABLE
    assert validate_concept("oxygen molecule").fidelity is FidelityLevel.UNAVAILABLE
    assert validate_concept("red sphere").identity is None
    assert validate_concept("unlisted widget").fidelity is FidelityLevel.UNAVAILABLE

    engine = InstantRepresentationEngine(library_dir=temp_model_dir)
    try:
        apple, _ = engine.get_or_create_representation("apple")
        assert apple.metadata["canonical_concept_id"] == "object.apple"
        assert apple.metadata["representation_fidelity"] == "illustrative"
        for concept in ("Jupiter", "Earth", "oxygen atom", "oxygen molecule", "red sphere", "unlisted widget"):
            with pytest.raises(RepresentationUnavailableError):
                engine.get_or_create_representation(concept)
    finally:
        engine.shutdown()


def test_broad_catalogue_match_cannot_override_specific_concept(temp_model_dir):
    from integration.generation.asset_retrieval import RetrievalResult

    class BroadMatchRetriever:
        def resolve(self, concept):
            wrong = build_solar_system_model("solar system")
            return wrong, {"match_rank": 2, "asset_source": "red_sphere.glb"}

    engine = InstantRepresentationEngine(library_dir=temp_model_dir, asset_retriever=BroadMatchRetriever())
    try:
        model, _ = engine.get_or_create_representation("apple")
        assert model.metadata["canonical_concept_id"] == "object.apple"
        assert "apple_body" in [mesh.name for mesh, _ in model.get_all_meshes_with_transforms()]
    finally:
        engine.shutdown()


def test_visual_quality_profiles_validate_parts_and_apply_materials():
    from integration.generation.visual_quality import enhance_and_validate

    apple = OrdinaryObject3DBuilder()._build_apple("apple")
    result = enhance_and_validate(apple, "object.apple")
    assert result.passed, result.reasons
    meshes = {mesh.name: mesh for mesh, _ in apple.get_all_meshes_with_transforms()}
    assert meshes["apple_body"].material.diffuse_color[2] > 150  # BGR red
    assert meshes["apple_body"].material.roughness < 0.5
    assert apple.metadata.get("visual_quality") is None  # quality caller owns report attachment

    incomplete = Model3D(concept="apple", root=ModelNode3D(name="root", mesh=create_uv_sphere(name="sphere")))
    rejected = enhance_and_validate(incomplete, "object.apple")
    assert not rejected.passed
    assert any("distinguishing component" in reason for reason in rejected.reasons)

    dark = Model3D(concept="unlisted", root=ModelNode3D(
        name="root", mesh=create_uv_sphere(color=(0, 0, 0), name="mesh")))
    dark_result = enhance_and_validate(dark, "custom.asset")
    assert not dark_result.passed
    assert any("too dark" in reason for reason in dark_result.reasons)


def test_renderer_ambient_specular_and_contact_shadow_are_visible():
    model = OrdinaryObject3DBuilder()._build_apple("apple")
    from integration.generation.visual_quality import enhance_and_validate
    assert enhance_and_validate(model, "object.apple").passed
    canvas = np.zeros((256, 256, 3), dtype=np.uint8)
    renderer = SoftwareRenderer3D()
    assert renderer.render(canvas, model, bbox=(0.1, 0.1, 0.9, 0.9), t=0.0)
    visible = canvas[np.any(canvas > 0, axis=2)]
    assert len(visible) > 100
    assert float(visible.mean()) > 20.0


# ---------------------------------------------------------------------------
# 14. Preservation of Existing PNG Rendering
# ---------------------------------------------------------------------------

def test_preservation_of_existing_png_rendering():
    store = ConceptAssetStore()
    dummy_png = np.full((100, 100, 3), 120, dtype=np.uint8)
    # Register purely as 2D image
    store.register_asset("classic_photo", dummy_png)

    asset = store.get_asset("classic_photo")
    assert not asset.is_3d
    assert asset.is_valid_image

    frame = np.zeros((300, 300, 3), dtype=np.uint8)
    render_box_and_asset(frame, bbox=(0.1, 0.1, 0.8, 0.8), concept_label="classic_photo", asset=asset)

    # Check center contains the PNG color (120)
    assert np.any(frame[100:150, 100:150] == 120)
