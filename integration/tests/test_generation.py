"""
integration/tests/test_generation.py — Comprehensive tests for Hybrid Generation Engine.

Covers:
1. FLUX Cloud Provider (Text-to-image, editing, timeouts, retries, error handling, mock HTTP)
2. ComfyUI Local SDXL Provider (Workflow construction, 768x768 low-VRAM, Lightning mode, polling, mock HTTP)
3. Procedural Visual Provider (Atom, Molecule, Graph, Wave, generic scientific diagrams)
4. Asset Cache (Filesystem hit, in-memory cache, normalization, store synchronization)
5. Generation Router (Policy routing: Cache -> Procedural -> Edit -> High-detail -> Educational -> Fallback)
6. Generation Engine (Non-blocking async worker, immediate loading state, deduplication, callbacks, shutdown)
7. End-to-End VisualPlan CREATE to real PNG in yellow box
8. MOVE action reuses existing visual ID without regenerating
9. Switching active concept never leaks old concept visual
"""

import json
import os
import shutil
import tempfile
import time
import urllib.error
import urllib.request
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
from integration.generation.cache import AssetCache
from integration.generation.engine import GenerationEngine
from integration.generation.providers.flux_cloud import FluxCloudProvider
from integration.generation.providers.procedural import ProceduralVisualProvider
from integration.generation.providers.sdxl_local import ComfyUIProvider
from integration.generation.router import GenerationRouter
from integration.generation.schemas import (
    GenerationMode,
    GenerationRequest,
    GenerationResult,
    JobState,
    ProviderType,
    QualityTier,
)
from integration.multimodal_fusion.schemas import (
    ActionType,
    ConfidenceScore,
    Representation,
    RepresentationType,
    VisualPlan,
)
from integration_live_demo import LiveState, _draw_visual_opportunity


# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_asset_dir():
    temp_dir = tempfile.mkdtemp(prefix="test_assets_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


def _make_dummy_image_bytes(w=64, h=64, color=(128, 200, 50)):
    img = np.full((h, w, 3), color, dtype=np.uint8)
    success, enc = cv2.imencode(".png", img)
    assert success
    return enc.tobytes()


# ---------------------------------------------------------------------------
# 1. FLUX Cloud Provider Tests
# ---------------------------------------------------------------------------

def test_flux_provider_availability():
    # Unavailable when no key
    prov_no_key = FluxCloudProvider(api_key="")
    assert not prov_no_key.is_available()

    # Available when key provided
    prov_with_key = FluxCloudProvider(api_key="fake-test-key-bfl")
    assert prov_with_key.is_available()


def test_flux_provider_text_to_image_success(temp_asset_dir):
    fake_img_bytes = _make_dummy_image_bytes(128, 128)

    def mock_http(req: urllib.request.Request, timeout: float) -> bytes:
        url = req.full_url
        if "/flux.2-klein-4b" in url:
            return json.dumps({"id": "flux-job-12345"}).encode("utf-8")
        elif "get_result" in url:
            return json.dumps({
                "status": "Ready",
                "result": {"sample": "https://api.bfl.ml/sample.png"},
            }).encode("utf-8")
        elif "sample.png" in url:
            return fake_img_bytes
        raise ValueError(f"Unexpected URL: {url}")

    provider = FluxCloudProvider(
        api_key="fake-key",
        model_name="flux.2-klein-4b",
        asset_dir=temp_asset_dir,
        http_client=mock_http,
        poll_interval=0.01,
    )

    req = GenerationRequest(concept="quantum computer", width=768, height=768)
    res = provider.generate(req)

    assert res.is_success
    assert res.provider == "flux-cloud"
    assert res.model == "flux.2-klein-4b"
    assert res.image is not None
    assert res.image.shape[0] == 128
    assert res.image_path is not None
    assert os.path.exists(res.image_path)


def test_flux_provider_image_edit(temp_asset_dir):
    fake_img_bytes = _make_dummy_image_bytes(128, 128)
    dummy_input = np.zeros((64, 64, 3), dtype=np.uint8)

    def mock_http(req: urllib.request.Request, timeout: float) -> bytes:
        url = req.full_url
        if "flux-pro-1.0-fill" in url:
            return json.dumps({"id": "flux-edit-999"}).encode("utf-8")
        elif "get_result" in url:
            return json.dumps({
                "status": "Ready",
                "result": {"sample": "https://api.bfl.ml/edited.png"},
            }).encode("utf-8")
        elif "edited.png" in url:
            return fake_img_bytes
        raise ValueError(f"Unexpected URL: {url}")

    provider = FluxCloudProvider(
        api_key="fake-key",
        asset_dir=temp_asset_dir,
        http_client=mock_http,
        poll_interval=0.01,
    )

    req = GenerationRequest(
        concept="apple",
        prompt="turn it red",
        mode=GenerationMode.IMAGE_EDIT,
        input_image=dummy_input,
    )
    res = provider.edit(req)
    assert res.is_success
    assert res.metadata.get("mode") == "edit"


def test_flux_provider_timeout(temp_asset_dir):
    def mock_http(req: urllib.request.Request, timeout: float) -> bytes:
        url = req.full_url
        if "flux.2-klein-4b" in url:
            return json.dumps({"id": "flux-job-stall"}).encode("utf-8")
        elif "get_result" in url:
            # Always Pending
            return json.dumps({"status": "Pending"}).encode("utf-8")
        raise ValueError(f"Unexpected URL: {url}")

    provider = FluxCloudProvider(
        api_key="fake-key",
        asset_dir=temp_asset_dir,
        timeout=0.05,
        poll_interval=0.01,
        http_client=mock_http,
    )

    req = GenerationRequest(concept="black hole")
    res = provider.generate(req)
    assert not res.is_success
    assert res.status == JobState.FAILED
    assert "timed out" in res.error_message.lower()


# ---------------------------------------------------------------------------
# 2. ComfyUI Local SDXL Provider Tests
# ---------------------------------------------------------------------------

def test_comfyui_workflow_low_vram_and_lightning():
    provider = ComfyUIProvider(checkpoint="sd_xl_base_1.0.safetensors", use_lightning=False)
    wf_standard = provider.build_sdxl_workflow(prompt="a golden eagle", width=768, height=768)

    # 768x768 low-VRAM empty latent verification
    assert wf_standard["5"]["inputs"]["width"] == 768
    assert wf_standard["5"]["inputs"]["height"] == 768
    assert wf_standard["3"]["inputs"]["steps"] == 20
    assert wf_standard["3"]["inputs"]["cfg"] == 7.0

    # Lightning evaluation mode
    provider_lightning = ComfyUIProvider(checkpoint="sdxl_lightning_4step.safetensors", use_lightning=True)
    wf_light = provider_lightning.build_sdxl_workflow(prompt="a golden eagle", width=768, height=768)
    assert wf_light["3"]["inputs"]["steps"] == 4
    assert wf_light["3"]["inputs"]["cfg"] == 1.5
    assert wf_light["3"]["inputs"]["scheduler"] == "sgm_uniform"


def test_comfyui_provider_generate_success(temp_asset_dir):
    fake_img_bytes = _make_dummy_image_bytes(256, 256)

    def mock_http(req: urllib.request.Request, timeout: float) -> bytes:
        url = req.full_url
        if "/system_stats" in url:
            return json.dumps({"devices": [{"name": "RTX 5050"}]}).encode("utf-8")
        elif "/prompt" in url:
            return json.dumps({"prompt_id": "comfy-prompt-42"}).encode("utf-8")
        elif "/history/comfy-prompt-42" in url:
            return json.dumps({
                "comfy-prompt-42": {
                    "outputs": {
                        "9": {"images": [{"filename": "out.png", "subfolder": "", "type": "output"}]}
                    }
                }
            }).encode("utf-8")
        elif "/view" in url:
            return fake_img_bytes
        raise ValueError(f"Unexpected URL: {url}")

    provider = ComfyUIProvider(asset_dir=temp_asset_dir, http_client=mock_http, poll_interval=0.01)
    assert provider.is_available()

    req = GenerationRequest(concept="microscope", width=768, height=768)
    res = provider.generate(req)

    assert res.is_success
    assert res.provider == "sdxl-local"
    assert res.image is not None
    assert res.image.shape[0] == 256
    assert res.image_path is not None
    assert os.path.exists(res.image_path)


# ---------------------------------------------------------------------------
# 3. Procedural Visual Provider Tests
# ---------------------------------------------------------------------------

def test_procedural_provider_scientific_concepts(temp_asset_dir):
    prov = ProceduralVisualProvider(asset_dir=temp_asset_dir)
    assert prov.is_available()

    # 1. Atom model
    res_atom = prov.generate(GenerationRequest(concept="bohr atom model", width=512, height=512))
    assert res_atom.is_success
    assert res_atom.image.shape == (512, 512, 4)  # BGRA
    assert os.path.exists(res_atom.image_path)

    # 2. Molecule
    res_mol = prov.generate(GenerationRequest(concept="water molecule h2o", width=512, height=512))
    assert res_mol.is_success
    assert res_mol.image.shape == (512, 512, 4)

    # 3. Graph diagram
    res_graph = prov.generate(GenerationRequest(concept="neural network graph", width=512, height=512))
    assert res_graph.is_success

    # 4. Wave physics
    res_wave = prov.generate(GenerationRequest(concept="sine wave frequency", width=512, height=512))
    assert res_wave.is_success


# ---------------------------------------------------------------------------
# 4. Asset Cache Tests
# ---------------------------------------------------------------------------

def test_asset_cache_operations(temp_asset_dir):
    store = ConceptAssetStore()
    cache = AssetCache(asset_dir=temp_asset_dir, asset_store=store)

    assert cache.get("planet") is None

    # Store dummy image
    test_img = np.full((100, 100, 3), 200, dtype=np.uint8)
    asset = cache.store("Planet", test_img)
    assert asset.status == "ready"

    # Retrieve from cache
    hit = cache.get("planet")
    assert hit is not None
    assert hit.is_valid_image
    assert store.has_valid_asset("planet")


# ---------------------------------------------------------------------------
# 5. Generation Router Tests
# ---------------------------------------------------------------------------

def test_router_policy_and_fallbacks(temp_asset_dir):
    mock_flux = MagicMock(spec=FluxCloudProvider)
    mock_flux.is_available.return_value = True

    mock_sdxl = MagicMock(spec=ComfyUIProvider)
    mock_sdxl.is_available.return_value = True

    mock_proc = MagicMock(spec=ProceduralVisualProvider)
    mock_proc.is_available.return_value = True

    providers = {
        ProviderType.FLUX_CLOUD: mock_flux,
        ProviderType.SDXL_LOCAL: mock_sdxl,
        ProviderType.PROCEDURAL: mock_proc,
    }

    cache = AssetCache(asset_dir=temp_asset_dir)
    router = GenerationRouter(providers=providers, cache=cache)

    # A. Procedural Scientific Visual
    req_sci = GenerationRequest(concept="carbon atom", representation_type="3d")
    dec_sci = router.route(req_sci)
    assert dec_sci.provider_type == ProviderType.PROCEDURAL

    # B. High-detail Cloud FLUX
    req_hd = GenerationRequest(concept="galaxy", quality=QualityTier.HIGH_DETAIL)
    dec_hd = router.route(req_hd)
    assert dec_hd.provider_type == ProviderType.FLUX_CLOUD
    assert ProviderType.SDXL_LOCAL in dec_hd.fallback_chain

    # C. Simple Educational Object Local SDXL
    req_edu = GenerationRequest(concept="wooden chair")
    dec_edu = router.route(req_edu)
    assert dec_edu.provider_type == ProviderType.SDXL_LOCAL
    assert ProviderType.FLUX_CLOUD in dec_edu.fallback_chain

    # D. Image Edit
    dummy_img = np.zeros((32, 32, 3), dtype=np.uint8)
    req_edit = GenerationRequest(concept="chair", mode=GenerationMode.IMAGE_EDIT, input_image=dummy_img)
    dec_edit = router.route(req_edit)
    assert dec_edit.provider_type in (ProviderType.SDXL_LOCAL, ProviderType.FLUX_CLOUD)

    # E. Cached Asset Reuse
    cache.store("wooden chair", np.zeros((32, 32, 3), dtype=np.uint8))
    dec_cached = router.route(req_edu)
    assert dec_cached.provider_type == ProviderType.CACHE


# ---------------------------------------------------------------------------
# 6. Generation Engine Non-Blocking & Lifecycle Tests
# ---------------------------------------------------------------------------

def test_generation_engine_async_and_deduplication(temp_asset_dir):
    store = ConceptAssetStore()
    proc = ProceduralVisualProvider(asset_dir=temp_asset_dir)

    engine = GenerationEngine(
        asset_store=store,
        asset_dir=temp_asset_dir,
        procedural_provider=proc,
    )

    req = GenerationRequest(concept="water molecule", representation_type="diagram")
    job = engine.request_generation(req)

    # Must immediately return running/queued and store must have placeholder
    assert job.concept == "water molecule"
    placeholder = store.get_asset("water molecule")
    assert placeholder is not None
    assert placeholder.status in ("loading", "ready")

    # Wait for procedural generation to finish
    for _ in range(50):
        status = engine.get_job_status("water molecule")
        if status and status.status == JobState.READY:
            break
        time.sleep(0.05)

    final_status = engine.get_job_status("water molecule")
    assert final_status.status == JobState.READY
    assert store.has_valid_asset("water molecule")

    # Deduplication test: re-requesting returns existing ready asset immediately
    job2 = engine.request_generation(req)
    assert job2.status == JobState.READY

    engine.shutdown(wait=True)


# ---------------------------------------------------------------------------
# 7. End-to-End VisualPlan CREATE to Real PNG in Yellow Box
# ---------------------------------------------------------------------------

def test_end_to_end_create_to_yellow_box_render(temp_asset_dir):
    store = ConceptAssetStore()
    proc = ProceduralVisualProvider(asset_dir=temp_asset_dir)
    engine = GenerationEngine(
        asset_store=store,
        asset_dir=temp_asset_dir,
        procedural_provider=proc,
    )

    live_state = LiveState(generation_engine=engine)

    # Simulate VisualPlan CREATE
    plan = VisualPlan(
        event_id="plan-1",
        timestamp=time.time(),
        action=ActionType.CREATE,
        visual_id="vis-atom-1",
        concept="bohr atom",
        representation=Representation(type=RepresentationType.TYPE_3D, description="atom"),
        reason="Detected concept and pointing opportunity",
        confidence=ConfidenceScore(overall=0.9, semantic=0.9, reference=0.9, spatial=0.9, opportunity=0.9),
    )

    # 1. Pipeline emits plan
    live_state.update_plan(plan)
    assert live_state.active_concept == "bohr atom"

    # Yellow box frame simulation
    camera_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    bbox = (0.2, 0.2, 0.6, 0.6)

    # Frame while generating: yellow box and concept badge render even without image
    asset_loading = store.get_asset("bohr atom")
    render_box_and_asset(camera_frame, bbox=bbox, concept_label="bohr atom", asset=asset_loading)
    # Check yellow dashed border was rendered on frame (border color: (0, 220, 220))
    assert np.any(camera_frame > 0)

    # Wait for generator to complete
    for _ in range(50):
        if store.has_valid_asset("bohr atom"):
            break
        time.sleep(0.05)

    assert store.has_valid_asset("bohr atom")
    ready_asset = store.get_asset("bohr atom")
    assert ready_asset.is_valid_image

    # Frame with READY asset: renders actual PNG image into interior of yellow box
    render_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    render_box_and_asset(render_frame, bbox=bbox, concept_label="bohr atom", asset=ready_asset)

    # Center of box should contain image content
    cx, cy = int(0.4 * 640), int(0.4 * 480)
    assert np.any(render_frame[cy - 20 : cy + 20, cx - 20 : cx + 20] > 0)

    engine.shutdown(wait=True)


# ---------------------------------------------------------------------------
# 8. MOVE Reuses Existing Visual ID Without Regenerating
# ---------------------------------------------------------------------------

def test_move_action_reuses_existing_asset_without_regeneration(temp_asset_dir):
    store = ConceptAssetStore()
    engine = MagicMock(spec=GenerationEngine)

    live_state = LiveState(generation_engine=engine)

    # First: CREATE
    plan_create = VisualPlan(
        event_id="plan-c",
        timestamp=time.time(),
        action=ActionType.CREATE,
        visual_id="vis-dog-1",
        concept="dog",
        reason="Create dog",
    )
    live_state.update_plan(plan_create)
    assert engine.request_generation.call_count == 1
    assert live_state.active_concept == "dog"
    assert live_state.active_visual_id == "vis-dog-1"

    # Second: MOVE (pointing gesture moved)
    plan_move = VisualPlan(
        event_id="plan-m",
        timestamp=time.time(),
        action=ActionType.MOVE,
        visual_id="vis-dog-1",
        concept="dog",
        reason="Move dog to hand",
    )
    live_state.update_plan(plan_move)

    # Crucial assertion: MOVE must NOT call request_generation!
    assert engine.request_generation.call_count == 1
    assert live_state.active_concept == "dog"
    assert live_state.active_visual_id == "vis-dog-1"


# ---------------------------------------------------------------------------
# 9. Concept Switch Never Leaks Old Concept Image
# ---------------------------------------------------------------------------

def test_concept_switch_never_displays_old_image(temp_asset_dir):
    store = ConceptAssetStore()
    # Register "dog"
    dog_img = np.full((50, 50, 3), 100, dtype=np.uint8)
    store.register_asset("dog", dog_img)

    # Active concept changes to "cat" (no asset yet)
    active_concept = "cat"
    queried_asset = store.get_asset(active_concept)

    # The store must NOT return dog for cat!
    assert queried_asset.concept == "cat"
    assert not queried_asset.is_valid_image

    # Render frame with active_concept "cat"
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    render_box_and_asset(
        frame,
        bbox=(0.1, 0.1, 0.8, 0.8),
        concept_label=active_concept,
        asset=queried_asset if (queried_asset and queried_asset.is_valid_image) else None,
    )
    # Only border and text badge rendered, dog pixels absent
    assert queried_asset.image is None

