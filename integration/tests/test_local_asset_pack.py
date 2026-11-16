import json

import numpy as np

from integration.generation.asset_catalogue import LocalAssetPackResolver
from integration.generation.instant_engine import InstantRepresentationEngine
from integration.generation.renderer3d import SoftwareRenderer3D
from integration.generation.engine import GenerationEngine
from integration.generation.schemas import GenerationRequest
from MagicMirror_Visualisation.src.perception.assets import ConceptAssetStore


FIXTURE = "integration/tests/fixtures/glb_real_world/BoxTextured.glb"


def _asset(pack, name="Red Apple", aliases=None, license="CC0"):
    source = pack / "model.glb"
    source.write_bytes(open(FIXTURE, "rb").read())
    (pack / "model.json").write_text(json.dumps({
        "name": name,
        "aliases": aliases or [],
        "license": license,
        "source": "local curated fixture",
        "author": "test author",
        "attribution": "Test attribution",
    }), encoding="utf-8")
    return source


def test_local_asset_pack_exact_and_alias_match(tmp_path):
    pack = tmp_path / "pack"
    pack.mkdir()
    _asset(pack, aliases=["fruit apple"])
    resolver = LocalAssetPackResolver(pack)
    exact = resolver.resolve("Red Apple")
    alias = resolver.resolve("fruit apple")
    assert exact and exact[0].concept == "Red Apple"
    assert alias and alias[0].metadata["license"] == "CC0"
    assert alias[0].metadata["author"] == "test author"


def test_license_filter_and_unknown_license_policy(tmp_path):
    pack = tmp_path / "pack"
    pack.mkdir()
    _asset(pack, name="unknown asset", license=None)
    assert LocalAssetPackResolver(pack).resolve("unknown asset") is None
    assert LocalAssetPackResolver(pack, allow_unknown_license=True).resolve("unknown asset")
    (pack / "model.json").write_text(json.dumps({"name": "unknown asset", "license": "NC"}), encoding="utf-8")
    assert LocalAssetPackResolver(pack, allowed_licenses={"CC0"}).resolve("unknown asset") is None
    assert LocalAssetPackResolver(pack, allowed_licenses={"NC"}).resolve("unknown asset")


def test_legacy_filename_lookup_observes_license_policy(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    (models / "apple.glb").write_bytes(open(FIXTURE, "rb").read())
    engine = InstantRepresentationEngine(library_dir=models)
    permissive = InstantRepresentationEngine(library_dir=models, allow_unknown_asset_licenses=True)
    try:
        rejected, _ = engine.get_or_create_representation("apple", "rejected")
        accepted, _ = permissive.get_or_create_representation("apple", "accepted")
        assert "asset_source" not in rejected.metadata
        assert accepted.metadata["asset_source"].endswith("apple.glb")
    finally:
        engine.shutdown()
        permissive.shutdown()


def test_missing_and_corrupt_assets_fall_back_to_procedural(tmp_path):
    engine = InstantRepresentationEngine(library_dir=tmp_path / "models", asset_pack_dir=tmp_path / "pack")
    try:
        missing, _ = engine.get_or_create_representation("apple", "missing-id")
        assert missing.root.children
        (tmp_path / "pack").mkdir()
        (tmp_path / "pack" / "bad.glb").write_bytes(b"not a GLB")
        (tmp_path / "pack" / "bad.json").write_text(json.dumps({"name": "corrupt apple", "license": "CC0"}), encoding="utf-8")
        corrupt, _ = engine.get_or_create_representation("corrupt apple", "corrupt-id")
        assert corrupt.category == "3D_OBJECT"
        assert corrupt.root.children
    finally:
        engine.shutdown()


def test_loaded_asset_reaches_existing_renderer(tmp_path):
    pack = tmp_path / "pack"
    pack.mkdir()
    _asset(pack, name="fixture box")
    engine = InstantRepresentationEngine(library_dir=tmp_path / "models", asset_pack_dir=pack)
    try:
        model, _ = engine.get_or_create_representation("fixture box", "stable-visual")
        assert model.visual_id == "stable-visual"
        assert model.metadata["asset_source"].endswith("model.glb")
        canvas = np.zeros((256, 256, 3), dtype=np.uint8)
        assert SoftwareRenderer3D().render(canvas, model, bbox=(0.0, 0.0, 1.0, 1.0), t=0.0)
        assert np.count_nonzero(canvas) > 0
        moved = engine.update_representation("stable-visual", {"position": [1, 2, 3]})
        assert moved is model and model.visual_id == "stable-visual"
    finally:
        engine.shutdown()


def test_generation_engine_registers_resolved_model_in_asset_store(tmp_path):
    pack = tmp_path / "curated pack"
    pack.mkdir()
    _asset(pack, name="curated apple")
    store = ConceptAssetStore()
    engine = GenerationEngine(asset_store=store, asset_dir=tmp_path / "generated",
                              asset_pack_dir=pack)
    try:
        job = engine.request_generation(GenerationRequest(
            concept="curated apple", visual_id="pipeline-visual"))
        assert job.result.model_3d.visual_id == "pipeline-visual"
        registered = store.get_asset("curated apple")
        assert registered is not None and registered.is_3d
        assert registered.model_3d.metadata["license"] == "CC0"
    finally:
        engine.shutdown()
        engine.instant_engine.shutdown()
