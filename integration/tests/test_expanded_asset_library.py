"""
integration/tests/test_expanded_asset_library.py — Verification of the 220-Model Expanded Asset Library.
"""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import time
import numpy as np
import pytest

from integration.generation.asset_catalogue import LocalAssetPackResolver, search_catalogue
from integration.generation.model3d import Model3D, load_glb
from integration.generation.renderer3d import SoftwareRenderer3D
from scripts.asset_pack_builder.glb_writer import validate_glb_asset

STARTER_PACK_DIR = Path("assets/starter_pack")
MODELS_DIR = STARTER_PACK_DIR / "models"
MANIFEST_PATH = STARTER_PACK_DIR / "manifest.json"
README_PATH = STARTER_PACK_DIR / "README.md"
CATALOGUE_DB = STARTER_PACK_DIR / "catalogue.db"


def test_manifest_structure_and_counts():
    """Manifest must exist, be valid JSON, and detail 220 models across 11 categories."""
    assert MANIFEST_PATH.is_file(), f"Manifest missing at {MANIFEST_PATH}"
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert manifest["manifest_version"] == "2.0.0"
    assert manifest["total_models"] == 220
    assert manifest["total_categories"] == 11
    assert manifest["license"] == "CC0-1.0 Universal"

    categories = manifest["categories_summary"]
    assert len(categories) == 11

    # Check category distribution
    expected_counts = {
        "Astronomy and Space": 25,
        "Chemistry and the Periodic Table": 30,
        "Physics and Mathematics": 20,
        "Biology and Human Anatomy": 25,
        "Animals and Wildlife": 20,
        "Sports Equipment and Balls": 15,
        "Vehicles and Transportation": 20,
        "Engineering, Machines and Technology": 15,
        "Everyday Objects and Furniture": 25,
        "Geography, Earth Science and Environment": 15,
        "History, Architecture and Culture": 10,
    }

    for cat_name, exp_cnt in expected_counts.items():
        assert cat_name in categories, f"Missing category: {cat_name}"
        assert categories[cat_name]["count"] == exp_cnt, (
            f"Expected {exp_cnt} for {cat_name}, got {categories[cat_name]['count']}"
        )

    assert len(manifest["models"]) == 220


def test_all_files_exist_and_sidecars_valid():
    """All 220 GLB files and sidecars must exist, have valid JSON, and non-empty metadata."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    for record in manifest["models"]:
        rel_file = record["file"]
        glb_path = MODELS_DIR / rel_file
        assert glb_path.is_file(), f"GLB file missing: {glb_path}"

        # Check sidecar
        sidecar_path = glb_path.with_suffix(".json")
        assert sidecar_path.is_file(), f"Sidecar missing: {sidecar_path}"
        meta = json.loads(sidecar_path.read_text(encoding="utf-8"))

        assert meta["name"] == record["name"]
        assert meta["license"] == "CC0-1.0"
        assert len(meta.get("distinguishing_features", [])) >= 2
        assert len(meta.get("educational_purpose", "")) > 10

        # Validate GLB binary format
        val = validate_glb_asset(glb_path)
        assert val["valid"] is True, f"Invalid GLB {glb_path}: {val.get('error')}"


def test_sqlite_catalogue_search():
    """SQLite catalogue must index all 220 models with CC0-1.0 license."""
    assert CATALOGUE_DB.is_file(), f"Catalogue DB missing at {CATALOGUE_DB}"
    with sqlite3.connect(str(CATALOGUE_DB)) as db:
        cnt = db.execute("SELECT COUNT(*) FROM assets").fetchone()[0]
        assert cnt == 220, f"Expected 220 entries in catalogue, found {cnt}"

    # Search for specific queries
    results_jupiter = search_catalogue(CATALOGUE_DB, query="jupiter")
    assert len(results_jupiter) >= 1
    assert any("jupiter" in r["name"].lower() for r in results_jupiter)

    results_dna = search_catalogue(CATALOGUE_DB, query="dna")
    assert len(results_dna) >= 1


def test_local_asset_resolver_resolution():
    """LocalAssetPackResolver must resolve canonical names and aliases cleanly."""
    resolver = LocalAssetPackResolver(MODELS_DIR, allowed_licenses=["cc0-1.0", "cc0"])

    test_queries = [
        # Astronomy
        ("Jupiter", "Jupiter"),
        ("Gas giant Jupiter", "Jupiter"),
        ("Mars", "Mars"),
        ("Saturn", "Saturn"),
        # Chemistry
        ("Hydrogen Atom", "Hydrogen Atom"),
        ("Oxygen Atom", "Oxygen Atom"),
        ("Oxygen Molecule", "Oxygen Molecule"),
        ("Water Molecule", "Water Molecule"),
        ("H2O", "Water Molecule"),
        ("DNA", "DNA Double Helix"),
        # Physics
        ("Proton", "Proton"),
        ("Bar Magnet", "Bar Magnet"),
        ("Pendulum", "Pendulum"),
        # Biology
        ("Human Heart", "Human Heart"),
        ("Human Brain", "Human Brain"),
        ("Neuron", "Neuron"),
        # Sports
        ("Football", "Football"),
        ("Basketball", "Basketball"),
        ("Tennis Ball", "Tennis Ball"),
        # Vehicles
        ("Formula 1 Car", "Formula 1 Car"),
        ("f1 car", "Formula 1 Car"),
        ("Airplane", "Airplane"),
        # Everyday & History
        ("Coffee Mug", "Coffee Mug"),
        ("Pyramid", "Pyramid"),
        ("Globe", "Globe"),
    ]

    for query, expected_canonical in test_queries:
        res = resolver.resolve(query)
        assert res is not None, f"Failed to resolve query: '{query}'"
        model, meta = res
        assert meta["name"] == expected_canonical, (
            f"Query '{query}' resolved to '{meta['name']}' instead of '{expected_canonical}'"
        )
        assert model.num_triangles > 0


def test_concept_distinction_no_misleading_substitutions():
    """Critical scientific concepts must NEVER be substituted for one another."""
    resolver = LocalAssetPackResolver(MODELS_DIR, allowed_licenses=["cc0-1.0"])

    # 1. Oxygen Atom vs Oxygen Molecule
    res_atom = resolver.resolve("Oxygen Atom")
    res_mol = resolver.resolve("Oxygen Molecule")
    assert res_atom is not None and res_mol is not None
    assert res_atom[1]["name"] == "Oxygen Atom"
    assert res_mol[1]["name"] == "Oxygen Molecule"
    assert res_atom[1]["name"] != res_mol[1]["name"]

    # 2. Carbon Atom vs Carbon Dioxide
    res_c = resolver.resolve("Carbon Atom")
    res_co2 = resolver.resolve("Carbon Dioxide")
    assert res_c is not None and res_co2 is not None
    assert res_c[1]["name"] == "Carbon Atom"
    assert res_co2[1]["name"] == "Carbon Dioxide"

    # 3. Football vs Basketball
    res_fb = resolver.resolve("Football")
    res_bb = resolver.resolve("Basketball")
    assert res_fb is not None and res_bb is not None
    assert res_fb[1]["name"] != res_bb[1]["name"]


def test_load_latency_and_rendering_performance():
    """Resolving and loading models must take < 50 ms and render with positive pixel coverage."""
    resolver = LocalAssetPackResolver(MODELS_DIR, allowed_licenses=["cc0-1.0"])
    renderer = SoftwareRenderer3D()

    sample_concepts = [
        "Mercury", "Jupiter", "Oxygen Atom", "Water Molecule", "Proton",
        "Human Heart", "Dog", "Football", "Formula 1 Car", "Robotic Arm",
        "Coffee Mug", "Globe", "Pyramid"
    ]

    for concept in sample_concepts:
        t0 = time.perf_counter()
        res = resolver.resolve(concept)
        t_resolve = (time.perf_counter() - t0) * 1000.0

        assert res is not None, f"Failed to resolve: {concept}"
        model, _ = res
        assert t_resolve < 50.0, f"Resolution took too long ({t_resolve:.2f} ms) for {concept}"

        # Render model onto test frame
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        rendered_ok = renderer.render(frame, model, bbox=(0.1, 0.1, 0.9, 0.9))
        assert rendered_ok is True, f"Renderer returned False for {concept}"

        # Verify pixels were drawn
        colored_pixels = np.count_nonzero(frame)
        assert colored_pixels > 0, f"Renderer produced empty frame for {concept}"
