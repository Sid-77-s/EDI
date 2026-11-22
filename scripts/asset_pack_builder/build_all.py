"""
scripts/asset_pack_builder/build_all.py — Orchestrates Full 3D Asset Library Generation (220 Models).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import time
from typing import Dict, List, Tuple

from integration.generation.asset_catalogue import build_catalogue
from scripts.asset_pack_builder.glb_writer import (
    model_to_glb,
    validate_glb_asset,
    write_asset_sidecar,
)

# Import all 11 category generators
from scripts.asset_pack_builder.categories import (
    animals,
    astronomy,
    biology,
    chemistry,
    engineering,
    everyday,
    geography,
    history,
    physics,
    sports,
    vehicles,
)

CATEGORIES = [
    ("astronomy", "Astronomy and Space", astronomy.get_models),
    ("chemistry", "Chemistry and the Periodic Table", chemistry.get_models),
    ("physics", "Physics and Mathematics", physics.get_models),
    ("biology", "Biology and Human Anatomy", biology.get_models),
    ("animals", "Animals and Wildlife", animals.get_models),
    ("sports", "Sports Equipment and Balls", sports.get_models),
    ("vehicles", "Vehicles and Transportation", vehicles.get_models),
    ("engineering", "Engineering, Machines and Technology", engineering.get_models),
    ("everyday", "Everyday Objects and Furniture", everyday.get_models),
    ("geography", "Geography, Earth Science and Environment", geography.get_models),
    ("history", "History, Architecture and Culture", history.get_models),
]


def slugify(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip()).strip("_").lower()
    return cleaned


def build_library(
    starter_pack_dir: Path = Path("assets/starter_pack"),
    assets_dir: Path = Path("assets/models"),
) -> Dict:
    start_time = time.time()
    models_starter_dir = starter_pack_dir / "models"
    models_starter_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    catalog_records = []
    category_summary = {}
    total_triangles = 0
    total_vertices = 0
    total_bytes = 0

    print("=================================================================")
    print(" Magic Mirror — Building Complete 3D Asset Library (220 Models)")
    print("=================================================================")

    for cat_slug, cat_title, generator_fn in CATEGORIES:
        cat_models = generator_fn()
        cat_start_dir = models_starter_dir / cat_slug
        cat_start_dir.mkdir(parents=True, exist_ok=True)
        cat_runtime_dir = assets_dir / cat_slug
        cat_runtime_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nBuilding Category: {cat_title} ({len(cat_models)} models) -> {cat_slug}/")
        category_summary[cat_title] = {
            "slug": cat_slug,
            "model_count": len(cat_models),
            "models": [],
        }

        for model, meta in cat_models:
            name = meta["name"]
            file_stem = slugify(name)
            meta["uid"] = f"mm_{cat_slug}_{file_stem}"
            meta["fileIdentifier"] = f"{file_stem}.glb"
            meta["source"] = "Magic Mirror Curated Library"

            sp_glb_path = cat_start_dir / f"{file_stem}.glb"
            file_size = model_to_glb(model, sp_glb_path)
            val_res = validate_glb_asset(sp_glb_path)
            if not val_res.get("valid", False):
                raise RuntimeError(f"GLB validation failed for {name}: {val_res.get('error')}")

            # Collect geometry statistics from model
            total_model_verts = 0
            total_model_faces = 0
            def count_stats(node):
                nonlocal total_model_verts, total_model_faces
                if node.mesh is not None:
                    total_model_verts += len(node.mesh.vertices)
                    total_model_faces += len(node.mesh.faces)
                for child in node.children:
                    count_stats(child)
            count_stats(model.root)

            total_triangles += total_model_faces
            total_vertices += total_model_verts
            total_bytes += file_size

            meta["statistics"] = {
                "triangle_count": total_model_faces,
                "vertex_count": total_model_verts,
                "file_size_bytes": file_size,
            }

            write_asset_sidecar(sp_glb_path, meta)

            # Mirror to runtime assets/models
            rt_glb_path = cat_runtime_dir / f"{file_stem}.glb"
            shutil.copy2(sp_glb_path, rt_glb_path)
            shutil.copy2(sp_glb_path.with_suffix(".json"), rt_glb_path.with_suffix(".json"))

            record = {
                "uid": meta["uid"],
                "name": name,
                "category": cat_title,
                "subcategory": meta.get("subcategory", "general"),
                "file": f"{cat_slug}/{file_stem}.glb",
                "aliases": meta.get("aliases", []),
                "triangles": total_model_faces,
                "vertices": total_model_verts,
                "size_bytes": file_size,
                "license": meta.get("license", "CC0-1.0"),
                "distinguishing_features": meta.get("distinguishing_features", []),
                "description": meta.get("description", ""),
            }
            catalog_records.append(record)
            category_summary[cat_title]["models"].append(record)
            print(f"  + {name:<30} ({total_model_faces:>5} tris, {file_size:>6} B) -> {file_stem}.glb")

    elapsed = time.time() - start_time
    total_models = len(catalog_records)

    # 1. Generate assets/starter_pack/manifest.json
    manifest = {
        "manifest_version": "2.0.0",
        "library_name": "Magic Mirror Multimodal Spatial 3D Asset Library",
        "description": "Comprehensive local-first, low-latency, educationally accurate 3D model library for Magic Mirror.",
        "build_timestamp": "2026-11-22T17:00:00Z",
        "license": "CC0-1.0 Universal",
        "author": "Magic Mirror Spatial AI Team",
        "total_models": total_models,
        "total_categories": len(CATEGORIES),
        "total_triangles": total_triangles,
        "total_vertices": total_vertices,
        "total_bytes": total_bytes,
        "build_duration_seconds": round(elapsed, 2),
        "categories_summary": {
            cat_name: {
                "slug": data["slug"],
                "count": data["model_count"],
            }
            for cat_name, data in category_summary.items()
        },
        "models": catalog_records,
    }

    manifest_path = starter_pack_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] Wrote manifest to {manifest_path}")

    # Also mirror manifest to assets/manifest.json
    Path("assets/manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    # 2. Generate assets/starter_pack/README.md
    readme_lines = [
        "# Magic Mirror — Expanded 3D Asset Library",
        "",
        "## Overview",
        "",
        "This library provides an expanded, fully validated, local-first 3D asset collection for **Magic Mirror**,",
        "a real-time multimodal spatial visualization system. Every asset satisfies the strict core principles of",
        "identity correctness, visual recognizability, educational distinguishability, standard GLB v2 compatibility,",
        "zero-paid-API local execution, and permissive open licensing (**CC0 1.0 Universal**).",
        "",
        f"- **Total 3D Assets:** {total_models} models",
        f"- **Categories:** {len(CATEGORIES)} distinct domains",
        f"- **Total Polygons:** {total_triangles:,} triangles (optimized for ultra-low latency & >= 30 FPS rendering)",
        f"- **Total Library Size:** {total_bytes / (1024 * 1024):.2f} MB",
        "- **License:** CC0-1.0 Universal (Public Domain Dedication)",
        "",
        "## Category Summary",
        "",
        "| Category | Slug | Model Count | Key Educational Distinctions |",
        "|---|---|---|---|",
    ]

    for cat_title, data in category_summary.items():
        slug = data["slug"]
        cnt = data["model_count"]
        readme_lines.append(f"| **{cat_title}** | `{slug}/` | {cnt} | Strict physical/biological/mechanical differentiation |")

    readme_lines.extend([
        "",
        "## Technical Architecture & Importer Compatibility",
        "",
        "1. **Format:** Binary glTF 2.0 (`.glb`) compliant with Khronos Group specifications and Magic Mirror's native `load_glb()` parser (`integration/generation/model3d.py`).",
        "2. **Chunk Layout:** 12-byte header (`0x46546C67`, version 2), structured JSON chunk (`0x4E4F534A`), binary buffer chunk (`0x004E4942`), and 4-byte alignment padding.",
        "3. **Zero External Dependencies:** Every asset renders entirely locally in offline mode without requiring network connectivity, external texture URLs, or paid APIs.",
        "4. **Sidecar Metadata:** Every `<model>.glb` is accompanied by an authoritative `<model>.json` sidecar detailing canonical name, search aliases, category, subcategory, distinguishing features, educational purpose, and CC0 attribution.",
        "",
        "## Model Catalogue Index",
        "",
        "| Name | Category | File | Triangles | Aliases | Distinguishing Features |",
        "|---|---|---|---|---|---|",
    ])

    for rec in catalog_records:
        aliases_str = ", ".join(rec["aliases"][:3])
        features_str = "; ".join(rec["distinguishing_features"][:2])
        readme_lines.append(
            f"| {rec['name']} | {rec['category']} | `{rec['file']}` | {rec['triangles']} | {aliases_str} | {features_str} |"
        )

    readme_lines.extend([
        "",
        "## Attribution & Licensing",
        "",
        "All assets in this starter pack are released under **Creative Commons Zero 1.0 Universal (CC0-1.0)**.",
        "They are free for educational, commercial, and research use without restrictions.",
        "",
        "**Author / Curator:** Magic Mirror Spatial AI Team  ",
        "**Release Date:** 2026-11-22  ",
        "**Repository:** [Magic Mirror NLP-Visualisation](https://github.com/Yash-P29/Magic_Mirror_NLP-Visualisation)  ",
    ])

    readme_path = starter_pack_dir / "README.md"
    readme_path.write_text("\n".join(readme_lines), encoding="utf-8")
    print(f"[OK] Wrote documentation to {readme_path}")

    # 3. Write UID-keyed catalogue.json for SQLite catalogue indexing
    uid_catalog = {
        rec["uid"]: {
            "name": rec["name"],
            "category": rec["category"],
            "subcategory": rec["subcategory"],
            "description": rec["description"],
            "tags": [rec["category"], rec["subcategory"], *rec["aliases"]],
            "license": rec["license"],
            "author": "Magic Mirror Spatial AI Team",
            "fileIdentifier": rec["file"],
            "source": "Magic Mirror Curated Library",
            "fileType": "glb",
        }
        for rec in catalog_records
    }
    catalog_json_path = starter_pack_dir / "catalogue.json"
    catalog_json_path.write_text(json.dumps(uid_catalog, indent=2, ensure_ascii=False), encoding="utf-8")
    Path("assets/catalogue.json").write_text(json.dumps(uid_catalog, indent=2, ensure_ascii=False), encoding="utf-8")

    # Build SQLite metadata catalogue using asset_catalogue.build_catalogue
    db_path = starter_pack_dir / "catalogue.db"
    if db_path.exists():
        db_path.unlink()
    indexed_count = build_catalogue(catalog_json_path, db_path)
    print(f"[OK] Built SQLite metadata catalogue at {db_path} ({indexed_count} indexed entries)")

    # Also build runtime database
    rt_db_path = Path("assets/catalogue.db")
    if rt_db_path.exists():
        rt_db_path.unlink()
    build_catalogue(catalog_json_path, rt_db_path)
    print(f"[OK] Built runtime SQLite catalogue at {rt_db_path}")

    print("\n=================================================================")
    print(f" Build Complete: {total_models} models generated and verified in {elapsed:.2f}s!")
    print("=================================================================")
    return manifest


if __name__ == "__main__":
    build_library()
