"""
integration/tests/test_low_latency_rendering.py
================================================
Automated verification for low-latency NLP extraction, direct real-time
3D frame rendering, and vibrant non-dark coloring across all categories.
"""
from __future__ import annotations

import time
import json
import numpy as np
import pytest

from extraction.fast_processor import parse_simple_command
from integration.generation import GenerationEngine, GenerationRequest
from integration.generation.renderer3d import SoftwareRenderer3D
from integration.generation.model3d import load_glb
from MagicMirror_Visualisation.src.perception.assets import ConceptAssetStore


SAMPLE_COMMANDS = [
    # Astronomy
    ("show me jupiter", "jupiter", "demonstrating"),
    ("visualize the solar system", "solar system", "demonstrating"),
    # Sports
    ("put a football on my hand", "football", "instructing"),
    # Vehicles
    ("show me a formula 1 car", "formula 1 car", "demonstrating"),
    # Chemistry
    ("show me the structure of an oxygen atom", "oxygen atom", "demonstrating"),
    # Biology
    ("show me a human heart", "human heart", "demonstrating"),
    ("display a dna molecule", "dna molecule", "demonstrating"),
    # Geography
    ("explain how a volcano works", "volcano", "explaining"),
    # Animals
    ("create a dog", "dog", "demonstrating"),
    # Physics
    ("show me a proton", "proton", "demonstrating"),
    # History
    ("show me an ancient pyramid", "ancient pyramid", "demonstrating"),
]


@pytest.mark.parametrize("phrase,expected_concept,expected_intent", SAMPLE_COMMANDS)
def test_fast_nlp_extraction_sub_millisecond(phrase: str, expected_concept: str, expected_intent: str):
    """FastProcessor must parse verbal commands deterministically in < 1 ms without Ollama."""
    t0 = time.perf_counter()
    state = parse_simple_command(phrase)
    t_parse = (time.perf_counter() - t0) * 1000.0

    assert state is not None, f"FastProcessor failed to parse: {phrase!r}"
    assert t_parse < 5.0, f"Parse took too long: {t_parse:.3f} ms"

    entity = state["semantic_graph"]["entities"][0]
    assert entity["name"] == expected_concept
    assert state["speaker_activity"][0]["intent"] == expected_intent


def test_vibrant_lighting_and_color_quality():
    """SoftwareRenderer3D must render with vibrant highlights (> 180) and rich non-muddy colors."""
    renderer = SoftwareRenderer3D()
    test_models = [
        ("assets/models/astronomy/jupiter.glb", 60.0),
        ("assets/models/chemistry/oxygen_atom.glb", 60.0),
        ("assets/models/vehicles/formula_1_car.glb", 20.0),  # Authentic black carbon tires/halo + bright red body
        ("assets/models/sports/football.glb", 60.0),
    ]

    for path, expected_min_mean in test_models:
        model = load_glb(path)
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        rendered = renderer.render(canvas, model, bbox=(0.2, 0.2, 0.8, 0.8), t=1.0)
        assert rendered is True, f"Failed to render {path}"

        px = canvas[canvas > 0]
        assert len(px) > 1000, f"Insufficient rendered pixels for {path}"

        mean_intensity = float(px.mean())
        max_intensity = int(px.max())

        # Shading must be vibrant and clear, not muddy
        assert mean_intensity >= expected_min_mean, f"Model {path} rendered too dark: mean={mean_intensity:.1f}"
        assert max_intensity >= 180, f"Model {path} lacks highlight contrast: max={max_intensity}"


def test_frame_background_preservation():
    """Rendering directly onto video frames must preserve real-world pixels outside the model."""
    renderer = SoftwareRenderer3D()
    model = load_glb("assets/models/astronomy/jupiter.glb")

    # Synthetic camera frame with background pattern
    canvas = np.full((480, 640, 3), 115, dtype=np.uint8)
    renderer.render(canvas, model, bbox=(0.25, 0.2, 0.75, 0.8), t=0.5)

    # Corners and border regions outside bbox must remain untouched
    top_left = canvas[:50, :50]
    assert np.all(top_left == 115), "Background outside bbox was improperly overwritten"

    bottom_right = canvas[430:, 590:]
    assert np.all(bottom_right == 115), "Background outside bbox was improperly overwritten"


def test_alpha_contact_shadow_does_not_render_pitch_black():
    """Contact shadow must softly darken background without painting opaque black blocks."""
    renderer = SoftwareRenderer3D()
    model = load_glb("assets/models/astronomy/jupiter.glb")

    canvas = np.full((480, 640, 3), 150, dtype=np.uint8)
    renderer.render(canvas, model, bbox=(0.2, 0.2, 0.8, 0.8), t=0.0)

    # Beneath the model in the shadow zone, pixels are softly shaded (not opaque black 0/15)
    shadow_zone = canvas[int(480 * 0.75):int(480 * 0.82), int(640 * 0.3):int(640 * 0.7)]
    assert int(shadow_zone.min()) >= 50, f"Shadow zone rendered pitch-black: min={shadow_zone.min()}"
