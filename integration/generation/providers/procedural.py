"""
integration/generation/providers/procedural.py — Procedural Visual Generation for Scientific & Structural Concepts.

Generates high-contrast, publication-quality procedural diagrams and scientific visualizations:
- Atomic models (nucleus with proton/neutron cluster, electron orbits with energy levels)
- Molecular structures (bonded atoms with element symbols and 3D sphere shading)
- Graph & Network diagrams (connected nodes, directional vectors)
- Physics & Wave patterns (coordinate axes, wave functions, vector arrows)
- Generic structural concepts
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Optional, Tuple, Union
import cv2
import numpy as np

from integration.generation.providers.base import BaseImageProvider
from integration.generation.schemas import (
    GenerationRequest,
    GenerationResult,
    JobState,
)

logger = logging.getLogger(__name__)


class ProceduralVisualProvider(BaseImageProvider):
    """Generates procedural visuals for scientific concepts without heavy neural inference."""

    def __init__(self, asset_dir: Union[str, Path] = "assets") -> None:
        super().__init__(name="procedural")
        self.asset_dir = Path(asset_dir)

    def is_available(self) -> bool:
        """Always available locally with zero external dependencies."""
        return True

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generates procedural scientific image tailored to concept."""
        start_time = time.monotonic()
        concept = request.concept.strip().lower()
        w = request.width or 768
        h = request.height or 768

        # 4-channel BGRA canvas with transparent background
        canvas = np.zeros((h, w, 4), dtype=np.uint8)

        # Dispatch based on scientific concept type
        if any(term in concept for term in ("atom", "electron", "nucleus", "bohr", "quantum", "proton")):
            self._render_atom(canvas, request.concept)
        elif any(term in concept for term in ("molecule", "water", "h2o", "co2", "methane", "bond", "dna", "chemical")):
            self._render_molecule(canvas, request.concept)
        elif any(term in concept for term in ("graph", "network", "node", "tree", "flowchart")):
            self._render_graph(canvas, request.concept)
        elif any(term in concept for term in ("wave", "frequency", "oscillation", "vector", "force", "field", "sinusoid")):
            self._render_wave(canvas, request.concept)
        else:
            self._render_generic_concept(canvas, request.concept)

        file_path = self.save_image_to_disk(canvas, request.concept, self.asset_dir, prefix="proc_")
        latency = (time.monotonic() - start_time) * 1000.0

        return GenerationResult(
            concept=request.concept,
            visual_id=request.visual_id,
            image=canvas,
            image_path=file_path,
            status=JobState.READY,
            provider=self.name,
            model="procedural-2d-canvas",
            latency_ms=latency,
            metadata={"render_type": "procedural", "resolution": f"{w}x{h}"},
        )

    def edit(self, request: GenerationRequest) -> GenerationResult:
        """Procedural edit: redraws with updated parameters/labels."""
        return self.generate(request)

    def _render_atom(self, canvas: np.ndarray, label: str) -> None:
        """Draws Bohr atomic model with nucleus and tilted orbital rings."""
        h, w = canvas.shape[:2]
        cx, cy = w // 2, h // 2

        # 1. Outer electron orbits (ellipses with cyan/teal tint)
        orbit_color = (255, 220, 0, 180)  # BGRA
        axes_list = [
            (int(w * 0.38), int(h * 0.14), 0),
            (int(w * 0.38), int(h * 0.14), 60),
            (int(w * 0.38), int(h * 0.14), 120),
        ]
        for ax, ay, angle in axes_list:
            cv2.ellipse(canvas, (cx, cy), (ax, ay), angle, 0, 360, orbit_color, 2, cv2.LINE_AA)

        # 2. Orbiting electrons
        electron_color = (255, 255, 100, 255)
        for i, (ax, ay, angle) in enumerate(axes_list):
            rad = math.radians(angle)
            t = math.radians(45 + i * 110)
            # Parametric ellipse rotation
            ex = cx + int(ax * math.cos(t) * math.cos(rad) - ay * math.sin(t) * math.sin(rad))
            ey = cy + int(ax * math.cos(t) * math.sin(rad) + ay * math.sin(t) * math.cos(rad))
            # Glowing dot
            cv2.circle(canvas, (ex, ey), 8, (255, 200, 0, 100), -1, cv2.LINE_AA)
            cv2.circle(canvas, (ex, ey), 5, electron_color, -1, cv2.LINE_AA)

        # 3. Dense central nucleus (protons/neutrons cluster)
        nucleus_radius = int(min(w, h) * 0.08)
        cv2.circle(canvas, (cx, cy), nucleus_radius + 4, (0, 140, 255, 90), -1, cv2.LINE_AA)
        
        # Cluster of nucleons
        offsets = [(-8, -6), (6, -7), (-6, 7), (7, 5), (0, 0)]
        for idx, (ox, oy) in enumerate(offsets):
            color = (50, 50, 250, 240) if idx % 2 == 0 else (50, 200, 50, 240)
            cv2.circle(canvas, (cx + ox, cy + oy), 11, color, -1, cv2.LINE_AA)

        # 4. Text title
        self._draw_title(canvas, label.upper(), cx, int(h * 0.92))

    def _render_molecule(self, canvas: np.ndarray, label: str) -> None:
        """Draws ball-and-stick molecular structure."""
        h, w = canvas.shape[:2]
        cx, cy = w // 2, h // 2

        # Central atom (e.g. Oxygen / Carbon)
        atoms = [
            (cx, cy, 32, (30, 30, 220, 255), "O"),
            (cx - int(w * 0.22), cy + int(h * 0.15), 24, (240, 240, 240, 255), "H"),
            (cx + int(w * 0.22), cy + int(h * 0.15), 24, (240, 240, 240, 255), "H"),
        ]

        # Bonds (thick cylinders)
        bond_color = (180, 180, 180, 220)
        for i in [1, 2]:
            cv2.line(canvas, (cx, cy), (atoms[i][0], atoms[i][1]), bond_color, 8, cv2.LINE_AA)

        # Draw atom spheres with element labels
        for x, y, r, color, sym in atoms:
            # Drop shadow
            cv2.circle(canvas, (x, y), r + 3, (10, 10, 10, 120), -1, cv2.LINE_AA)
            cv2.circle(canvas, (x, y), r, color, -1, cv2.LINE_AA)
            # Text symbol inside atom
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(sym, font, 0.6, 2)
            text_color = (20, 20, 20, 255) if color[0] > 180 else (240, 240, 240, 255)
            cv2.putText(canvas, sym, (x - tw // 2, y + th // 2), font, 0.6, text_color, 2, cv2.LINE_AA)

        self._draw_title(canvas, f"MOLECULE: {label.upper()}", cx, int(h * 0.92))

    def _render_graph(self, canvas: np.ndarray, label: str) -> None:
        """Draws directed concept graph."""
        h, w = canvas.shape[:2]
        cx, cy = w // 2, h // 2

        nodes = [
            (cx, int(h * 0.25), "ROOT"),
            (int(w * 0.25), int(h * 0.65), "A"),
            (cx, int(h * 0.65), "B"),
            (int(w * 0.75), int(h * 0.65), "C"),
        ]

        # Draw directed edges
        edge_color = (0, 220, 220, 200)
        for i in [1, 2, 3]:
            cv2.arrowedLine(canvas, (nodes[0][0], nodes[0][1] + 20), (nodes[i][0], nodes[i][1] - 20), edge_color, 2, cv2.LINE_AA, tipLength=0.08)

        # Draw nodes
        for x, y, name in nodes:
            cv2.circle(canvas, (x, y), 24, (40, 40, 40, 230), -1, cv2.LINE_AA)
            cv2.circle(canvas, (x, y), 24, (0, 220, 220, 255), 2, cv2.LINE_AA)
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(name, font, 0.45, 1)
            cv2.putText(canvas, name, (x - tw // 2, y + th // 2), font, 0.45, (240, 240, 240, 255), 1, cv2.LINE_AA)

        self._draw_title(canvas, label.upper(), cx, int(h * 0.92))

    def _render_wave(self, canvas: np.ndarray, label: str) -> None:
        """Draws oscillating wave function on coordinate axis."""
        h, w = canvas.shape[:2]
        cx, cy = w // 2, h // 2

        # Axes
        axis_color = (140, 140, 140, 180)
        cv2.line(canvas, (int(w * 0.1), cy), (int(w * 0.9), cy), axis_color, 1, cv2.LINE_AA)
        cv2.line(canvas, (cx, int(h * 0.15)), (cx, int(h * 0.85)), axis_color, 1, cv2.LINE_AA)

        # Sine wave points
        pts = []
        amp = h * 0.22
        freq = 3.5
        for x in range(int(w * 0.1), int(w * 0.9)):
            phase = (x - cx) / (w * 0.8) * 2 * math.pi * freq
            y = int(cy - amp * math.sin(phase))
            pts.append([x, y])

        pts_arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [pts_arr], False, (0, 255, 200, 255), 3, cv2.LINE_AA)
        self._draw_title(canvas, label.upper(), cx, int(h * 0.92))

    def _render_generic_concept(self, canvas: np.ndarray, label: str) -> None:
        """Draws clean geometric badge representation."""
        h, w = canvas.shape[:2]
        cx, cy = w // 2, h // 2

        # Concentric glowing rings
        cv2.circle(canvas, (cx, cy), int(w * 0.32), (0, 220, 220, 100), 2, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), int(w * 0.22), (255, 200, 50, 160), 2, cv2.LINE_AA)

        # Center diamond
        d = int(w * 0.12)
        pts = np.array([[cx, cy - d], [cx + d, cy], [cx, cy + d], [cx - d, cy]], dtype=np.int32)
        cv2.fillPoly(canvas, [pts], (40, 40, 40, 220), cv2.LINE_AA)
        cv2.polylines(canvas, [pts], True, (0, 220, 220, 255), 2, cv2.LINE_AA)

        self._draw_title(canvas, label.upper(), cx, cy + 6)

    def _draw_title(self, canvas: np.ndarray, text: str, cx: int, y: int) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.65
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        tx = max(10, cx - tw // 2)

        # Background badge for text
        pad_x = 8
        pad_y = 4
        bx1 = tx - pad_x
        by1 = y - th - pad_y
        bx2 = tx + tw + pad_x
        by2 = y + baseline + pad_y
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (20, 20, 20, 200), -1)
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (0, 220, 220, 255), 1)

        cv2.putText(canvas, text, (tx, y), font, scale, (0, 220, 220, 255), thickness, cv2.LINE_AA)

