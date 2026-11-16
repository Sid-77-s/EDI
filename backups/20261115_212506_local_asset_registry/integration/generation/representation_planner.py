"""
integration/generation/representation_planner.py — Representation Category Selection Component.

Classifies incoming VisualPlan or concept requests into:
- 3D_OBJECT: Ordinary physical real-world objects (apple, car, chair, dog, tree, cup)
- SCIENTIFIC_MODEL: Domain-specific scientific models (atom, molecule, solar system, wave, heart)
- TECHNICAL_ASSEMBLY: Mechanical, electrical, and engineering systems (gears, engine, circuit)
- 2D_ASSET: Secondary fallback for text cards, photos, or planar diagrams
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional, Set

from integration.multimodal_fusion.schemas import RepresentationType, VisualPlan


class RepresentationCategory(str, Enum):
    OBJECT_3D = "3D_OBJECT"
    SCIENTIFIC_MODEL = "SCIENTIFIC_MODEL"
    TECHNICAL_ASSEMBLY = "TECHNICAL_ASSEMBLY"
    ASSET_2D = "2D_ASSET"


SCIENTIFIC_KEYWORDS: Set[str] = {
    "atom", "bohr", "electron", "proton", "neutron", "nucleus", "orbital",
    "molecule", "water", "h2o", "co2", "methane", "ch4", "salt", "nacl",
    "chemical", "dna", "protein", "glucose", "caffeine",
    "solar system", "planet", "sun", "orbit", "earth", "mars", "jupiter", "saturn",
    "wave", "oscillation", "frequency", "sinusoid", "electromagnetic",
    "heart", "cardiac", "organ", "brain", "lung", "cell", "mitochondria",
}

TECHNICAL_KEYWORDS: Set[str] = {
    "gear", "gears", "piston", "engine", "motor", "mechanical", "linkage",
    "assembly", "crankshaft", "transmission", "mechanism",
    "circuit", "resistor", "battery", "capacitor", "led", "diode", "schematic",
    "wire", "electrical", "pcb",
}

PLAIN_2D_KEYWORDS: Set[str] = {
    "photo", "photograph", "flat image", "text card", "slide", "2d diagram",
}


class RepresentationPlanner:
    """Lightweight planner selecting representation category from structured plan info."""

    @classmethod
    def select_category(
        cls,
        concept: str,
        plan: Optional[VisualPlan] = None,
        properties: Optional[Dict[str, Any]] = None,
    ) -> RepresentationCategory:
        """Determines best representation category without natural language processing."""
        clean = concept.strip().lower()

        # Check explicit VisualPlan representation type
        if plan and plan.representation:
            rep_type = plan.representation.type
            if rep_type == RepresentationType.TEXT:
                return RepresentationCategory.ASSET_2D
            elif rep_type == RepresentationType.DIAGRAM:
                # Diagrams could be scientific or 2D; check keywords
                if any(k in clean for k in SCIENTIFIC_KEYWORDS):
                    return RepresentationCategory.SCIENTIFIC_MODEL
                if any(k in clean for k in TECHNICAL_KEYWORDS):
                    return RepresentationCategory.TECHNICAL_ASSEMBLY

        # Explicit 2D check
        if any(k in clean for k in PLAIN_2D_KEYWORDS):
            return RepresentationCategory.ASSET_2D

        # Scientific models
        if any(k in clean for k in SCIENTIFIC_KEYWORDS):
            return RepresentationCategory.SCIENTIFIC_MODEL

        # Technical assemblies
        if any(k in clean for k in TECHNICAL_KEYWORDS):
            return RepresentationCategory.TECHNICAL_ASSEMBLY

        # Default for physical objects is 3D_OBJECT
        return RepresentationCategory.OBJECT_3D

