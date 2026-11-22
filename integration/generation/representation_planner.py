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
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set, Tuple

from integration.multimodal_fusion.schemas import RepresentationType, VisualPlan


class RepresentationCategory(str, Enum):
    OBJECT_3D = "3D_OBJECT"
    SCIENTIFIC_MODEL = "SCIENTIFIC_MODEL"
    TECHNICAL_ASSEMBLY = "TECHNICAL_ASSEMBLY"
    ASSET_2D = "2D_ASSET"


class FidelityLevel(str, Enum):
    EXACT = "exact"
    SIMPLIFIED = "simplified"
    ILLUSTRATIVE = "illustrative"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class ConceptIdentity:
    canonical_id: str
    entity_type: str
    subtype: str
    aliases: Tuple[str, ...]
    distinguishing_features: Tuple[str, ...]


@dataclass(frozen=True)
class RepresentationEligibility:
    concept: str
    identity: Optional[ConceptIdentity]
    fidelity: FidelityLevel
    confidence: float
    validation_reasons: Tuple[str, ...]
    available: bool


_CONCEPTS = (
    ConceptIdentity("planet.jupiter", "celestial_body", "planet", ("jupiter",), ("single planet", "banded gas giant")),
    ConceptIdentity("planet.earth", "celestial_body", "planet", ("earth", "planet earth", "the earth"), ("single planet", "land and ocean surface")),
    ConceptIdentity("atom.oxygen", "atom", "oxygen atom", ("oxygen atom",), ("single oxygen atom", "eight protons", "eight electrons")),
    ConceptIdentity("molecule.oxygen", "molecule", "oxygen molecule", ("oxygen molecule", "o2", "dioxygen"), ("two oxygen atoms", "bond")),
    ConceptIdentity("system.solar", "system", "solar system", ("solar system",), ("sun", "multiple orbiting planets")),
    ConceptIdentity("object.apple", "physical_object", "fruit", ("apple",), ("lobed fruit body", "stem")),
    ConceptIdentity("object.atom", "atom", "generic atom", ("atom", "bohr atom", "bohr model"), ("nucleus", "electrons")),
    ConceptIdentity("object.molecule", "molecule", "generic molecule", ("molecule",), ("multiple atoms", "bonds")),
    ConceptIdentity("object.water", "molecule", "water molecule", ("water molecule", "h2o"), ("one oxygen", "two hydrogen atoms", "bent geometry")),
    *(ConceptIdentity(f"science.{x}", "scientific_model", x, (x,), features) for x, features in (
        ("wave", ("oscillating wave form",)), ("heart", ("heart chambers", "major vessels")),
        ("solar system", ("sun", "multiple orbiting planets")),
    )),
    *(ConceptIdentity(f"technical.{x}", "technical_assembly", x, aliases, features) for x, aliases, features in (
        ("gear train", ("gear train", "gears"), ("interlocking gears",)),
        ("electrical circuit", ("electrical circuit", "circuit"), ("electrical components", "conductive path")),
    )),
    *(ConceptIdentity(f"object.{x}", "physical_object", x, tuple(aliases), tuple(features)) for x, aliases, features in (
        ("chair", ("chair", "seat"), ("seat", "legs", "backrest")), ("car", ("car", "vehicle", "automobile"), ("body", "wheels")),
        ("tree", ("tree", "plant"), ("trunk", "foliage")), ("cup", ("cup", "mug", "coffee cup"), ("vessel", "opening", "handle")),
        ("book", ("book", "manual", "notebook"), ("covers", "pages")), ("dog", ("dog", "puppy", "hound"), ("four limbs", "head", "tail")),
        ("person", ("person", "human", "avatar", "man", "woman"), ("head", "torso", "limbs")),
    )),
)


def validate_concept(concept: str) -> RepresentationEligibility:
    """Resolve exact identity and procedural eligibility; never guess by broad substring."""
    clean = " ".join(concept.casefold().replace("-", " ").split())
    identity = next((item for item in _CONCEPTS if clean in item.aliases), None)
    if identity is None:
        return RepresentationEligibility(concept, None, FidelityLevel.UNAVAILABLE, 0.0,
                                         ("no verified exact identity or faithful procedural template",), False)
    supported = {"system.solar",
                 "object.apple", "object.atom", "object.molecule", "object.water", "object.chair", "object.car",
                 "object.tree", "object.cup", "object.book", "object.dog", "object.person", "science.wave",
                 "science.heart", "science.solar system", "technical.gear train", "technical.electrical circuit"}
    if identity.canonical_id not in supported:
        return RepresentationEligibility(concept, identity, FidelityLevel.UNAVAILABLE, 0.2,
                                         ("identity recognized but no verified representation template",), False)
    return RepresentationEligibility(concept, identity, FidelityLevel.ILLUSTRATIVE, 0.85,
                                     ("matched exact canonical identity", "procedural template preserves defining features"), True)


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
        eligibility = validate_concept(concept)
        if eligibility.identity:
            if eligibility.identity.entity_type in ("atom", "molecule", "system", "celestial_body", "scientific_model"):
                return RepresentationCategory.SCIENTIFIC_MODEL
            if eligibility.identity.entity_type == "technical_assembly":
                return RepresentationCategory.TECHNICAL_ASSEMBLY
            if eligibility.identity.entity_type == "physical_object":
                return RepresentationCategory.OBJECT_3D

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

