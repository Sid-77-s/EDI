"""
integration/generation/instant_engine.py — Instant, Zero-Cost, Local-First 3D Representation Engine.

Enforces:
1. Strict provider order: Cache -> Library -> Procedural 3D -> Domain Templates -> Generic Fallback -> Async Refinement.
2. Ultra-low latency: Immediate synchronous construction (< 50 ms), zero neural generation stalls.
3. Zero-cost offline operation: Strictly forbids paid APIs, cloud services, and external subscriptions.
4. Persistent Visual ID across CREATE, MOVE, UPDATE, ANIMATE.
5. Version checks for stale asynchronous refinement task rejection.
"""

from __future__ import annotations

import concurrent.futures
import logging
import time
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union
import numpy as np

from integration.generation.model3d import Model3D
from integration.generation.animation_engine import AnimationConfig, AnimationEngine
from integration.generation.asset_catalogue import LocalAssetPackResolver
from integration.generation.asset_retrieval import Local3DAssetRetriever
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
from integration.generation.representation_planner import (
    FidelityLevel,
    RepresentationCategory,
    RepresentationPlanner,
    validate_concept,
)
from integration.generation.visual_quality import enhance_and_validate
from integration.multimodal_fusion.schemas import VisualPlan

logger = logging.getLogger(__name__)


class RepresentationUnavailableError(ValueError):
    """A concept has no verified exact asset or faithful local template."""

    def __init__(self, eligibility):
        self.eligibility = eligibility
        super().__init__(f"Representation unavailable for '{eligibility.concept}': " + "; ".join(eligibility.validation_reasons))


class RepresentationQualityError(ValueError):
    """A verified identity did not meet its visual quality requirements."""

    def __init__(self, concept: str, result):
        self.result = result
        super().__init__(f"Visual quality rejected '{concept}': " + "; ".join(result.reasons))


class InstantRepresentationEngine:
    """Instant local 3D representation engine providing immediate visual models."""

    def __init__(
        self,
        library_dir: Union[str, Path] = "assets/models",
        on_refinement_complete: Optional[Callable[[Model3D], None]] = None,
        asset_pack_dir: Optional[Union[str, Path]] = None,
        allowed_asset_licenses=None,
        allow_unknown_asset_licenses: bool = False,
        asset_retriever: Optional[Local3DAssetRetriever] = None,
        catalogue_db: Optional[Union[str, Path]] = None,
        synonyms: Optional[Dict[str, list[str]]] = None,
    ) -> None:
        self.library_dir = Path(library_dir)
        self.library_dir.mkdir(parents=True, exist_ok=True)
        self.ordinary_builder = OrdinaryObject3DBuilder(
            model_library_dir=self.library_dir,
            allowed_licenses=allowed_asset_licenses,
            allow_unknown_license=allow_unknown_asset_licenses,
        )
        self.asset_resolver = LocalAssetPackResolver(
            asset_pack_dir or self.library_dir,
            allowed_licenses=allowed_asset_licenses,
            allow_unknown_license=allow_unknown_asset_licenses,
        )
        if asset_retriever is not None:
            self.asset_retriever: Optional[Local3DAssetRetriever] = asset_retriever
        elif catalogue_db is not None:
            self.asset_retriever = Local3DAssetRetriever(
                database_path=catalogue_db,
                models_dir=asset_pack_dir or self.library_dir,
                allowed_licenses=allowed_asset_licenses,
                allow_unknown_license=allow_unknown_asset_licenses,
                synonyms=synonyms,
            )
        else:
            self.asset_retriever = None
        self.on_refinement_complete = on_refinement_complete
        self.animation_engine = AnimationEngine()

        # In-memory asset cache and active visual tracking
        self._cache: Dict[str, Model3D] = {}
        self._active_visuals: Dict[str, Model3D] = {}  # visual_id -> Model3D
        self._visual_versions: Dict[str, int] = {}     # visual_id -> current_version

        # Background thread pool for non-blocking local refinements
        self._refine_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="3DRefineWorker",
        )

    def normalize_key(self, concept: Optional[str]) -> Optional[str]:
        if not concept:
            return None
        cleaned = concept.strip().lower()
        if not cleaned or cleaned in ("—", "unknown", "none"):
            return None
        return cleaned

    def get_or_create_representation(
        self,
        concept: str,
        visual_id: Optional[str] = None,
        plan: Optional[VisualPlan] = None,
    ) -> Tuple[Model3D, float]:
        """
        Retrieves or constructs a 3D model instantly.
        Returns:
            Tuple[Model3D, construction_latency_ms]
        """
        t0 = time.perf_counter()
        key = self.normalize_key(concept)
        if not key:
            raise ValueError(f"Invalid concept: '{concept}'")
        v_id = visual_id or f"vis_{key}"

        eligibility = validate_concept(concept)
        asset_required_concepts = {"planet.jupiter", "planet.earth", "atom.oxygen", "molecule.oxygen"}
        indexed_exact = None
        if self.asset_retriever is not None:
            try:
                candidate = self.asset_retriever.resolve(concept)
                if candidate is not None and int(candidate[1].get("match_rank", 99)) <= 1:
                    indexed_exact = candidate
            except Exception:
                indexed_exact = None
            if indexed_exact is not None and (not eligibility.available or eligibility.identity.canonical_id in asset_required_concepts):
                from integration.generation.representation_planner import ConceptIdentity, RepresentationEligibility
                identity = ConceptIdentity("asset." + "_".join(key.split()), "curated_asset", key,
                                           (key,), ("exact catalogue name or alias match",))
                eligibility = RepresentationEligibility(concept, identity, FidelityLevel.EXACT, 0.95,
                                                        ("matched exact catalogue metadata",), True)
        curated_exact = None
        if self.asset_resolver is not None:
            # Curated sidecars are an authoritative exact-identity source for custom concepts.
            try:
                curated = self.asset_resolver.resolve(concept)
            except Exception:
                curated = None
            if curated is not None and (not eligibility.available or eligibility.identity.canonical_id in asset_required_concepts):
                curated_exact = curated
                from integration.generation.representation_planner import ConceptIdentity, RepresentationEligibility
                identity = ConceptIdentity(
                    "asset." + "_".join(key.split()), "curated_asset", key,
                    (key,), ("curated asset metadata name or alias matches the requested concept",),
                )
                eligibility = RepresentationEligibility(
                    concept, identity, FidelityLevel.EXACT, 0.95,
                    ("matched exact curated asset metadata",), True,
                )
        if not eligibility.available:
            raise RepresentationUnavailableError(eligibility)
        canonical = eligibility.identity.canonical_id
        # These templates currently describe a system or a generic chemistry model,
        # so they cannot faithfully stand in for a single planet or oxygen species.
        if indexed_exact is None and curated_exact is None and canonical in asset_required_concepts:
            raise RepresentationUnavailableError(type(eligibility)(
                eligibility.concept, eligibility.identity, FidelityLevel.UNAVAILABLE, 0.0,
                ("no exact single-entity template is available",), False))

        if indexed_exact is not None or curated_exact is not None:
            model, _metadata = indexed_exact or curated_exact
            self._apply_visual_quality(model, canonical, eligibility)
            model.visual_id = v_id
            model.version = self._visual_versions.get(v_id, 0) + 1
            model.metadata.update({"canonical_concept_id": canonical, "entity_type": eligibility.identity.entity_type,
                                   "entity_subtype": eligibility.identity.subtype, "aliases": list(eligibility.identity.aliases),
                                   "required_features": list(eligibility.identity.distinguishing_features),
                                   "representation_fidelity": eligibility.fidelity.value,
                                   "representation_confidence": eligibility.confidence,
                                   "validation_reasons": list(eligibility.validation_reasons)})
            self._visual_versions[v_id] = model.version
            self._cache[key] = model
            self._active_visuals[v_id] = model
            self.animation_engine.register(v_id, model)
            return model, (time.perf_counter() - t0) * 1000.0

        # 1. Existing Active Representation check (MOVE / persistence)
        if v_id in self._active_visuals:
            existing = self._active_visuals[v_id]
            if self.normalize_key(existing.concept) == key:
                self.animation_engine.register(v_id, existing)
                latency = (time.perf_counter() - t0) * 1000.0
                return existing, latency

        # 2. Local Asset Cache check (in-memory)
        if key in self._cache:
            model = self._cache[key]
            model.visual_id = v_id
            self._active_visuals[v_id] = model
            self.animation_engine.register(v_id, model)
            latency = (time.perf_counter() - t0) * 1000.0
            logger.info("Instant 3D cache hit for '%s' in %.2f ms", key, latency)
            return model, latency

        # 3. Local Asset Retrieval Layer (Search indexed catalogue -> GLB Importer)
        resolved = None
        if self.asset_retriever is not None:
            try:
                resolved = self.asset_retriever.resolve(concept)
                if resolved is not None and int(resolved[1].get("match_rank", 99)) > 1:
                    logger.info("Rejected non-specific catalogue match for '%s'", key)
                    resolved = None
            except Exception as exc:
                logger.warning("Local asset retriever failed for '%s': %s", key, exc)
                resolved = None

        if resolved is None and self.asset_resolver is not None:
            try:
                resolved = self.asset_resolver.resolve(concept)
            except Exception as exc:
                logger.warning("Local GLB asset load failed for '%s': %s", key, exc)
                resolved = None
        if resolved is not None:
            local_disk_asset, _metadata = resolved
            self._apply_visual_quality(local_disk_asset, canonical, eligibility)
            local_disk_asset.visual_id = v_id
            local_disk_asset.version = self._visual_versions.get(v_id, 0) + 1
            self._visual_versions[v_id] = local_disk_asset.version
            self._cache[key] = local_disk_asset
            self._active_visuals[v_id] = local_disk_asset
            self.animation_engine.register(v_id, local_disk_asset)
            latency = (time.perf_counter() - t0) * 1000.0
            logger.info("Loaded curated local GLB for '%s' in %.2f ms", key, latency)
            return local_disk_asset, latency

        local_disk_asset = self.ordinary_builder.find_local_asset(key)
        if local_disk_asset is not None:
            self._apply_visual_quality(local_disk_asset, canonical, eligibility)
            local_disk_asset.visual_id = v_id
            self._cache[key] = local_disk_asset
            self._active_visuals[v_id] = local_disk_asset
            self.animation_engine.register(v_id, local_disk_asset)
            latency = (time.perf_counter() - t0) * 1000.0
            logger.info("Loaded local 3D file for '%s' in %.2f ms", key, latency)
            return local_disk_asset, latency

        # 4. Representation Planner selection
        cat = RepresentationPlanner.select_category(concept, plan)

        model: Optional[Model3D] = None
        canonical_concept = eligibility.identity.subtype

        if cat in (RepresentationCategory.SCIENTIFIC_MODEL, RepresentationCategory.TECHNICAL_ASSEMBLY):
            model = self._build_scientific_or_technical(canonical_concept, canonical_concept)

        if model is None:
            # 5. Ordinary physical object or parametric 3D construction
            model = self.ordinary_builder.build_or_retrieve(canonical_concept)

        self._apply_visual_quality(model, canonical, eligibility)

        model.metadata.update({
            "canonical_concept_id": canonical,
            "entity_type": eligibility.identity.entity_type,
            "entity_subtype": eligibility.identity.subtype,
            "aliases": list(eligibility.identity.aliases),
            "required_features": list(eligibility.identity.distinguishing_features),
            "representation_fidelity": eligibility.fidelity.value,
            "representation_confidence": eligibility.confidence,
            "validation_reasons": list(eligibility.validation_reasons),
        })

        model.visual_id = v_id
        current_version = self._visual_versions.get(v_id, 0) + 1
        model.version = current_version
        self._visual_versions[v_id] = current_version

        # Cache in memory
        self._cache[key] = model
        self._active_visuals[v_id] = model
        self.animation_engine.register(v_id, model)

        latency = (time.perf_counter() - t0) * 1000.0
        logger.info("Constructed instant 3D model for '%s' [%s] in %.2f ms", key, cat.value, latency)

        # 6. Dispatch optional local refinement in background
        self._refine_executor.submit(self._run_async_refinement, v_id, key, current_version)

        return model, latency

    @staticmethod
    def _apply_visual_quality(model: Model3D, canonical_id: str, eligibility=None) -> None:
        result = enhance_and_validate(model, canonical_id)
        if eligibility is not None and eligibility.identity is not None:
            model.metadata.update({
                "canonical_concept_id": eligibility.identity.canonical_id,
                "entity_type": eligibility.identity.entity_type,
                "entity_subtype": eligibility.identity.subtype,
                "aliases": list(eligibility.identity.aliases),
                "required_features": list(eligibility.identity.distinguishing_features),
                "representation_fidelity": eligibility.fidelity.value,
                "representation_confidence": eligibility.confidence,
                "validation_reasons": list(eligibility.validation_reasons),
            })
        model.metadata["visual_quality"] = {
            "passed": result.passed,
            "score": result.score,
            "reasons": list(result.reasons),
            "mesh_count": result.mesh_count,
            "triangle_count": result.triangle_count,
        }
        if not result.passed:
            raise RepresentationQualityError(model.concept, result)

    def _build_scientific_or_technical(self, concept: str, key: str) -> Optional[Model3D]:
        """Routes to domain-specific scientific and technical templates."""
        if any(w in key for w in ("atom", "bohr", "electron", "proton", "nucleus", "quantum")):
            return build_atom_model(concept)
        elif any(w in key for w in ("molecule", "water", "h2o", "co2", "methane", "ch4", "salt", "nacl", "chemical", "bond")):
            return build_molecule_model(concept)
        elif any(w in key for w in ("solar system", "planet", "sun", "orbit", "earth", "mars", "jupiter", "saturn")):
            return build_solar_system_model(concept)
        elif any(w in key for w in ("wave", "frequency", "sinusoid", "oscillation")):
            return build_wave_model(concept)
        elif any(w in key for w in ("heart", "cardiac")):
            return build_heart_model(concept)
        elif any(w in key for w in ("gear", "mechanical", "piston", "engine", "assembly")):
            return build_mechanical_assembly_model(concept)
        elif any(w in key for w in ("circuit", "resistor", "battery", "electrical")):
            return build_circuit_model(concept)
        return None

    def update_representation(self, visual_id: str, updates: dict) -> Optional[Model3D]:
        """Updates an existing active 3D representation in-place."""
        model = self._active_visuals.get(visual_id)
        if model is None:
            return None

        # Version bump
        new_version = model.version + 1
        model.version = new_version
        self._visual_versions[visual_id] = new_version

        # Apply spatial / component updates
        if "scale" in updates and updates["scale"] is not None:
            s = float(updates["scale"])
            model.root.scale[:] = [s, s, s]
        if "position" in updates and updates["position"] is not None:
            model.root.position[:] = updates["position"][:3]
        if "color" in updates and updates["color"] is not None:
            for mesh, _ in model.get_all_meshes_with_transforms():
                mesh.material.diffuse_color = updates["color"]

        self.animation_engine.rebase(visual_id)

        return model

    def animate_representation(self, visual_id: str, config: AnimationConfig, request_id: Optional[str] = None) -> str:
        model = self._active_visuals.get(visual_id)
        if model is None:
            raise ValueError(f"No active representation for visual_id '{visual_id}'")
        self.animation_engine.register(visual_id, model)
        return self.animation_engine.start(visual_id, config, request_id=request_id)

    def pause_animation(self, visual_id: str, track_id: Optional[str] = None) -> bool:
        return self.animation_engine.pause(visual_id, track_id)

    def resume_animation(self, visual_id: str, track_id: Optional[str] = None) -> bool:
        return self.animation_engine.resume(visual_id, track_id)

    def stop_animation(self, visual_id: str, track_id: Optional[str] = None) -> bool:
        return self.animation_engine.stop(visual_id, track_id)

    def set_animation_diagnostic_sink(self, sink) -> None:
        self.animation_engine.set_diagnostic_sink(sink)

    def animation_compatibility(self, visual_id: str, config: AnimationConfig):
        return self.animation_engine.compatible(visual_id, config)

    def remove_representation(self, visual_id: str) -> None:
        """Removes a visual from active scene tracking."""
        self.animation_engine.unregister(visual_id)
        if visual_id in self._active_visuals:
            del self._active_visuals[visual_id]

    def _run_async_refinement(self, visual_id: str, key: str, task_version: int) -> None:
        """Background thread refining model materials and geometry without blocking main loop."""
        time.sleep(0.08)  # Simulate local detail pass (e.g. normal calculation, beveling)
        current_version = self._visual_versions.get(visual_id)

        # Stale task rejection: if representation has been updated or removed, abort
        if current_version != task_version:
            logger.debug("Rejected stale refinement task for visual_id '%s' (task_v=%d, cur_v=%s)", visual_id, task_version, current_version)
            return

        model = self._active_visuals.get(visual_id)
        if model is None:
            return

        # Refine: enhance material specular highlights and smooth normals
        for mesh, _ in model.get_all_meshes_with_transforms():
            mesh.material.specular_coeff = min(0.6, mesh.material.specular_coeff + 0.15)
            mesh.material.shininess = 32.0

        if self.on_refinement_complete:
            try:
                self.on_refinement_complete(model)
            except Exception:
                pass

    def shutdown(self) -> None:
        self.animation_engine.shutdown()
        self._refine_executor.shutdown(wait=False)

