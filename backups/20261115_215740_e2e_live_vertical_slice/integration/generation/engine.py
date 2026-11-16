"""
integration/generation/engine.py — Hybrid Generation Engine for Magic Mirror.

Orchestrates:
- Local SDXL (ComfyUI) and Cloud FLUX (Black Forest Labs)
- Procedural scientific visual rendering
- Asset caching and ConceptAssetStore lifecycle synchronization
- Non-blocking background worker pool (never stalls camera/perception loop)
- Graceful fallbacks and robust error reporting
"""

from __future__ import annotations

import concurrent.futures
import logging
import threading
import time
from pathlib import Path
from typing import Callable, Dict, Optional, Set, Union
import numpy as np

from MagicMirror_Visualisation.src.perception.assets import (
    ConceptAsset,
    ConceptAssetStore,
    get_global_asset_store,
)
from integration.generation.cache import AssetCache
from integration.generation.instant_engine import InstantRepresentationEngine
from integration.generation.model3d import Model3D
from integration.generation.providers.base import BaseImageProvider
from integration.generation.providers.flux_cloud import FluxCloudProvider
from integration.generation.providers.procedural import ProceduralVisualProvider
from integration.generation.providers.sdxl_local import ComfyUIProvider
from integration.generation.representation_planner import (
    RepresentationCategory,
    RepresentationPlanner,
)
from integration.generation.router import GenerationRouter
from integration.generation.schemas import (
    GenerationMode,
    GenerationRequest,
    GenerationResult,
    JobState,
    JobStatus,
    ProviderType,
    QualityTier,
)

logger = logging.getLogger(__name__)


class GenerationEngine:
    """Central engine managing hybrid image generation, routing, and asset delivery."""

    def __init__(
        self,
        asset_store: Optional[ConceptAssetStore] = None,
        asset_dir: Union[str, Path] = "assets",
        flux_provider: Optional[FluxCloudProvider] = None,
        sdxl_provider: Optional[ComfyUIProvider] = None,
        procedural_provider: Optional[ProceduralVisualProvider] = None,
        instant_engine: Optional[InstantRepresentationEngine] = None,
        asset_pack_dir: Optional[Union[str, Path]] = None,
        allowed_asset_licenses=None,
        allow_unknown_asset_licenses: bool = False,
        on_asset_ready: Optional[Callable[[str, ConceptAsset], None]] = None,
        on_asset_failed: Optional[Callable[[str, str], None]] = None,
        max_workers: int = 2,
    ) -> None:
        self.asset_dir = Path(asset_dir)
        self.asset_dir.mkdir(parents=True, exist_ok=True)
        self.asset_store = asset_store or get_global_asset_store()
        self.on_asset_ready = on_asset_ready
        self.on_asset_failed = on_asset_failed

        # Instantiate instant 3D engine (local-first, zero-cost, immediate)
        self.instant_engine = instant_engine or InstantRepresentationEngine(
            library_dir=self.asset_dir / "models",
            asset_pack_dir=asset_pack_dir or self.asset_dir / "models",
            allowed_asset_licenses=allowed_asset_licenses,
            allow_unknown_asset_licenses=allow_unknown_asset_licenses,
        )

        # Instantiate secondary 2D providers
        self.flux_provider = flux_provider or FluxCloudProvider(asset_dir=self.asset_dir)
        self.sdxl_provider = sdxl_provider or ComfyUIProvider(asset_dir=self.asset_dir)
        self.procedural_provider = procedural_provider or ProceduralVisualProvider(asset_dir=self.asset_dir)

        self.providers: Dict[ProviderType, BaseImageProvider] = {
            ProviderType.FLUX_CLOUD: self.flux_provider,
            ProviderType.SDXL_LOCAL: self.sdxl_provider,
            ProviderType.PROCEDURAL: self.procedural_provider,
        }

        self.cache = AssetCache(asset_dir=self.asset_dir, asset_store=self.asset_store)
        self.router = GenerationRouter(providers=self.providers, cache=self.cache)

        # Worker pool for non-blocking secondary generation
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="GenEngineWorker",
        )
        self._lock = threading.Lock()
        self._active_jobs: Dict[str, JobStatus] = {}
        self._generating_concepts: Set[str] = set()
        self._current_concept: Optional[str] = None

    def normalize_key(self, concept: Optional[str]) -> Optional[str]:
        if not concept:
            return None
        cleaned = concept.strip().lower()
        if not cleaned or cleaned in ("—", "unknown", "none"):
            return None
        return cleaned

    def request_generation(self, request: GenerationRequest) -> JobStatus:
        """
        Submits request. For 3D representations (the primary output), delivers immediately
        via InstantRepresentationEngine with zero latency stalls.
        For 2D assets, executes via cached/procedural background worker.
        """
        key = self.normalize_key(request.concept)
        if not key:
            raise ValueError(f"Invalid concept name: '{request.concept}'")

        with self._lock:
            self._current_concept = key

            # Check representation category
            cat = RepresentationPlanner.select_category(request.concept)
            is_3d_request = request.prefer_3d and (cat != RepresentationCategory.ASSET_2D)

            # Instant 3D Representation (Primary Output Path)
            if is_3d_request:
                model, latency_ms = self.instant_engine.get_or_create_representation(
                    concept=request.concept,
                    visual_id=request.visual_id,
                )

                # Render preview snapshot for 2D asset compatibility
                preview_img = np.zeros((256, 256, 3), dtype=np.uint8)
                try:
                    from integration.generation.renderer3d import SoftwareRenderer3D
                    renderer = SoftwareRenderer3D()
                    renderer.render(preview_img, model, bbox=(0.0, 0.0, 1.0, 1.0), t=0.0)
                except Exception:
                    preview_img = None

                registered_asset = self.asset_store.register_asset(
                    concept=request.concept,
                    image_or_path=preview_img,
                    source_uri=f"3d://{key}",
                    status="ready",
                    model_3d=model,
                    version=model.version,
                )

                job = JobStatus(
                    job_id=f"3d_{key}_{model.version}",
                    concept=request.concept,
                    status=JobState.READY,
                    progress=1.0,
                    provider="instant_3d",
                    result=GenerationResult(
                        concept=request.concept,
                        visual_id=request.visual_id,
                        image=preview_img,
                        model_3d=model,
                        status=JobState.READY,
                        provider="instant_3d",
                        model=model.category,
                        latency_ms=latency_ms,
                        metadata={"category": model.category, "version": model.version},
                    ),
                    created_at=time.time(),
                    updated_at=time.time(),
                )
                self._active_jobs[key] = job

                if self.on_asset_ready:
                    try:
                        self.on_asset_ready(request.concept, registered_asset)
                    except Exception as cb_err:
                        logger.error("Error in on_asset_ready callback: %s", cb_err)

                return job

            # Secondary 2D Image / Asset Generation Path
            cached_asset = self.cache.get(key)
            if cached_asset and cached_asset.is_valid_image:
                job = JobStatus(
                    job_id=f"cached_{key}",
                    concept=request.concept,
                    status=JobState.READY,
                    progress=1.0,
                    provider="cache",
                    result=GenerationResult(
                        concept=request.concept,
                        visual_id=request.visual_id,
                        image=cached_asset.image,
                        image_path=cached_asset.source_uri,
                        status=JobState.READY,
                        provider="cache",
                    ),
                    created_at=time.time(),
                    updated_at=time.time(),
                )
                self._active_jobs[key] = job
                return job

            # Check if already generating 2D
            if key in self._generating_concepts:
                existing_job = self._active_jobs.get(key)
                if existing_job:
                    return existing_job

            # Register 'loading' placeholder immediately so yellow box stays visible
            self.asset_store.register_asset(
                concept=request.concept,
                image_or_path="",
                status="loading",
            )

            job_id = f"gen_{key}_{int(time.time() * 1000)}"
            job = JobStatus(
                job_id=job_id,
                concept=request.concept,
                status=JobState.RUNNING,
                progress=0.1,
                created_at=time.time(),
                updated_at=time.time(),
            )
            self._active_jobs[key] = job
            self._generating_concepts.add(key)

            self._executor.submit(self._execute_generation_task, request, job_id, key)
            return job

    def update_representation(self, visual_id: str, updates: dict) -> Optional[Model3D]:
        """Modifies an existing active 3D representation where possible."""
        updated_model = self.instant_engine.update_representation(visual_id, updates)
        if updated_model:
            self.asset_store.register_3d_model(
                concept=updated_model.concept,
                model_3d=updated_model,
                version=updated_model.version,
            )
        return updated_model

    def remove_representation(self, visual_id: str) -> None:
        """Removes a representation from active tracking."""
        self.instant_engine.remove_representation(visual_id)

    def _execute_generation_task(self, request: GenerationRequest, job_id: str, key: str) -> None:
        """Background thread worker function."""
        logger.info("Starting visual generation for concept '%s' (job: %s)", key, job_id)
        decision = self.router.route(request)
        logger.info("Routing decision for '%s': %s (%s)", key, decision.provider_type.value, decision.reason)

        chain = [decision.provider_type] + [p for p in decision.fallback_chain if p != decision.provider_type]
        final_result: Optional[GenerationResult] = None
        attempted_errors = []

        for ptype in chain:
            if ptype == ProviderType.CACHE:
                cached = self.cache.get(key)
                if cached and cached.is_valid_image:
                    final_result = GenerationResult(
                        concept=request.concept,
                        visual_id=request.visual_id,
                        image=cached.image,
                        image_path=cached.source_uri,
                        status=JobState.READY,
                        provider="cache",
                    )
                    break
                continue

            provider = self.providers.get(ptype)
            if not provider or not provider.is_available():
                logger.info("Provider '%s' unavailable, trying next in fallback chain", ptype.value)
                continue

            try:
                logger.info("Executing generation with provider '%s' for '%s'", ptype.value, key)
                if request.mode == GenerationMode.IMAGE_EDIT:
                    res = provider.edit(request)
                else:
                    res = provider.generate(request)

                if res and res.is_success:
                    final_result = res
                    break
                else:
                    err = res.error_message if res else "Unknown provider failure"
                    attempted_errors.append(f"{ptype.value}: {err}")
                    logger.warning("Provider '%s' failed for '%s': %s", ptype.value, key, err)
            except Exception as e:
                attempted_errors.append(f"{ptype.value}: {e}")
                logger.warning("Exception during '%s' generation for '%s': %s", ptype.value, key, e)

        # Process completion or failure under lock
        with self._lock:
            self._generating_concepts.discard(key)
            is_still_active = (self._current_concept == key)

            if final_result and final_result.is_success:
                # Cache and store asset
                registered_asset = self.cache.store(
                    concept=request.concept,
                    image_or_path=final_result.image,
                    source_uri=final_result.image_path,
                )

                job = self._active_jobs.get(key)
                if job:
                    job.status = JobState.READY
                    job.progress = 1.0
                    job.provider = final_result.provider
                    job.result = final_result
                    job.updated_at = time.time()

                logger.info("Generation succeeded for '%s' via %s in %.1fms", key, final_result.provider, final_result.latency_ms)

                if is_still_active and self.on_asset_ready:
                    try:
                        self.on_asset_ready(request.concept, registered_asset)
                    except Exception as cb_err:
                        logger.error("Error in on_asset_ready callback: %s", cb_err)

            else:
                error_summary = "; ".join(attempted_errors) if attempted_errors else "All generation providers failed"
                self.asset_store.register_asset(
                    concept=request.concept,
                    image_or_path="",
                    status="error",
                )
                job = self._active_jobs.get(key)
                if job:
                    job.status = JobState.FAILED
                    job.error = error_summary
                    job.updated_at = time.time()

                logger.error("Generation completely failed for '%s': %s", key, error_summary)

                if is_still_active and self.on_asset_failed:
                    try:
                        self.on_asset_failed(request.concept, error_summary)
                    except Exception as cb_err:
                        logger.error("Error in on_asset_failed callback: %s", cb_err)

    def get_job_status(self, concept: str) -> Optional[JobStatus]:
        key = self.normalize_key(concept)
        if not key:
            return None
        with self._lock:
            return self._active_jobs.get(key)

    def is_generating(self, concept: str) -> bool:
        key = self.normalize_key(concept)
        if not key:
            return False
        with self._lock:
            return key in self._generating_concepts

    def shutdown(self, wait: bool = False) -> None:
        """Shuts down background generation worker threads."""
        self._executor.shutdown(wait=wait)

