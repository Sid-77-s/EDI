"""
integration/generation/router.py — Configurable Provider Routing for Generation Engine.

Applies routing policy:
1. Cache lookup: reuse cached assets without invoking models.
2. Procedural scientific visuals: atom, molecule, graph, wave, diagram -> procedural engine.
3. Image editing: routes based on editing support and requested quality.
4. High-detail generation: cloud FLUX.
5. Simple educational objects: local SDXL.
6. Failure recovery: automatic fallback along configured chain.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from integration.generation.cache import AssetCache
from integration.generation.providers.base import BaseImageProvider
from integration.generation.schemas import (
    GenerationMode,
    GenerationRequest,
    ProviderType,
    QualityTier,
    RoutingDecision,
)

logger = logging.getLogger(__name__)

SCIENTIFIC_KEYWORDS = {
    "atom", "bohr", "electron", "proton", "neutron", "nucleus", "orbital",
    "molecule", "water", "h2o", "co2", "methane", "chemical", "bond", "dna",
    "cell", "mitochondria", "organelle",
    "graph", "network", "node", "tree", "flowchart", "diagram",
    "wave", "frequency", "oscillation", "vector", "force", "field", "sinusoid",
}


class GenerationRouter:
    """Selects the best visual generation provider for a given request."""

    def __init__(
        self,
        providers: Dict[ProviderType, BaseImageProvider],
        cache: Optional[AssetCache] = None,
        default_local_provider: ProviderType = ProviderType.SDXL_LOCAL,
        default_cloud_provider: ProviderType = ProviderType.FLUX_CLOUD,
    ) -> None:
        self.providers = providers
        self.cache = cache
        self.default_local = default_local_provider
        self.default_cloud = default_cloud_provider

    def route(self, request: GenerationRequest) -> RoutingDecision:
        """Determines primary provider and ordered fallback chain."""
        concept_clean = request.concept.strip().lower()

        # 1. Cache hit
        if self.cache and self.cache.get(request.concept) is not None:
            return RoutingDecision(
                provider_type=ProviderType.CACHE,
                reason="Cached suitable asset found; reusing without generating",
                fallback_chain=[],
            )

        # 2. Procedural scientific visual
        is_scientific = (
            any(kw in concept_clean for kw in SCIENTIFIC_KEYWORDS)
            or (request.representation_type and request.representation_type.lower() in ("diagram", "graph", "process"))
            or ("scientific" in (request.prompt or "").lower())
        )
        if is_scientific:
            return RoutingDecision(
                provider_type=ProviderType.PROCEDURAL,
                reason="Procedural scientific visual requested for domain concept",
                fallback_chain=[self.default_local, self.default_cloud],
            )

        # 3. Image editing
        if request.mode == GenerationMode.IMAGE_EDIT:
            if request.quality == QualityTier.HIGH_DETAIL and self._is_available(self.default_cloud):
                return RoutingDecision(
                    provider_type=self.default_cloud,
                    reason="High-detail image edit routed to cloud FLUX fill endpoint",
                    fallback_chain=[self.default_local, ProviderType.PROCEDURAL],
                )
            elif self._is_available(self.default_local):
                return RoutingDecision(
                    provider_type=self.default_local,
                    reason="Image edit routed to local SDXL img2img pipeline",
                    fallback_chain=[self.default_cloud, ProviderType.PROCEDURAL],
                )
            else:
                return RoutingDecision(
                    provider_type=self.default_cloud,
                    reason="Image edit routed to cloud FLUX (local SDXL unavailable)",
                    fallback_chain=[ProviderType.PROCEDURAL],
                )

        # 4. High-detail generation
        is_high_detail = (
            request.quality == QualityTier.HIGH_DETAIL
            or "high detail" in (request.prompt or "").lower()
            or "photorealistic" in (request.prompt or "").lower()
            or "hyperrealistic" in (request.prompt or "").lower()
        )
        if is_high_detail:
            if self._is_available(self.default_cloud):
                return RoutingDecision(
                    provider_type=self.default_cloud,
                    reason="High-detail visual generation routed to cloud FLUX",
                    fallback_chain=[self.default_local, ProviderType.PROCEDURAL],
                )
            else:
                return RoutingDecision(
                    provider_type=self.default_local,
                    reason="Cloud FLUX unavailable for high-detail; falling back to local SDXL",
                    fallback_chain=[ProviderType.PROCEDURAL],
                )

        # 5. Simple educational object generation (Standard default)
        if self._is_available(self.default_local):
            return RoutingDecision(
                provider_type=self.default_local,
                reason="Simple educational object routed to local SDXL",
                fallback_chain=[self.default_cloud, ProviderType.PROCEDURAL],
            )
        elif self._is_available(self.default_cloud):
            return RoutingDecision(
                provider_type=self.default_cloud,
                reason="Local SDXL unavailable; routing educational object to cloud FLUX",
                fallback_chain=[ProviderType.PROCEDURAL],
            )
        else:
            return RoutingDecision(
                provider_type=ProviderType.PROCEDURAL,
                reason="Neither local SDXL nor cloud FLUX available; using procedural fallback",
                fallback_chain=[],
            )

    def _is_available(self, ptype: ProviderType) -> bool:
        prov = self.providers.get(ptype)
        return prov is not None and prov.is_available()

