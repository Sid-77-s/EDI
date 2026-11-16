"""
integration/generation — Hybrid Image Generation Subsystem for Magic Mirror.
"""

from integration.generation.schemas import (
    GenerationMode,
    GenerationRequest,
    GenerationResult,
    JobState,
    JobStatus,
    ProviderType,
    QualityTier,
    RoutingDecision,
)
from integration.generation.cache import AssetCache
from integration.generation.router import GenerationRouter
from integration.generation.providers.base import BaseImageProvider
from integration.generation.providers.flux_cloud import FluxCloudProvider
from integration.generation.providers.sdxl_local import ComfyUIProvider
from integration.generation.providers.procedural import ProceduralVisualProvider
from integration.generation.engine import GenerationEngine

__all__ = [
    "GenerationMode",
    "GenerationRequest",
    "GenerationResult",
    "JobState",
    "JobStatus",
    "ProviderType",
    "QualityTier",
    "RoutingDecision",
    "AssetCache",
    "GenerationRouter",
    "BaseImageProvider",
    "FluxCloudProvider",
    "ComfyUIProvider",
    "ProceduralVisualProvider",
    "GenerationEngine",
]

