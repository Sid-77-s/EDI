"""
integration/generation — Instant Multimodal 3D Representation Subsystem for Magic Mirror.
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
from integration.generation.model3d import (
    Material3D,
    Mesh3D,
    Model3D,
    ModelNode3D,
    load_glb,
    load_obj,
)
from integration.generation.renderer3d import SoftwareRenderer3D
from integration.generation.representation_planner import (
    RepresentationCategory,
    RepresentationPlanner,
)
from integration.generation.instant_engine import InstantRepresentationEngine
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
    "Material3D",
    "Mesh3D",
    "Model3D",
    "ModelNode3D",
    "load_glb",
    "load_obj",
    "SoftwareRenderer3D",
    "RepresentationCategory",
    "RepresentationPlanner",
    "InstantRepresentationEngine",
    "OrdinaryObject3DBuilder",
    "build_atom_model",
    "build_circuit_model",
    "build_heart_model",
    "build_mechanical_assembly_model",
    "build_molecule_model",
    "build_solar_system_model",
    "build_wave_model",
]
