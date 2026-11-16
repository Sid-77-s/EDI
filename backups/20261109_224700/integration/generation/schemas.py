"""
integration/generation/schemas.py — Schemas and data models for Hybrid Generation Engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np


class GenerationMode(str, Enum):
    TEXT_TO_IMAGE = "text-to-image"
    IMAGE_EDIT = "image-edit"


class QualityTier(str, Enum):
    DRAFT = "draft"
    STANDARD = "standard"
    HIGH_DETAIL = "high-detail"


class ProviderType(str, Enum):
    SDXL_LOCAL = "sdxl-local"
    FLUX_CLOUD = "flux-cloud"
    PROCEDURAL = "procedural"
    CACHE = "cache"


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    READY = "ready"
    FAILED = "failed"


@dataclass
class GenerationRequest:
    """Request payload for visual asset generation or editing."""
    concept: str
    prompt: Optional[str] = None
    visual_id: Optional[str] = None
    mode: GenerationMode = GenerationMode.TEXT_TO_IMAGE
    input_image: Optional[Union[np.ndarray, str, Path]] = None
    quality: QualityTier = QualityTier.STANDARD
    width: int = 768
    height: int = 768
    seed: Optional[int] = None
    steps: Optional[int] = None
    representation_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def effective_prompt(self) -> str:
        """Returns prompt or a descriptive default prompt for the concept."""
        if self.prompt and self.prompt.strip():
            return self.prompt.strip()
        return f"A high-quality educational visualization of {self.concept}, isolated on dark background, detailed, 3d render"


@dataclass
class GenerationResult:
    """Output payload from a generation attempt."""
    concept: str
    visual_id: Optional[str] = None
    image: Optional[np.ndarray] = None
    image_path: Optional[str] = None
    status: JobState = JobState.READY
    provider: str = ""
    model: str = ""
    latency_ms: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        return self.status == JobState.READY and self.image is not None and self.image.size > 0


@dataclass
class JobStatus:
    """Tracking structure for a generation job."""
    job_id: str
    concept: str
    status: JobState
    progress: float = 0.0
    provider: Optional[str] = None
    error: Optional[str] = None
    result: Optional[GenerationResult] = None
    created_at: float = 0.0
    updated_at: float = 0.0


@dataclass
class RoutingDecision:
    """Outcome of router evaluation."""
    provider_type: ProviderType
    reason: str
    fallback_chain: List[ProviderType] = field(default_factory=list)

