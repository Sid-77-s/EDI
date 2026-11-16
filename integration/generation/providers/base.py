"""
integration/generation/providers/base.py — Base class for visual generation providers.
"""

from __future__ import annotations

import abc
import logging
import os
import time
from pathlib import Path
from typing import Optional, Tuple, Union
import cv2
import numpy as np

from integration.generation.schemas import GenerationRequest, GenerationResult, JobState

logger = logging.getLogger(__name__)


class BaseImageProvider(abc.ABC):
    """Abstract interface for image generation / editing providers."""

    def __init__(self, name: str) -> None:
        self.name = name

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available (credentials present, service reachable)."""
        pass

    @abc.abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate an image based on the text prompt / concept."""
        pass

    @abc.abstractmethod
    def edit(self, request: GenerationRequest) -> GenerationResult:
        """Edit or refine an existing image."""
        pass

    def save_image_to_disk(
        self,
        image: np.ndarray,
        concept: str,
        output_dir: Union[str, Path] = "assets",
        prefix: str = "",
    ) -> str:
        """Utility to safely save an OpenCV image array to PNG."""
        p_dir = Path(output_dir)
        p_dir.mkdir(parents=True, exist_ok=True)
        safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in concept.lower().strip())
        fname = f"{prefix}{safe_name}.png" if prefix else f"{safe_name}.png"
        out_path = p_dir / fname
        cv2.imwrite(str(out_path), image)
        return str(out_path)
