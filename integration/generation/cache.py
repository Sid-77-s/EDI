"""
integration/generation/cache.py — Asset caching and persistence manager.

Ensures generated assets are reused across frames and sessions without duplicate generation.
Integrates directly with filesystem storage and ConceptAssetStore.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Union
import cv2
import numpy as np

from MagicMirror_Visualisation.src.perception.assets import ConceptAsset, ConceptAssetStore
from integration.generation.schemas import GenerationResult, JobState

logger = logging.getLogger(__name__)


class AssetCache:
    """Manages disk and in-memory caching of visual assets."""

    def __init__(
        self,
        asset_dir: Union[str, Path] = "assets",
        asset_store: Optional[ConceptAssetStore] = None,
    ) -> None:
        self.asset_dir = Path(asset_dir)
        self.asset_dir.mkdir(parents=True, exist_ok=True)
        self.asset_store = asset_store
        self._memory_cache: Dict[str, ConceptAsset] = {}

    def normalize_key(self, concept: Optional[str]) -> Optional[str]:
        if not concept:
            return None
        cleaned = concept.strip().lower()
        if not cleaned or cleaned in ("—", "unknown", "none"):
            return None
        return cleaned

    def get(self, concept: str) -> Optional[ConceptAsset]:
        """Looks up asset in memory, ConceptAssetStore, and filesystem."""
        key = self.normalize_key(concept)
        if not key:
            return None

        # 1. In-memory local cache
        if key in self._memory_cache:
            asset = self._memory_cache[key]
            if asset.is_valid_image:
                return asset

        # 2. ConceptAssetStore check
        if self.asset_store:
            store_asset = self.asset_store.get_asset(key)
            if store_asset and store_asset.is_valid_image:
                self._memory_cache[key] = store_asset
                return store_asset

        # 3. Disk check in asset directory
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            # Check standard and prefixed variants
            candidates = [
                self.asset_dir / f"{key}{ext}",
                self.asset_dir / f"proc_{key}{ext}",
                self.asset_dir / f"edit_{key}{ext}",
            ]
            for cand in candidates:
                if cand.is_file():
                    img = cv2.imread(str(cand), cv2.IMREAD_UNCHANGED)
                    if img is not None and img.size > 0:
                        cached_asset = ConceptAsset(
                            concept=concept.strip(),
                            image=img,
                            source_uri=str(cand),
                            status="ready",
                        )
                        self._memory_cache[key] = cached_asset
                        if self.asset_store:
                            self.asset_store.register_asset(
                                concept=concept.strip(),
                                image_or_path=img,
                                source_uri=str(cand),
                                status="ready",
                            )
                        logger.info("Asset cache hit from disk for '%s': %s", key, cand)
                        return cached_asset

        return None

    def store(
        self,
        concept: str,
        image_or_path: Union[np.ndarray, str, Path],
        source_uri: Optional[str] = None,
    ) -> ConceptAsset:
        """Saves image to cache directory and registers with store."""
        key = self.normalize_key(concept)
        if not key:
            raise ValueError(f"Invalid concept name for caching: {concept}")

        image: Optional[np.ndarray] = None
        uri_str = source_uri

        if isinstance(image_or_path, np.ndarray):
            image = image_or_path
            out_file = self.asset_dir / f"{key}.png"
            cv2.imwrite(str(out_file), image)
            uri_str = str(out_file)
        elif isinstance(image_or_path, (str, Path)):
            p = Path(image_or_path)
            uri_str = str(p)
            if p.is_file():
                image = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)

        asset = ConceptAsset(
            concept=concept.strip(),
            image=image,
            source_uri=uri_str,
            status="ready" if (image is not None and image.size > 0) else "error",
        )
        self._memory_cache[key] = asset

        if self.asset_store:
            self.asset_store.register_asset(
                concept=concept.strip(),
                image_or_path=image if image is not None else uri_str,
                source_uri=uri_str,
                status=asset.status,
            )

        return asset

    def clear(self) -> None:
        self._memory_cache.clear()

