"""
perception/assets.py — Concept Asset Store and In-Box Visual Rendering.

Provides:
1. ConceptAsset data model and ConceptAssetStore lifecycle management.
2. Graceful fallback when assets are missing, loading, or failed.
3. Proportional, un-distorted image scaling and clipping inside the yellow VisualOpportunity box.
4. Clean rendering interface preserving yellow border and high-contrast concept labels.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Colors (BGR)
_C_BORDER = (0, 220, 220)       # Yellow-teal Visual Opportunity border
_C_LABEL = (0, 220, 220)        # Label text color
_C_LABEL_BG = (20, 20, 20)      # High-contrast label background badge
_C_SHADOW = (10, 10, 10)        # Text shadow
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FS_LABEL = 0.52
_FT = 1
_LT = cv2.LINE_AA


@dataclass
class ConceptAsset:
    """Represents a visual asset (e.g. generated image or loaded icon/diagram) for a concept."""
    concept: str
    image: Optional[np.ndarray] = None
    source_uri: Optional[str] = None
    status: str = "ready"  # "ready" | "loading" | "missing" | "error"
    timestamp: float = 0.0

    @property
    def is_valid_image(self) -> bool:
        return self.image is not None and isinstance(self.image, np.ndarray) and self.image.size > 0


class ConceptAssetStore:
    """
    Manages concept-to-asset bindings.
    Guarantees:
    - When a concept changes, unrelated images from previous concepts are not shown.
    - Missing or failed assets are handled gracefully without exceptions.
    - Full concept identity is preserved in backend state.
    """

    def __init__(self, asset_dir: Optional[Union[str, Path]] = None) -> None:
        self._asset_dir: Optional[Path] = Path(asset_dir) if asset_dir else None
        self._store: Dict[str, ConceptAsset] = {}

    def normalize_key(self, concept: Optional[str]) -> Optional[str]:
        if not concept:
            return None
        cleaned = concept.strip().lower()
        if not cleaned or cleaned in ("—", "unknown", "none"):
            return None
        return cleaned

    def register_asset(
        self,
        concept: str,
        image_or_path: Union[np.ndarray, str, Path],
        source_uri: Optional[str] = None,
        status: str = "ready",
    ) -> ConceptAsset:
        """Register an in-memory image or image file path for a concept."""
        key = self.normalize_key(concept)
        if not key:
            raise ValueError(f"Invalid concept name for registration: '{concept}'")

        image: Optional[np.ndarray] = None
        uri_str = source_uri

        if isinstance(image_or_path, np.ndarray):
            image = image_or_path
        elif isinstance(image_or_path, (str, Path)):
            p = Path(image_or_path)
            uri_str = str(p)
            if p.is_file():
                loaded = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                if loaded is not None:
                    image = loaded
                    status = "ready"
                else:
                    status = "error"
            else:
                status = "missing"

        asset = ConceptAsset(
            concept=concept.strip(),
            image=image,
            source_uri=uri_str,
            status=status,
        )
        self._store[key] = asset
        return asset

    def get_asset(self, concept: Optional[str]) -> Optional[ConceptAsset]:
        """Retrieve the asset for the concept, or check filesystem if asset_dir configured."""
        key = self.normalize_key(concept)
        if not key:
            return None

        if key in self._store:
            return self._store[key]

        # Check optional asset directory
        if self._asset_dir and self._asset_dir.is_dir():
            for ext in (".png", ".jpg", ".jpeg", ".webp"):
                cand_path = self._asset_dir / f"{key}{ext}"
                if cand_path.is_file():
                    loaded = cv2.imread(str(cand_path), cv2.IMREAD_UNCHANGED)
                    if loaded is not None:
                        asset = ConceptAsset(
                            concept=concept.strip(),
                            image=loaded,
                            source_uri=str(cand_path),
                            status="ready",
                        )
                        self._store[key] = asset
                        return asset

        # Return a missing placeholder entry
        missing_asset = ConceptAsset(concept=concept.strip(), image=None, status="missing")
        return missing_asset

    def has_valid_asset(self, concept: Optional[str]) -> bool:
        asset = self.get_asset(concept)
        return asset is not None and asset.is_valid_image

    def clear(self) -> None:
        self._store.clear()


# Default singleton instance
_GLOBAL_ASSET_STORE = ConceptAssetStore()


def get_global_asset_store() -> ConceptAssetStore:
    return _GLOBAL_ASSET_STORE


def format_concept_label(concept: Optional[str]) -> str:
    """Format the concept label for clean, readable display without fabricating values."""
    if not concept:
        return "waiting for concept"
    cleaned = concept.strip()
    if not cleaned or cleaned in ("—", "unknown", "none"):
        return "waiting for concept"
    return cleaned


def clip_text_to_width(text: str, max_width: int, scale: float = _FS_LABEL, thickness: int = _FT) -> str:
    """Clips or shortens label text with ellipsis if it exceeds max_width in pixels."""
    if max_width <= 24:
        return text
    (tw, _), _ = cv2.getTextSize(text, _FONT, scale, thickness)
    if tw <= max_width:
        return text
    clipped = text
    while len(clipped) > 3:
        clipped = clipped[:-1]
        candidate = clipped + "..."
        (cw, _), _ = cv2.getTextSize(candidate, _FONT, scale, thickness)
        if cw <= max_width:
            return candidate
    return text[:4]


def _draw_dashed_rect(
    img: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 2,
    dash_length: int = 15,
) -> None:
    x1, y1 = pt1
    x2, y2 = pt2
    lines = [
        ((x1, y1), (x2, y1)),
        ((x2, y1), (x2, y2)),
        ((x2, y2), (x1, y2)),
        ((x1, y2), (x1, y1)),
    ]
    for start, end in lines:
        length = ((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5
        dashes = max(1, int(length / dash_length))
        dx = (end[0] - start[0]) / dashes
        dy = (end[1] - start[1]) / dashes
        for i in range(0, dashes, 2):
            p1 = (int(start[0] + i * dx), int(start[1] + i * dy))
            p2_x = start[0] + (i + 1) * dx
            p2_y = start[1] + (i + 1) * dy
            if i + 1 >= dashes:
                p2_x, p2_y = end[0], end[1]
            cv2.line(img, p1, (int(p2_x), int(p2_y)), color, thickness, _LT)


def render_box_and_asset(
    frame: np.ndarray,
    bbox: Tuple[float, float, float, float],
    concept_label: Optional[str] = None,
    asset: Optional[ConceptAsset] = None,
    border_color: Tuple[int, int, int] = _C_BORDER,
    thickness: int = 2,
    dash_length: int = 15,
) -> None:
    """
    Renders the visual opportunity box with:
    1. Proportional, un-distorted visual image inside the box (if available).
    2. Clipped cleanly inside box interior.
    3. Prominent yellow dashed border on top.
    4. High-contrast, legible concept badge along the border.
    """
    if frame is None or frame.size == 0:
        return

    h, w = frame.shape[:2]
    x1 = int(round(bbox[0] * w))
    y1 = int(round(bbox[1] * h))
    x2 = int(round(bbox[2] * w))
    y2 = int(round(bbox[3] * h))

    # Clamp to frame bounds
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(x1 + 10, min(w, x2))
    y2 = max(y1 + 10, min(h, y2))

    box_w = x2 - x1
    box_h = y2 - y1

    # 1. Render Image Inside Box (if asset is available and concept matches)
    if asset is not None and asset.is_valid_image:
        img = asset.image
        # Ensure interior margin
        pad = thickness + 2
        ix1 = x1 + pad
        iy1 = y1 + pad
        ix2 = x2 - pad
        iy2 = y2 - pad
        max_iw = ix2 - ix1
        max_ih = iy2 - iy1

        if max_iw > 4 and max_ih > 4:
            src_h, src_w = img.shape[:2]
            if src_w > 0 and src_h > 0:
                # Proportional scaling without stretching
                scale = min(max_iw / src_w, max_ih / src_h)
                scaled_w = max(1, int(round(src_w * scale)))
                scaled_h = max(1, int(round(src_h * scale)))

                interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
                scaled_img = cv2.resize(img, (scaled_w, scaled_h), interpolation=interp)

                # Center within interior
                dest_x = ix1 + (max_iw - scaled_w) // 2
                dest_y = iy1 + (max_ih - scaled_h) // 2

                # ROI boundaries
                rx1 = max(0, dest_x)
                ry1 = max(0, dest_y)
                rx2 = min(w, dest_x + scaled_w)
                ry2 = min(h, dest_y + scaled_h)

                crop_x1 = rx1 - dest_x
                crop_y1 = ry1 - dest_y
                crop_w = rx2 - rx1
                crop_h = ry2 - ry1

                if crop_w > 0 and crop_h > 0:
                    roi = scaled_img[crop_y1 : crop_y1 + crop_h, crop_x1 : crop_x1 + crop_w]
                    if roi.shape[2] == 4:
                        # BGRA alpha blending
                        alpha = roi[:, :, 3:4].astype(np.float32) / 255.0
                        bgr = roi[:, :, :3].astype(np.float32)
                        bg = frame[ry1:ry2, rx1:rx2].astype(np.float32)
                        blended = bgr * alpha + bg * (1.0 - alpha)
                        frame[ry1:ry2, rx1:rx2] = blended.astype(np.uint8)
                    else:
                        frame[ry1:ry2, rx1:rx2] = roi[:, :, :3]

    # 2. Draw Yellow Dashed Border (always crisp and on top)
    _draw_dashed_rect(frame, (x1, y1), (x2, y2), border_color, thickness=thickness, dash_length=dash_length)

    # 3. Draw Concept Title Badge
    display_concept = format_concept_label(concept_label)
    clipped_label = clip_text_to_width(display_concept, max(30, box_w - 12), scale=_FS_LABEL, thickness=_FT)

    (tw, th), baseline = cv2.getTextSize(clipped_label, _FONT, _FS_LABEL, _FT)

    # Position badge along top border
    badge_pad_x = 6
    badge_pad_y = 3
    badge_w = tw + badge_pad_x * 2
    badge_h = th + badge_pad_y * 2 + baseline

    badge_x1 = x1 + 4
    badge_x2 = min(w - 2, badge_x1 + badge_w)

    if y1 >= badge_h + 4:
        badge_y1 = y1 - badge_h
        badge_y2 = y1
    else:
        badge_y1 = y1
        badge_y2 = y1 + badge_h

    # Semi-transparent dark background for text legibility against any background
    if badge_y2 > badge_y1 and badge_x2 > badge_x1:
        bg_sub = frame[badge_y1:badge_y2, badge_x1:badge_x2]
        frame[badge_y1:badge_y2, badge_x1:badge_x2] = (bg_sub * 0.25).astype(frame.dtype)
        # Thin border around badge
        cv2.rectangle(frame, (badge_x1, badge_y1), (badge_x2, badge_y2), (60, 60, 60), 1)

    text_x = badge_x1 + badge_pad_x
    text_y = badge_y2 - badge_pad_y - baseline // 2

    # Draw text with shadow for maximum contrast
    cv2.putText(frame, clipped_label, (text_x + 1, text_y + 1), _FONT, _FS_LABEL, _C_SHADOW, _FT + 1, _LT)
    cv2.putText(frame, clipped_label, (text_x, text_y), _FONT, _FS_LABEL, _C_LABEL, _FT, _LT)
