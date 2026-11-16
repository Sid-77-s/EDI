"""
perception/opportunity.py — Visual Opportunity Detector.

Finds suitable regions in the scene to render future visualisations.
Uses action-driven logic to anchor visualizations to gestures and body positions.

Milestone 2B.1 — Stability:
  - Hysteresis gate, candidate persistence, active-hand locking,
    type-transition bypass, deadband, EMA smoothing.

Milestone 2B.1.1 — Explicit Spatial Anchor Lock:
  - Pointing ray (index_mcp → index_tip) creates a locked POINTING_TARGET anchor.
  - Palm dwell (HAND_DWELL_MS) creates a locked ON_HAND anchor.
  - Smooth hand-to-hand transfer animation (HAND_TRANSFER_DURATION_MS).
  - Fallback candidates disabled while anchor_locked=True.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from src.perception.models import (
    Gesture,
    HandData,
    SceneState,
    VisualOpportunity,
    VisualOpportunityState,
    VisualOpportunityType,
)

# ---------------------------------------------------------------------------
# Stability constants
# ---------------------------------------------------------------------------

SWITCH_MARGIN           = 0.12
SWITCH_CONFIRM_FRAMES   = 4
EMA_ALPHA               = 0.22
DEADBAND_DIST           = 0.04

HAND_LOSS_HOLD_MS       = 400
FALLBACK_CONFIRM_FRAMES = 5
FALLBACK_SWITCH_MARGIN  = 0.15

HAND_DWELL_MS           = 350
HAND_TRANSFER_DURATION_MS = 250.0
PALM_VELOCITY_THRESHOLD = 0.015  # normalised units per second
POINTING_RAY_DISTANCE   = 0.18   # default anchor distance along ray (empty air)

ANCHOR_PRIORITY = (
    VisualOpportunityType.POINTING_TARGET,
    VisualOpportunityType.ON_HAND,
    VisualOpportunityType.IN_FRONT_OF_PERSON,
    VisualOpportunityType.NEAR_PERSON,
    VisualOpportunityType.BACKGROUND,
    VisualOpportunityType.EMPTY_SPACE,
)


class TransitionState(str, Enum):
    NONE = "NONE"
    LEFT_TO_RIGHT = "LEFT_TO_RIGHT"
    RIGHT_TO_LEFT = "RIGHT_TO_LEFT"


@dataclass
class AnchorState:
    """Internal anchor lock state — never exposed outside the detector."""
    active_anchor_type: Optional[VisualOpportunityType] = None
    active_anchor_position: Tuple[float, float] = (0.5, 0.5)
    active_anchor_bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    active_anchor_hand: Optional[str] = None   # "LEFT" | "RIGHT"
    anchor_locked: bool = False
    anchor_confidence: float = 0.0
    anchor_last_seen: float = 0.0
    anchor_source: str = ""
    anchor_label: str = ""
    anchor_reason: str = ""
    transition_state: TransitionState = TransitionState.NONE
    transition_start_time: float = 0.0
    transition_start_bbox: Optional[Tuple[float, float, float, float]] = None
    transition_target_bbox: Optional[Tuple[float, float, float, float]] = None
    transition_target_hand: Optional[str] = None


@dataclass
class CandidateRegion:
    label: str
    type: VisualOpportunityType
    center_x: float
    center_y: float
    width: float
    height: float
    bbox: Tuple[float, float, float, float]
    score: float = 0.0
    reason: str = ""
    source: str = ""


@dataclass
class StabilityDebugInfo:
    """Snapshot of internal stability state for HUD display."""
    active_hand: str
    candidate_hand: str
    candidate_streak: int
    switch_count: int
    current_score: float
    challenger_score: float
    opportunity_mode: str
    fallback_candidate: str
    fallback_streak: int
    time_since_hand_loss_ms: float
    switch_margin: float = SWITCH_MARGIN
    target_hand: str = "NONE"
    anchor_locked: bool = False
    anchor_type: str = "NONE"
    anchor_confidence: float = 0.0
    transfer_state: str = "NONE"
    transfer_pct: float = 0.0
    anchor_mode: str = "ACTIVE"
    pointing_active: bool = False
    index_mcp: Tuple[float, float] = (0.0, 0.0)
    index_tip: Tuple[float, float] = (0.0, 0.0)
    pointing_target: Tuple[float, float] = (0.0, 0.0)


@dataclass
class _PalmTrackState:
    last_cx: float = 0.0
    last_cy: float = 0.0
    last_timestamp: float = 0.0
    dwell_start: Optional[float] = None
    locked: bool = False


class VisualOpportunityDetector:
    """Detects where a visualization can be rendered safely."""

    def __init__(
        self,
        min_score_threshold: float = -10.0,
        stability_threshold: float = 0.05,
    ):
        self._min_score_threshold = min_score_threshold
        self._stability_threshold = stability_threshold

        self._fallback_anchors = [
            ("LEFT", 0.15, 0.50),
            ("RIGHT", 0.85, 0.50),
            ("UPPER_LEFT", 0.15, 0.20),
            ("UPPER_RIGHT", 0.85, 0.20),
            ("CENTER_LEFT", 0.30, 0.50),
            ("CENTER_RIGHT", 0.70, 0.50),
            ("LOWER_LEFT", 0.15, 0.80),
            ("LOWER_RIGHT", 0.85, 0.80),
        ]

        self._sizes = {
            VisualOpportunityType.ON_HAND: (0.12, 0.12),
            VisualOpportunityType.POINTING_TARGET: (0.20, 0.17),
            VisualOpportunityType.IN_FRONT_OF_PERSON: (0.22, 0.20),
            VisualOpportunityType.NEAR_PERSON: (0.22, 0.20),
            VisualOpportunityType.BACKGROUND: (0.30, 0.27),
            VisualOpportunityType.EMPTY_SPACE: (0.30, 0.27),
        }

        self._base_scores = {
            VisualOpportunityType.ON_HAND: 50.0,
            VisualOpportunityType.POINTING_TARGET: 40.0,
            VisualOpportunityType.IN_FRONT_OF_PERSON: 30.0,
            VisualOpportunityType.NEAR_PERSON: 20.0,
            VisualOpportunityType.BACKGROUND: 10.0,
            VisualOpportunityType.EMPTY_SPACE: 0.0,
        }

        # Active selection state (fallback path)
        self._active_type: Optional[VisualOpportunityType] = None
        self._active_source: Optional[str] = None
        self._active_hand: Optional[str] = None
        self._active_label: Optional[str] = None

        self._candidate_type: Optional[VisualOpportunityType] = None
        self._candidate_source: Optional[str] = None
        self._candidate_label: Optional[str] = None
        self._candidate_streak: int = 0

        self._smooth_cx: float = 0.5
        self._smooth_cy: float = 0.5
        self._smooth_w: float = 0.12
        self._smooth_h: float = 0.12
        self._smoothing_active: bool = False

        self._switch_count: int = 0
        self._current_score: float = 0.0
        self._challenger_score: float = 0.0

        self._opportunity_mode: str = "ACTIVE"
        self._last_valid_opportunity: Optional[VisualOpportunity] = None
        self._last_valid_timestamp: float = 0.0

        self._fallback_candidate_type: Optional[VisualOpportunityType] = None
        self._fallback_candidate_source: Optional[str] = None
        self._fallback_candidate_label: Optional[str] = None
        self._fallback_streak: int = 0

        # Anchor lock state (Milestone 2B.1.1)
        self._anchor = AnchorState()
        self._palm_track: Dict[str, _PalmTrackState] = {}
        self._target_hand: Optional[str] = None

        self._pointing_debug_active = False
        self._pointing_mcp: Tuple[float, float] = (0.0, 0.0)
        self._pointing_tip: Tuple[float, float] = (0.0, 0.0)
        self._pointing_target: Tuple[float, float] = (0.0, 0.0)

        self._debug_info = StabilityDebugInfo(
            active_hand="NONE",
            candidate_hand="NONE",
            candidate_streak=0,
            switch_count=0,
            current_score=0.0,
            challenger_score=0.0,
            opportunity_mode="ACTIVE",
            fallback_candidate="NONE",
            fallback_streak=0,
            time_since_hand_loss_ms=0.0,
        )
        self._active_concept_label: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def debug_info(self) -> StabilityDebugInfo:
        return self._debug_info

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _rect_overlap_area(
        self,
        r1: Tuple[float, float, float, float],
        r2: Tuple[float, float, float, float],
    ) -> float:
        x_left = max(r1[0], r2[0])
        y_top = max(r1[1], r2[1])
        x_right = min(r1[2], r2[2])
        y_bottom = min(r1[3], r2[3])
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        return (x_right - x_left) * (y_bottom - y_top)

    @staticmethod
    def _distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

    @staticmethod
    def _ease_in_out(t: float) -> float:
        t = max(0.0, min(1.0, t))
        if t < 0.5:
            return 2.0 * t * t
        return 1.0 - (-2.0 * t + 2.0) ** 2 / 2.0

    @staticmethod
    def _bbox_from_center(cx: float, cy: float, w: float, h: float) -> Tuple[float, float, float, float]:
        return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

    @staticmethod
    def _bbox_center_size(bbox: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        return ((x1 + x2) / 2, (y1 + y2) / 2, w, h)

    def _make_candidate(
        self,
        label: str,
        typ: VisualOpportunityType,
        cx: float,
        cy: float,
        source: str,
        reason: str,
    ) -> CandidateRegion:
        w, h = self._sizes[typ]
        bbox = self._bbox_from_center(cx, cy, w, h)
        return CandidateRegion(
            label=label,
            type=typ,
            center_x=cx,
            center_y=cy,
            width=w,
            height=h,
            bbox=bbox,
            source=source,
            reason=reason,
        )

    @staticmethod
    def _hand_id(source: str) -> str:
        if source == "left_hand":
            return "LEFT"
        if source == "right_hand":
            return "RIGHT"
        return "NONE"

    @staticmethod
    def _hand_str(hand: HandData) -> str:
        return "left_hand" if hand.is_left else "right_hand"

    @staticmethod
    def _palm_center(hand: HandData) -> Optional[Tuple[float, float]]:
        palm_lms = [
            hand.landmarks[i]
            for i in (0, 5, 9, 13, 17)
            if i < len(hand.landmarks)
        ]
        if not palm_lms:
            return None
        cx = sum(lm.x for lm in palm_lms) / len(palm_lms)
        cy = sum(lm.y for lm in palm_lms) / len(palm_lms)
        return (cx, cy)

    @staticmethod
    def _ray_rect_intersection(
        ox: float, oy: float, dx: float, dy: float,
        rect: Tuple[float, float, float, float],
    ) -> Optional[Tuple[float, float, float]]:
        """Return (hit_x, hit_y, t) for first forward ray hit on rect, or None."""
        x_min, y_min, x_max, y_max = rect
        t_min = float("inf")
        hit: Optional[Tuple[float, float, float]] = None

        if abs(dx) > 1e-9:
            for x_edge in (x_min, x_max):
                t = (x_edge - ox) / dx
                if t >= 0:
                    y = oy + dy * t
                    if y_min <= y <= y_max and t < t_min:
                        t_min = t
                        hit = (x_edge, y, t)
        if abs(dy) > 1e-9:
            for y_edge in (y_min, y_max):
                t = (y_edge - oy) / dy
                if t >= 0:
                    x = ox + dx * t
                    if x_min <= x <= x_max and t < t_min:
                        t_min = t
                        hit = (x, y_edge, t)

        return hit

    @staticmethod
    def _is_hand_pointing(hand: HandData) -> bool:
        wrist = hand.wrist
        index_tip = hand.index_tip
        index_mcp = hand.index_mcp
        middle_tip = hand.middle_tip
        ring_tip = hand.ring_tip
        pinky_tip = hand.pinky_tip
        if not all([wrist, index_tip, index_mcp]):
            return False
        idx_ext = ((wrist.x - index_tip.x) ** 2 + (wrist.y - index_tip.y) ** 2) ** 0.5
        idx_base = ((wrist.x - index_mcp.x) ** 2 + (wrist.y - index_mcp.y) ** 2) ** 0.5
        if not all([middle_tip, ring_tip, pinky_tip]):
            return idx_ext > idx_base * 1.3
        mid_dist = ((wrist.x - middle_tip.x) ** 2 + (wrist.y - middle_tip.y) ** 2) ** 0.5
        ring_dist = ((wrist.x - ring_tip.x) ** 2 + (wrist.y - ring_tip.y) ** 2) ** 0.5
        pinky_dist = ((wrist.x - pinky_tip.x) ** 2 + (wrist.y - pinky_tip.y) ** 2) ** 0.5
        if idx_ext > idx_base * 1.3:
            if max(mid_dist, ring_dist, pinky_dist) < idx_ext * 0.75:
                return True
        return False

    def _compute_pointing_anchor(
        self,
        hand: HandData,
        object_bboxes: List[Tuple[float, float, float, float]],
    ) -> Optional[Tuple[float, float, str, str, str]]:
        """Return (cx, cy, label, source, reason) for a pointing ray anchor."""
        mcp = hand.index_mcp
        tip = hand.index_tip
        if not mcp or not tip:
            return None

        dx = tip.x - mcp.x
        dy = tip.y - mcp.y
        mag = (dx ** 2 + dy ** 2) ** 0.5
        if mag < 1e-6:
            return None
        dx /= mag
        dy /= mag

        ox, oy = mcp.x, mcp.y
        hand_str = self._hand_str(hand)
        min_t = mag * 0.35

        w_box, h_box = self._sizes[VisualOpportunityType.POINTING_TARGET]

        best_hit: Optional[Tuple[float, float, float]] = None
        for obbox in object_bboxes:
            hit = self._ray_rect_intersection(ox, oy, dx, dy, obbox)
            if hit is None:
                continue
            hx, hy, t_val = hit
            if t_val < min_t:
                continue
            if best_hit is None or t_val < best_hit[2]:
                best_hit = (hx, hy, t_val)

        if best_hit is not None:
            hx, hy, t_val = best_hit
            label = f"RAY_{t_val:.2f}"
            self._pointing_debug_active = True
            self._pointing_mcp = (mcp.x, mcp.y)
            self._pointing_tip = (tip.x, tip.y)
            self._pointing_target = (hx, hy)
            return (
                hx, hy, label, hand_str,
                f"Pointing anchor at object via {hand_str}",
            )

        # Empty region: extend MCP → tip direction to place box close to fingertip
        t_use = mag + POINTING_RAY_DISTANCE
        cx = ox + dx * t_use
        cy = oy + dy * t_use

        # Clamp inside frame boundaries to keep the box visible
        cx = max(w_box / 2.0, min(1.0 - w_box / 2.0, cx))
        cy = max(h_box / 2.0, min(1.0 - h_box / 2.0, cy))

        self._pointing_debug_active = True
        self._pointing_mcp = (mcp.x, mcp.y)
        self._pointing_tip = (tip.x, tip.y)
        self._pointing_target = (cx, cy)
        return (
            cx, cy, f"RAY_{t_use:.2f}", hand_str,
            f"Target along {hand_str} pointing ray",
        )

    def _update_palm_dwell(
        self,
        hand_key: str,
        palm: Tuple[float, float],
        timestamp: float,
        is_open: bool,
    ) -> bool:
        """Update palm tracking; return True when dwell lock is satisfied."""
        track = self._palm_track.setdefault(hand_key, _PalmTrackState())
        dt = timestamp - track.last_timestamp if track.last_timestamp > 0 else 0.0

        velocity = 0.0
        if dt > 0:
            velocity = self._distance(palm, (track.last_cx, track.last_cy)) / dt

        track.last_cx, track.last_cy = palm
        track.last_timestamp = timestamp

        if not is_open:
            track.dwell_start = None
            track.locked = False
            return False

        if velocity <= PALM_VELOCITY_THRESHOLD:
            if track.dwell_start is None:
                track.dwell_start = timestamp
            elif (timestamp - track.dwell_start) >= (HAND_DWELL_MS / 1000.0) - 1e-6:
                track.locked = True
                return True
        else:
            track.dwell_start = None
            if not track.locked:
                return False

        return track.locked

    # ------------------------------------------------------------------
    # Anchor lock path
    # ------------------------------------------------------------------

    def _try_pointing_lock(
        self,
        state: SceneState,
        object_bboxes: List[Tuple[float, float, float, float]],
    ) -> Optional[Tuple[float, float, str, str, str, str]]:
        """Return (cx, cy, label, source, reason, hand_id) if pointing is active."""
        if not state.human_perception or not state.human_perception.hands:
            return None

        left_pointing = bool(
            state.activity and state.activity.left_hand_gesture == Gesture.POINTING
        )
        right_pointing = bool(
            state.activity and state.activity.right_hand_gesture == Gesture.POINTING
        )

        hands = list(state.human_perception.hands)
        if self._anchor.active_anchor_hand:
            hands.sort(
                key=lambda h: 0 if ("LEFT" if h.is_left else "RIGHT") == self._anchor.active_anchor_hand else 1
            )

        for hand in hands:
            is_pointing = (left_pointing if hand.is_left else right_pointing) or self._is_hand_pointing(hand)
            if not is_pointing:
                continue
            anchor = self._compute_pointing_anchor(hand, object_bboxes)
            if anchor is None:
                continue
            cx, cy, label, source, reason = anchor
            return (cx, cy, label, source, reason, "LEFT" if hand.is_left else "RIGHT")

        return None

    def _try_palm_anchor(
        self,
        state: SceneState,
    ) -> Optional[Tuple[float, float, str, str, str, str, bool]]:
        """Return palm anchor info; last element is dwell_locked."""
        if not state.human_perception or not state.human_perception.hands:
            return None

        left_pointing = bool(
            state.activity and state.activity.left_hand_gesture == Gesture.POINTING
        )
        right_pointing = bool(
            state.activity and state.activity.right_hand_gesture == Gesture.POINTING
        )

        best: Optional[Tuple[float, float, str, str, str, str, bool]] = None

        hands = list(state.human_perception.hands)
        if self._anchor.active_anchor_hand:
            hands.sort(
                key=lambda h: 0 if ("LEFT" if h.is_left else "RIGHT") == self._anchor.active_anchor_hand else 1
            )

        for hand in hands:
            is_pointing = (left_pointing if hand.is_left else right_pointing) or self._is_hand_pointing(hand)
            if is_pointing:
                continue
            palm = self._palm_center(hand)
            if palm is None:
                continue
            hand_key = "LEFT" if hand.is_left else "RIGHT"
            hand_str = self._hand_str(hand)
            dwell_locked = self._update_palm_dwell(hand_key, palm, state.timestamp, is_open=True)
            result = (
                palm[0], palm[1], "PALM_CENTER", hand_str,
                f"Attached to {hand_str}", hand_key, dwell_locked,
            )
            if best is None:
                best = result
            elif self._anchor.active_anchor_hand == hand_key:
                best = result
                break

        return best

    def _is_active_anchor_hand_present(self, state: SceneState) -> bool:
        """Return True if the locked anchor hand is still visible in the frame."""
        if not state.human_perception or not self._anchor.active_anchor_hand:
            return False
        for hand in state.human_perception.hands:
            hand_key = "LEFT" if hand.is_left else "RIGHT"
            if hand_key == self._anchor.active_anchor_hand:
                return True
        return False

    def _generate_on_hand_candidates(self, state: SceneState) -> List[CandidateRegion]:
        """Build ON_HAND candidates only — used for locked multi-hand selection."""
        candidates: List[CandidateRegion] = []
        if not state.human_perception:
            return candidates

        left_pointing = bool(
            state.activity and state.activity.left_hand_gesture == Gesture.POINTING
        )
        right_pointing = bool(
            state.activity and state.activity.right_hand_gesture == Gesture.POINTING
        )

        for hand in state.human_perception.hands:
            is_pointing = left_pointing if hand.is_left else right_pointing
            if is_pointing:
                continue
            palm = self._palm_center(hand)
            if palm is None:
                continue
            hand_str = self._hand_str(hand)
            candidates.append(self._make_candidate(
                "PALM_CENTER", VisualOpportunityType.ON_HAND,
                palm[0], palm[1], hand_str, f"Attached to {hand_str}",
            ))

        for cand in candidates:
            score = self._base_scores[cand.type]
            if (
                cand.bbox[0] < 0.0 or cand.bbox[1] < 0.0
                or cand.bbox[2] > 1.0 or cand.bbox[3] > 1.0
            ):
                score -= 10.0
                if (
                    cand.bbox[2] < 0 or cand.bbox[0] > 1.0
                    or cand.bbox[3] < 0 or cand.bbox[1] > 1.0
                ):
                    score -= 100.0
            cand.score = score

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates

    def _resolve_transfer_target(
        self, state: SceneState
    ) -> Optional[CandidateRegion]:
        """Find the ON_HAND candidate matching the in-progress transfer target hand."""
        target_hand = self._anchor.transition_target_hand or self._anchor.active_anchor_hand
        if not target_hand:
            return None
        for cand in self._generate_on_hand_candidates(state):
            if self._hand_id(cand.source) == target_hand:
                return cand
        return None

    def _emit_on_hand_opportunity(
        self,
        cx: float,
        cy: float,
        source: str,
        reason: str,
        label: str,
        hand_id: str,
        locked: bool,
        state: SceneState,
        start_time: float,
        concept_label: Optional[str] = None,
    ) -> VisualOpportunity:
        self._opportunity_mode = "ACTIVE"
        self._active_type = VisualOpportunityType.ON_HAND
        self._active_source = source
        self._active_label = label
        self._active_hand = hand_id

        opp = self._build_anchor_opportunity(
            VisualOpportunityType.ON_HAND,
            cx, cy, source, reason, label, hand_id,
            locked=locked,
            timestamp=state.timestamp,
            detector_ms=(time.monotonic() - start_time) * 1000.0,
            concept_label=concept_label,
        )
        self._last_valid_opportunity = opp
        self._last_valid_timestamp = state.timestamp
        self._update_debug(state.timestamp)
        return opp

    def _apply_ema(
        self,
        target_cx: float, target_cy: float,
        target_w: float, target_h: float,
    ) -> Tuple[float, float, float, float]:
        if self._smoothing_active:
            dist = self._distance((target_cx, target_cy), (self._smooth_cx, self._smooth_cy))
            if dist < DEADBAND_DIST:
                return self._smooth_cx, self._smooth_cy, self._smooth_w, self._smooth_h
            cx = EMA_ALPHA * target_cx + (1.0 - EMA_ALPHA) * self._smooth_cx
            cy = EMA_ALPHA * target_cy + (1.0 - EMA_ALPHA) * self._smooth_cy
            w  = EMA_ALPHA * target_w  + (1.0 - EMA_ALPHA) * self._smooth_w
            h  = EMA_ALPHA * target_h  + (1.0 - EMA_ALPHA) * self._smooth_h
        else:
            cx, cy, w, h = target_cx, target_cy, target_w, target_h
            self._smoothing_active = True

        self._smooth_cx, self._smooth_cy = cx, cy
        self._smooth_w, self._smooth_h = w, h
        return cx, cy, w, h

    def _render_transition_bbox(self, timestamp: float) -> Optional[Tuple[float, float, float, float]]:
        if self._anchor.transition_state == TransitionState.NONE:
            return None
        if self._anchor.transition_start_bbox is None or self._anchor.transition_target_bbox is None:
            return None

        elapsed_ms = (timestamp - self._anchor.transition_start_time) * 1000.0
        progress = elapsed_ms / HAND_TRANSFER_DURATION_MS
        if progress >= 1.0:
            self._anchor.transition_state = TransitionState.NONE
            self._anchor.transition_start_bbox = None
            self._anchor.transition_target_bbox = None
            self._anchor.active_anchor_hand = self._anchor.transition_target_hand
            self._anchor.transition_target_hand = None
            return None

        t = self._ease_in_out(progress)
        s_cx, s_cy, s_w, s_h = self._bbox_center_size(self._anchor.transition_start_bbox)
        e_cx, e_cy, e_w, e_h = self._bbox_center_size(self._anchor.transition_target_bbox)
        cx = s_cx + (e_cx - s_cx) * t
        cy = s_cy + (e_cy - s_cy) * t
        w  = s_w  + (e_w  - s_w)  * t
        h  = s_h  + (e_h  - s_h)  * t
        return self._bbox_from_center(cx, cy, w, h)

    def _maybe_start_hand_transfer(
        self,
        from_hand: str,
        to_hand: str,
        target_bbox: Tuple[float, float, float, float],
        timestamp: float,
        current_rendered_bbox: Tuple[float, float, float, float],
    ) -> None:
        if from_hand == to_hand:
            return
        if from_hand not in ("LEFT", "RIGHT") or to_hand not in ("LEFT", "RIGHT"):
            return

        # If already transitioning, restart from current rendered bbox if direction reversed
        if self._anchor.transition_state != TransitionState.NONE:
            current_from = (
                "LEFT" if self._anchor.transition_state == TransitionState.RIGHT_TO_LEFT else "RIGHT"
            )
            if to_hand == current_from:
                # Reversal — restart from current rendered position
                self._anchor.transition_start_bbox = current_rendered_bbox
                self._anchor.transition_target_bbox = target_bbox
                self._anchor.transition_start_time = timestamp
                self._anchor.transition_target_hand = to_hand
                self._anchor.transition_state = (
                    TransitionState.LEFT_TO_RIGHT if from_hand == "LEFT" else TransitionState.RIGHT_TO_LEFT
                )
                return

        self._anchor.transition_state = (
            TransitionState.LEFT_TO_RIGHT if from_hand == "LEFT" else TransitionState.RIGHT_TO_LEFT
        )
        self._anchor.transition_start_bbox = current_rendered_bbox
        self._anchor.transition_target_bbox = target_bbox
        self._anchor.transition_start_time = timestamp
        self._anchor.transition_target_hand = to_hand

    def _build_anchor_opportunity(
        self,
        anchor_type: VisualOpportunityType,
        cx: float, cy: float,
        source: str, reason: str, label: str,
        hand_id: str,
        locked: bool,
        timestamp: float,
        detector_ms: float,
        score: float = 1.0,
        concept_label: Optional[str] = None,
    ) -> VisualOpportunity:
        w, h = self._sizes[anchor_type]

        # During hand transfer use interpolation only (no EMA)
        transition_bbox = self._render_transition_bbox(timestamp)
        if transition_bbox is not None:
            final_bbox = transition_bbox
            fcx, fcy, fw, fh = self._bbox_center_size(final_bbox)
            self._smooth_cx, self._smooth_cy = fcx, fcy
            self._smooth_w, self._smooth_h = fw, fh
            self._smoothing_active = True
        else:
            fcx, fcy, fw, fh = self._apply_ema(cx, cy, w, h)
            final_bbox = self._bbox_from_center(fcx, fcy, fw, fh)

        self._anchor.active_anchor_type = anchor_type
        self._anchor.active_anchor_position = (fcx, fcy)
        self._anchor.active_anchor_bbox = final_bbox
        if self._anchor.transition_state == TransitionState.NONE:
            self._anchor.active_anchor_hand = hand_id
        self._anchor.anchor_locked = locked
        self._anchor.anchor_confidence = score
        self._anchor.anchor_last_seen = timestamp
        self._anchor.anchor_source = source
        self._anchor.anchor_label = label
        self._anchor.anchor_reason = reason

        normalized_score = min(1.0, max(0.0, self._base_scores.get(anchor_type, 0.0) / 60.0))

        effective_concept = concept_label or getattr(self, "_active_concept_label", None)

        return VisualOpportunity(
            state=VisualOpportunityState.AVAILABLE,
            type=anchor_type,
            available=True,
            bbox=final_bbox,
            center=(fcx, fcy),
            width=fw,
            height=fh,
            score=normalized_score,
            source=source,
            reason=reason,
            region_label=label,
            detector_ms=detector_ms,
            anchor_locked=locked,
            anchor_hand=hand_id,
            anchor_type=anchor_type,
            anchor_confidence=score,
            anchor_timestamp=timestamp,
            concept_label=effective_concept,
        )

    # ------------------------------------------------------------------
    # Fallback candidate path (unchanged stability logic)
    # ------------------------------------------------------------------

    def _is_type_transition(self, best: CandidateRegion, state: SceneState) -> bool:
        if self._active_type is None:
            return False
        if best.type == self._active_type:
            return False

        left_pointing = bool(
            state.activity and state.activity.left_hand_gesture == Gesture.POINTING
        )
        right_pointing = bool(
            state.activity and state.activity.right_hand_gesture == Gesture.POINTING
        )
        any_pointing = left_pointing or right_pointing

        if (
            self._active_type == VisualOpportunityType.ON_HAND
            and best.type == VisualOpportunityType.POINTING_TARGET
        ):
            return any_pointing

        if (
            self._active_type == VisualOpportunityType.POINTING_TARGET
            and best.type == VisualOpportunityType.ON_HAND
        ):
            return not any_pointing

        return False

    def _update_debug(self, state_ts: float) -> None:
        candidate_hand = (
            self._hand_id(self._candidate_source) if self._candidate_source else "NONE"
        )
        time_since_hand_loss = (
            (state_ts - self._last_valid_timestamp) * 1000.0
            if self._opportunity_mode in ("HAND_LOSS_HOLD", "FALLBACK_CONFIRMATION")
            and self._last_valid_timestamp > 0
            else 0.0
        )

        transfer_pct = 0.0
        if self._anchor.transition_state != TransitionState.NONE and self._anchor.transition_start_time > 0:
            elapsed_ms = (state_ts - self._anchor.transition_start_time) * 1000.0
            transfer_pct = min(100.0, max(0.0, elapsed_ms / HAND_TRANSFER_DURATION_MS * 100.0))

        anchor_mode = self._opportunity_mode
        if self._anchor.anchor_locked:
            anchor_mode = "ANCHOR_LOCKED"
        elif self._anchor.transition_state != TransitionState.NONE:
            anchor_mode = "HAND_TRANSFER"

        self._debug_info = StabilityDebugInfo(
            active_hand=self._active_hand or self._anchor.active_anchor_hand or "NONE",
            candidate_hand=candidate_hand,
            candidate_streak=self._candidate_streak,
            switch_count=self._switch_count,
            current_score=self._current_score,
            challenger_score=self._challenger_score,
            opportunity_mode=self._opportunity_mode,
            fallback_candidate=self._fallback_candidate_source or "NONE",
            fallback_streak=self._fallback_streak,
            time_since_hand_loss_ms=time_since_hand_loss,
            target_hand=self._target_hand or "NONE",
            anchor_locked=self._anchor.anchor_locked,
            anchor_type=(
                self._anchor.active_anchor_type.value.upper()
                if self._anchor.active_anchor_type else "NONE"
            ),
            anchor_confidence=self._anchor.anchor_confidence,
            transfer_state=self._anchor.transition_state.value,
            transfer_pct=transfer_pct,
            anchor_mode=anchor_mode,
            pointing_active=self._pointing_debug_active,
            index_mcp=self._pointing_mcp,
            index_tip=self._pointing_tip,
            pointing_target=self._pointing_target,
        )

    def _make_unavailable(self, detector_ms: float, concept_label: Optional[str] = None) -> VisualOpportunity:
        effective_concept = concept_label or getattr(self, "_active_concept_label", None)
        return VisualOpportunity(
            state=VisualOpportunityState.NO_SUITABLE_REGION,
            type=VisualOpportunityType.EMPTY_SPACE,
            available=False,
            bbox=(0.0, 0.0, 0.0, 0.0),
            center=(0.0, 0.0),
            width=0.0,
            height=0.0,
            score=0.0,
            source="none",
            reason="All regions penalized or out of bounds",
            region_label="NONE",
            detector_ms=detector_ms,
            anchor_locked=False,
            anchor_hand=None,
            anchor_type=None,
            anchor_confidence=0.0,
            anchor_timestamp=0.0,
            concept_label=effective_concept,
        )

    def _generate_fallback_candidates(
        self,
        state: SceneState,
        person_bbox: Optional[Tuple[float, float, float, float]],
        face_bbox: Optional[Tuple[float, float, float, float]],
        hand_bboxes: List[Tuple[float, float, float, float]],
        object_bboxes: List[Tuple[float, float, float, float]],
        include_hand_candidates: bool,
    ) -> List[CandidateRegion]:
        candidates: List[CandidateRegion] = []

        left_is_pointing = bool(
            state.activity and state.activity.left_hand_gesture == Gesture.POINTING
        )
        right_is_pointing = bool(
            state.activity and state.activity.right_hand_gesture == Gesture.POINTING
        )

        if include_hand_candidates and state.human_perception:
            for hand in state.human_perception.hands:
                is_pointing = left_is_pointing if hand.is_left else right_is_pointing
                hand_str = self._hand_str(hand)
                palm = self._palm_center(hand)
                if palm is None:
                    continue

                if is_pointing and hand.index_mcp and hand.index_tip:
                    anchor = self._compute_pointing_anchor(hand, object_bboxes)
                    if anchor:
                        cx, cy, label, source, reason = anchor
                        candidates.append(
                            self._make_candidate(label, VisualOpportunityType.POINTING_TARGET,
                                                 cx, cy, source, reason)
                        )
                else:
                    candidates.append(self._make_candidate(
                        "PALM_CENTER", VisualOpportunityType.ON_HAND,
                        palm[0], palm[1], hand_str, f"Attached to {hand_str}"))
                    candidates.append(self._make_candidate(
                        "ABOVE_PALM", VisualOpportunityType.ON_HAND,
                        palm[0], palm[1] - 0.1, hand_str, f"Slightly above {hand_str}"))

        if person_bbox:
            pcx = (person_bbox[0] + person_bbox[2]) / 2
            pcy = (person_bbox[1] + person_bbox[3]) / 2
            chest_y = pcy - 0.1
            if face_bbox:
                chest_y = face_bbox[3] + 0.15

            candidates.append(self._make_candidate(
                "FRONT_LEFT", VisualOpportunityType.IN_FRONT_OF_PERSON,
                pcx - 0.15, chest_y, "person", "In front (left offset)"))
            candidates.append(self._make_candidate(
                "FRONT_RIGHT", VisualOpportunityType.IN_FRONT_OF_PERSON,
                pcx + 0.15, chest_y, "person", "In front (right offset)"))
            candidates.append(self._make_candidate(
                "FRONT_CENTER", VisualOpportunityType.IN_FRONT_OF_PERSON,
                pcx, chest_y, "person", "In front (center)"))
            candidates.append(self._make_candidate(
                "NEAR_LEFT", VisualOpportunityType.NEAR_PERSON,
                person_bbox[0] - 0.15, pcy, "person", "Beside person (left)"))
            candidates.append(self._make_candidate(
                "NEAR_RIGHT", VisualOpportunityType.NEAR_PERSON,
                person_bbox[2] + 0.15, pcy, "person", "Beside person (right)"))

        for label, cx, cy in self._fallback_anchors:
            typ = (
                VisualOpportunityType.BACKGROUND
                if person_bbox or object_bboxes
                else VisualOpportunityType.EMPTY_SPACE
            )
            source = "background" if typ == VisualOpportunityType.BACKGROUND else "empty_scene"
            candidates.append(self._make_candidate(label, typ, cx, cy, source, "Fallback region"))

        for cand in candidates:
            area = cand.width * cand.height
            score = self._base_scores[cand.type]

            if (
                cand.bbox[0] < 0.0 or cand.bbox[1] < 0.0
                or cand.bbox[2] > 1.0 or cand.bbox[3] > 1.0
            ):
                score -= 10.0
                if (
                    cand.bbox[2] < 0 or cand.bbox[0] > 1.0
                    or cand.bbox[3] < 0 or cand.bbox[1] > 1.0
                ):
                    score -= 100.0

            person_overlap = (
                self._rect_overlap_area(cand.bbox, person_bbox) if person_bbox else 0.0
            )
            face_overlap = (
                self._rect_overlap_area(cand.bbox, face_bbox) if face_bbox else 0.0
            )
            obj_overlap = sum(self._rect_overlap_area(cand.bbox, ob) for ob in object_bboxes)
            hand_overlap = sum(self._rect_overlap_area(cand.bbox, hb) for hb in hand_bboxes)

            if cand.type == VisualOpportunityType.ON_HAND:
                if face_overlap > 0:
                    score -= (face_overlap / area) * 20.0
                if obj_overlap > 0:
                    score -= (obj_overlap / area) * 10.0

            elif cand.type == VisualOpportunityType.POINTING_TARGET:
                if hand_overlap > 0:
                    score -= (hand_overlap / area) * 15.0
                if face_overlap > 0:
                    score -= (face_overlap / area) * 20.0
                for obbox in object_bboxes:
                    ocx = (obbox[0] + obbox[2]) / 2
                    ocy = (obbox[1] + obbox[3]) / 2
                    dist = self._distance((cand.center_x, cand.center_y), (ocx, ocy))
                    if dist < 0.3:
                        score += (0.3 - dist) * 10.0

            elif cand.type == VisualOpportunityType.IN_FRONT_OF_PERSON:
                if face_overlap > 0:
                    score -= (face_overlap / area) * 30.0
                if hand_overlap > 0:
                    score -= (hand_overlap / area) * 20.0
                score -= (person_overlap / area) * 5.0
                if obj_overlap > 0:
                    score -= (obj_overlap / area) * 10.0

            elif cand.type == VisualOpportunityType.NEAR_PERSON:
                if person_overlap > 0:
                    score -= (person_overlap / area) * 20.0
                if obj_overlap > 0:
                    score -= (obj_overlap / area) * 10.0

            elif cand.type in (
                VisualOpportunityType.BACKGROUND, VisualOpportunityType.EMPTY_SPACE
            ):
                if person_overlap > 0:
                    score -= (person_overlap / area) * 30.0
                if obj_overlap > 0:
                    score -= (obj_overlap / area) * 20.0
                if hand_overlap > 0:
                    score -= (hand_overlap / area) * 30.0

            cand.score = score

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates

    def _select_from_candidates(
        self,
        candidates: List[CandidateRegion],
        state: SceneState,
    ) -> Optional[CandidateRegion]:
        if not candidates:
            return None

        active_cands = []
        if self._active_source:
            active_cands = [
                c for c in candidates
                if c.source == self._active_source
                and (
                    c.label == self._active_label
                    or self._active_type not in (
                        VisualOpportunityType.IN_FRONT_OF_PERSON,
                        VisualOpportunityType.NEAR_PERSON,
                        VisualOpportunityType.BACKGROUND,
                    )
                )
            ]
            if not active_cands:
                active_cands = [c for c in candidates if c.source == self._active_source]

        current_cand = active_cands[0] if active_cands else None

        is_active_hand = self._active_type in (
            VisualOpportunityType.ON_HAND,
            VisualOpportunityType.POINTING_TARGET,
        )

        if is_active_hand and current_cand is None:
            if self._opportunity_mode == "ACTIVE":
                self._opportunity_mode = "HAND_LOSS_HOLD"
                self._last_valid_timestamp = state.timestamp

            time_since_loss_ms = (state.timestamp - self._last_valid_timestamp) * 1000.0

            if self._opportunity_mode == "HAND_LOSS_HOLD":
                if time_since_loss_ms < HAND_LOSS_HOLD_MS:
                    return None  # signal hold
                self._opportunity_mode = "FALLBACK_CONFIRMATION"
                self._fallback_streak = 0
                self._fallback_candidate_source = None
                self._fallback_candidate_type = None
                self._candidate_streak = 0
        else:
            if self._opportunity_mode != "ACTIVE" and current_cand is not None:
                self._opportunity_mode = "ACTIVE"
                self._fallback_streak = 0
                self._candidate_streak = 0

        best_cand = candidates[0]
        self._challenger_score = best_cand.score
        current_score = current_cand.score if current_cand else 0.0

        is_fallback_active = self._active_type in (
            VisualOpportunityType.IN_FRONT_OF_PERSON,
            VisualOpportunityType.NEAR_PERSON,
            VisualOpportunityType.BACKGROUND,
            VisualOpportunityType.EMPTY_SPACE,
        )

        same_as_active = (
            best_cand.type == self._active_type
            and best_cand.source == self._active_source
            and (
                best_cand.label == self._active_label
                or not is_fallback_active
            )
        )

        margin = FALLBACK_SWITCH_MARGIN if is_fallback_active else SWITCH_MARGIN
        confirm_frames = FALLBACK_CONFIRM_FRAMES if is_fallback_active else SWITCH_CONFIRM_FRAMES

        winner: CandidateRegion

        if same_as_active or self._active_source is None or self._opportunity_mode == "FALLBACK_CONFIRMATION":
            if self._opportunity_mode == "FALLBACK_CONFIRMATION":
                if (
                    best_cand.type == self._fallback_candidate_type
                    and best_cand.label == self._fallback_candidate_label
                ):
                    self._fallback_streak += 1
                else:
                    self._fallback_candidate_type = best_cand.type
                    self._fallback_candidate_source = best_cand.source
                    self._fallback_candidate_label = best_cand.label
                    self._fallback_streak = 1

                if self._fallback_streak >= FALLBACK_CONFIRM_FRAMES:
                    winner = best_cand
                    switched = True
                    self._opportunity_mode = "ACTIVE"
                    self._anchor.anchor_locked = False
                else:
                    return None  # signal hold during fallback confirm
            else:
                self._candidate_type = None
                self._candidate_source = None
                self._candidate_streak = 0
                winner = best_cand
                switched = self._active_source is None
        else:
            if self._is_type_transition(best_cand, state):
                winner = best_cand
                switched = True
                self._candidate_type = None
                self._candidate_source = None
                self._candidate_streak = 0
            else:
                hysteresis_threshold = current_score * (1.0 + margin)
                eligible = best_cand.score > hysteresis_threshold

                if eligible:
                    if (
                        best_cand.type == self._candidate_type
                        and best_cand.label == self._candidate_label
                    ):
                        self._candidate_streak += 1
                    else:
                        self._candidate_type = best_cand.type
                        self._candidate_source = best_cand.source
                        self._candidate_label = best_cand.label
                        self._candidate_streak = 1
                else:
                    self._candidate_streak = 0
                    self._candidate_type = None
                    self._candidate_source = None
                    self._candidate_label = None

                if eligible and self._candidate_streak >= confirm_frames:
                    winner = best_cand
                    switched = True
                    self._candidate_streak = 0
                    self._candidate_type = None
                    self._candidate_source = None
                    self._candidate_label = None
                else:
                    winner = current_cand if current_cand else best_cand
                    switched = False

        # Hand transfer for ON_HAND switches between left/right
        if (
            switched
            and winner.type == VisualOpportunityType.ON_HAND
            and self._active_hand is not None
        ):
            new_hand = self._hand_id(winner.source)
            if new_hand != self._active_hand and new_hand in ("LEFT", "RIGHT"):
                self._target_hand = new_hand
                current_bbox = self._bbox_from_center(
                    self._smooth_cx, self._smooth_cy, self._smooth_w, self._smooth_h
                )
                target_bbox = winner.bbox
                self._maybe_start_hand_transfer(
                    self._active_hand, new_hand, target_bbox, state.timestamp, current_bbox
                )

        if switched or self._active_source is None:
            prev_hand = self._active_hand
            self._active_type = winner.type
            self._active_source = winner.source
            self._active_label = winner.label
            self._active_hand = self._hand_id(winner.source)
            if switched and prev_hand != self._active_hand:
                self._switch_count += 1

        self._current_score = winner.score
        return winner

    # ------------------------------------------------------------------
    # Main detection method
    # ------------------------------------------------------------------

    def detect(
        self,
        state: SceneState,
        prev: Optional[VisualOpportunity] = None,
        concept_label: Optional[str] = None,
    ) -> VisualOpportunity:
        start_time = time.monotonic()
        self._pointing_debug_active = False
        if concept_label is not None:
            self._active_concept_label = concept_label

        # ------------------------------------------------------------------
        # 1. Parse occupied regions
        # ------------------------------------------------------------------
        person_bbox: Optional[Tuple[float, float, float, float]] = None
        face_bbox: Optional[Tuple[float, float, float, float]] = None
        hand_bboxes: List[Tuple[float, float, float, float]] = []
        object_bboxes: List[Tuple[float, float, float, float]] = []

        if state.human_perception and state.human_perception.pose:
            lms = state.human_perception.pose.landmarks
            valid_lms = [lm for lm in lms if lm.visibility is None or lm.visibility > 0.5]
            if valid_lms:
                min_x = min(lm.x for lm in valid_lms)
                max_x = max(lm.x for lm in valid_lms)
                min_y = min(lm.y for lm in valid_lms)
                max_y = max(lm.y for lm in valid_lms)
                person_bbox = (
                    max(0.0, min_x), max(0.0, min_y),
                    min(1.0, max_x), min(1.0, max_y),
                )

            face_lms = [
                lms[i]
                for i in range(min(11, len(lms)))
                if lms[i].visibility is None or lms[i].visibility > 0.5
            ]
            if face_lms:
                f_min_x = min(lm.x for lm in face_lms)
                f_max_x = max(lm.x for lm in face_lms)
                f_min_y = min(lm.y for lm in face_lms)
                f_max_y = max(lm.y for lm in face_lms)
                fh = f_max_y - f_min_y
                fw = f_max_x - f_min_x
                face_bbox = (
                    max(0.0, f_min_x - fw * 0.2),
                    max(0.0, f_min_y - fh * 0.2),
                    min(1.0, f_max_x + fw * 0.2),
                    min(1.0, f_max_y + fh * 0.2),
                )

        if state.human_perception:
            for hand in state.human_perception.hands:
                valid_lms = list(hand.landmarks)
                if valid_lms:
                    min_x = min(lm.x for lm in valid_lms)
                    max_x = max(lm.x for lm in valid_lms)
                    min_y = min(lm.y for lm in valid_lms)
                    max_y = max(lm.y for lm in valid_lms)
                    hand_bboxes.append(
                        (max(0.0, min_x), max(0.0, min_y), min(1.0, max_x), min(1.0, max_y))
                    )

        for obj in state.tracked_objects:
            object_bboxes.append(
                (
                    max(0.0, obj.bbox.x_min),
                    max(0.0, obj.bbox.y_min),
                    min(1.0, obj.bbox.x_max),
                    min(1.0, obj.bbox.y_max),
                )
            )

        detector_ms_partial = (time.monotonic() - start_time) * 1000.0

        # ------------------------------------------------------------------
        # 2. Anchor lock path — highest priority
        # ------------------------------------------------------------------

        pointing = self._try_pointing_lock(state, object_bboxes)

        # Pointing returns during hand-loss hold → immediate restore
        if pointing is not None:
            cx, cy, label, source, reason, hand_id = pointing
            self._opportunity_mode = "ACTIVE"
            self._anchor.anchor_locked = True
            self._active_type = VisualOpportunityType.POINTING_TARGET
            self._active_source = source
            self._active_label = label
            self._active_hand = hand_id

            opp = self._build_anchor_opportunity(
                VisualOpportunityType.POINTING_TARGET,
                cx, cy, source, reason, label, hand_id,
                locked=True, timestamp=state.timestamp,
                detector_ms=(time.monotonic() - start_time) * 1000.0,
                concept_label=concept_label,
            )
            self._last_valid_opportunity = opp
            self._last_valid_timestamp = state.timestamp
            self._update_debug(state.timestamp)
            return opp

        # Locked pointing lost — unlock if hand still visible, else hand-loss hold
        if (
            self._anchor.anchor_locked
            and self._anchor.active_anchor_type == VisualOpportunityType.POINTING_TARGET
        ):
            hand_still_visible = bool(
                state.human_perception and state.human_perception.hands
            )
            if hand_still_visible:
                # Pointing stopped but hand present → release lock, fall through to ON_HAND
                self._anchor.anchor_locked = False
                self._opportunity_mode = "ACTIVE"
            else:
                if self._opportunity_mode == "ACTIVE":
                    self._opportunity_mode = "HAND_LOSS_HOLD"
                    self._last_valid_timestamp = state.timestamp

                time_since_loss_ms = (state.timestamp - self._last_valid_timestamp) * 1000.0

                if self._opportunity_mode == "HAND_LOSS_HOLD" and time_since_loss_ms < HAND_LOSS_HOLD_MS:
                    if self._last_valid_opportunity:
                        self._update_debug(state.timestamp)
                        return self._last_valid_opportunity

                if self._opportunity_mode == "HAND_LOSS_HOLD":
                    self._opportunity_mode = "FALLBACK_CONFIRMATION"
                    self._anchor.anchor_locked = False
                    self._fallback_streak = 0
                    self._fallback_candidate_source = None
                    self._fallback_candidate_type = None

        num_hands = state.human_perception.num_hands if state.human_perception else 0
        in_transfer = self._anchor.transition_state != TransitionState.NONE

        # In-progress hand transfer — interpolation only, must not fall through
        if in_transfer:
            target = self._resolve_transfer_target(state)
            hand_id = (
                self._anchor.transition_target_hand
                or self._anchor.active_anchor_hand
                or "NONE"
            )
            if target is not None:
                cx, cy = target.center_x, target.center_y
                source, reason, label = target.source, target.reason, target.label
            else:
                cx, cy = self._anchor.active_anchor_position
                source = self._anchor.anchor_source or "none"
                reason = self._anchor.anchor_reason or "Hand transfer"
                label = self._anchor.anchor_label or "PALM_CENTER"

            return self._emit_on_hand_opportunity(
                cx, cy, source, reason, label, hand_id,
                locked=True, state=state, start_time=start_time,
                concept_label=concept_label,
            )

        # Single-hand palm path
        if num_hands <= 1:
            palm = self._try_palm_anchor(state)
            if palm is not None:
                cx, cy, label, source, reason, hand_id, dwell_locked = palm

                locked = dwell_locked or (
                    self._anchor.anchor_locked
                    and self._anchor.active_anchor_type == VisualOpportunityType.ON_HAND
                    and self._anchor.active_anchor_hand == hand_id
                )

                if dwell_locked:
                    self._anchor.anchor_locked = True

                return self._emit_on_hand_opportunity(
                    cx, cy, source, reason, label, hand_id,
                    locked=locked, state=state, start_time=start_time,
                    concept_label=concept_label,
                )

        # Multi-hand ON_HAND: hysteresis + transfer (no fallback regions)
        elif num_hands >= 2:
            on_hand_cands = self._generate_on_hand_candidates(state)
            if on_hand_cands:
                winner = self._select_from_candidates(on_hand_cands, state)
                if winner is None:
                    if self._last_valid_opportunity:
                        self._update_debug(state.timestamp)
                        return self._last_valid_opportunity
                    winner = on_hand_cands[0]

                hand_key = self._hand_id(winner.source)
                dwell_locked = self._update_palm_dwell(
                    hand_key,
                    (winner.center_x, winner.center_y),
                    state.timestamp,
                    is_open=True,
                )
                locked = dwell_locked or (
                    self._anchor.anchor_locked
                    and self._anchor.active_anchor_type == VisualOpportunityType.ON_HAND
                    and self._anchor.active_anchor_hand == hand_key
                )
                if dwell_locked:
                    self._anchor.anchor_locked = True
                self._anchor.active_anchor_type = VisualOpportunityType.ON_HAND
                self._anchor.active_anchor_hand = hand_key

                return self._emit_on_hand_opportunity(
                    winner.center_x, winner.center_y,
                    winner.source, winner.reason, winner.label,
                    hand_key,
                    locked=locked, state=state, start_time=start_time,
                    concept_label=concept_label,
                )

        # Locked ON_HAND lost — active anchor hand no longer visible
        elif (
            self._anchor.anchor_locked
            and self._anchor.active_anchor_type == VisualOpportunityType.ON_HAND
            and not self._is_active_anchor_hand_present(state)
        ):
            if self._opportunity_mode == "ACTIVE":
                self._opportunity_mode = "HAND_LOSS_HOLD"
                self._last_valid_timestamp = state.timestamp

            time_since_loss_ms = (state.timestamp - self._last_valid_timestamp) * 1000.0

            if self._opportunity_mode == "HAND_LOSS_HOLD" and time_since_loss_ms < HAND_LOSS_HOLD_MS:
                if self._last_valid_opportunity:
                    self._update_debug(state.timestamp)
                    return self._last_valid_opportunity

            if self._opportunity_mode == "HAND_LOSS_HOLD":
                self._opportunity_mode = "FALLBACK_CONFIRMATION"
                self._anchor.anchor_locked = False

        # ------------------------------------------------------------------
        # 3. Fallback path — disabled while anchor_locked
        # ------------------------------------------------------------------

        if self._anchor.anchor_locked:
            if self._last_valid_opportunity:
                self._update_debug(state.timestamp)
                return self._last_valid_opportunity

        include_hand_candidates = not self._anchor.anchor_locked
        candidates = self._generate_fallback_candidates(
            state, person_bbox, face_bbox, hand_bboxes, object_bboxes,
            include_hand_candidates=include_hand_candidates,
        )

        if not candidates:
            self._update_debug(state.timestamp)
            return self._make_unavailable((time.monotonic() - start_time) * 1000.0, concept_label=concept_label)

        winner = self._select_from_candidates(candidates, state)

        if winner is None:
            # Hand-loss hold or fallback confirmation in progress
            if self._last_valid_opportunity:
                self._update_debug(state.timestamp)
                return self._last_valid_opportunity
            winner = candidates[0]

        # Update palm dwell lock when ON_HAND selected via multi-hand stability path
        if winner.type == VisualOpportunityType.ON_HAND:
            hand_key = self._hand_id(winner.source)
            dwell_locked = self._update_palm_dwell(
                hand_key,
                (winner.center_x, winner.center_y),
                state.timestamp,
                is_open=True,
            )
            if dwell_locked:
                self._anchor.anchor_locked = True
                self._anchor.active_anchor_type = VisualOpportunityType.ON_HAND
                self._anchor.active_anchor_hand = hand_key

        if winner.score < self._min_score_threshold:
            self._smoothing_active = False
            self._active_type = None
            self._active_source = None
            self._active_label = None
            self._active_hand = None
            self._anchor.anchor_locked = False
            self._opportunity_mode = "ACTIVE"
            self._update_debug(state.timestamp)
            return self._make_unavailable((time.monotonic() - start_time) * 1000.0, concept_label=concept_label)

        # EMA smoothing for fallback path (not during transfer)
        if self._anchor.transition_state != TransitionState.NONE:
            transition_bbox = self._render_transition_bbox(state.timestamp)
            if transition_bbox is not None:
                fcx, fcy, fw, fh = self._bbox_center_size(transition_bbox)
                self._smooth_cx, self._smooth_cy = fcx, fcy
                self._smooth_w, self._smooth_h = fw, fh
                final_bbox = transition_bbox
            else:
                fcx, fcy, fw, fh = self._apply_ema(
                    winner.center_x, winner.center_y, winner.width, winner.height
                )
                final_bbox = self._bbox_from_center(fcx, fcy, fw, fh)
        else:
            fcx, fcy, fw, fh = self._apply_ema(
                winner.center_x, winner.center_y, winner.width, winner.height
            )
            final_bbox = self._bbox_from_center(fcx, fcy, fw, fh)

        normalized_score = min(1.0, max(0.0, winner.score / 60.0))

        effective_concept = concept_label or getattr(self, "_active_concept_label", None)

        opp = VisualOpportunity(
            state=VisualOpportunityState.AVAILABLE,
            type=winner.type,
            available=True,
            bbox=final_bbox,
            center=(fcx, fcy),
            width=fw,
            height=fh,
            score=normalized_score,
            source=winner.source,
            reason=winner.reason,
            region_label=winner.label,
            detector_ms=(time.monotonic() - start_time) * 1000.0,
            anchor_locked=self._anchor.anchor_locked,
            anchor_hand=self._active_hand,
            anchor_type=self._anchor.active_anchor_type,
            anchor_confidence=self._anchor.anchor_confidence,
            anchor_timestamp=state.timestamp,
            concept_label=effective_concept,
        )

        self._last_valid_opportunity = opp
        if self._opportunity_mode == "ACTIVE":
            self._last_valid_timestamp = state.timestamp

        self._update_debug(state.timestamp)
        return opp
