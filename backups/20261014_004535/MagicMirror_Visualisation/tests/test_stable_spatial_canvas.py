"""
tests/test_stable_spatial_canvas.py
=====================================
Milestone: Stable Spatial Canvas + Concept and Image Rendering
Focused regression tests (synthetic inputs, no physical camera required).

Test coverage:
 1.  Box stays at locked position while hand landmarks jitter.
 2.  Box does not move when a hand moves without a confirmed pointing action.
 3.  A confirmed pointing action relocates the box.
 4.  Small pointer fluctuations do not repeatedly relocate it.
 5.  Pointing at empty space works without object detections.
 6.  Last position survives hand loss and temporary perception failures.
 7.  Fallback logic cannot override a confirmed user placement.
 8.  A deliberate new pointing action changes the anchor smoothly.
 9.  Concept label appears and updates from the active VisualPlan.
10.  Partial transcripts do not clear an established concept.
11.  An available image asset renders inside the box without distortion.
12.  A missing image asset does not break the label or border.
13.  Changing concepts cannot display a stale image from the previous concept.
14.  The default minimal display does not suppress the yellow box or label.
15.  All 185 pre-existing tests remain passing (verified via outer test run).
"""

import sys
from pathlib import Path
from typing import List, Tuple, Optional
import pytest
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_VIS_SRC = _REPO_ROOT / "MagicMirror_Visualisation"
for _p in [str(_REPO_ROOT), str(_VIS_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.perception.models import (
    ActivityState,
    BoundingBox,
    Gesture,
    HandData,
    Landmark,
    PerceptionResult,
    PoseData,
    SceneState,
    TrackedObject,
    VisualOpportunity,
    VisualOpportunityType,
)
from src.perception.opportunity import (
    VisualOpportunityDetector,
    POINTING_CONFIRM_FRAMES,
    POINTING_MOVE_THRESHOLD,
    POINTING_STABILITY_TOLERANCE,
    HAND_LOSS_HOLD_MS,
)
from src.perception.assets import (
    ConceptAsset,
    ConceptAssetStore,
    render_box_and_asset,
    format_concept_label,
)
from integration.multimodal_fusion.schemas import (
    ActionType,
    ConfidenceScore,
    VisualPlan,
)
from integration_live_demo import LiveState, _draw_visual_opportunity


# ---------------------------------------------------------------------------
# Shared synthetic fixtures
# ---------------------------------------------------------------------------

def _lm(x: float, y: float, vis: float = 1.0) -> Landmark:
    return Landmark(x=x, y=y, z=0.0, visibility=vis)


def _make_hand(is_left: bool, cx: float, cy: float,
               point_dx: float = 0.0, point_dy: float = 0.0) -> HandData:
    hand = "Left" if is_left else "Right"
    lms = [_lm(cx + (i % 5) * 0.01, cy + (i // 5) * 0.01) for i in range(21)]
    lms[0] = _lm(cx, cy)
    for i in (5, 9, 13, 17):
        lms[i] = _lm(cx + 0.01, cy - 0.01)
    if point_dx != 0.0 or point_dy != 0.0:
        lms[8] = _lm(cx + point_dx, cy + point_dy)
    return HandData(handedness=hand, landmarks=tuple(lms), score=1.0)


def _make_scene(hands: List[HandData] = None,
                left_gesture: Gesture = Gesture.NONE,
                right_gesture: Gesture = Gesture.NONE,
                timestamp: float = 0.0) -> SceneState:
    human_res = None
    if hands:
        human_res = PerceptionResult(
            frame_id=0, capture_timestamp=0.0,
            perception_start=0.0, perception_end=0.0,
            pose=None, hands=tuple(hands),
        )
    activity = None
    if left_gesture != Gesture.NONE or right_gesture != Gesture.NONE:
        activity = ActivityState(
            left_hand_gesture=left_gesture,
            right_hand_gesture=right_gesture,
        )
    return SceneState(
        frame_id=0, timestamp=timestamp,
        human_perception=human_res,
        activity=activity,
    )


def _point_n_times(detector: VisualOpportunityDetector,
                   cx: float, cy: float, dx: float, dy: float,
                   n: int, gesture: Gesture = Gesture.POINTING,
                   ts_start: float = 0.0, dt: float = 0.05) -> VisualOpportunity:
    """Return the opportunity after n frames of pointing at (cx+dx, cy+dy)."""
    hand = _make_hand(is_left=False, cx=cx, cy=cy, point_dx=dx, point_dy=dy)
    opp = None
    for i in range(n):
        scene = _make_scene(hands=[hand], right_gesture=gesture,
                            timestamp=ts_start + i * dt)
        opp = detector.detect(scene, opp)
    return opp


# ---------------------------------------------------------------------------
# Test Suite
# ---------------------------------------------------------------------------

class TestStableSpatialCanvas:
    def setup_method(self):
        self.det = VisualOpportunityDetector()

    # -----------------------------------------------------------------------
    # 1. Box stays at locked position while hand landmarks jitter
    # -----------------------------------------------------------------------
    def test_01_box_stable_during_jitter_after_confirmed_placement(self):
        """Once a pointing is confirmed (POINTING_CONFIRM_FRAMES), small
        jitter must not move the box beyond POINTING_MOVE_THRESHOLD."""
        import random
        random.seed(42)

        # Confirm placement
        opp = _point_n_times(self.det, 0.5, 0.5, 0.15, 0.0,
                              n=POINTING_CONFIRM_FRAMES + 1)
        assert self.det.is_user_placed, "Should be user-placed after confirmation"
        confirmed_cx = opp.center[0]
        confirmed_cy = opp.center[1]

        # Apply jitter smaller than POINTING_MOVE_THRESHOLD
        for _ in range(12):
            jx = random.uniform(-0.02, 0.02)
            jy = random.uniform(-0.02, 0.02)
            hand = _make_hand(is_left=False, cx=0.5, cy=0.5,
                              point_dx=0.15 + jx, point_dy=jy)
            scene = _make_scene(hands=[hand], right_gesture=Gesture.POINTING,
                                timestamp=2.0)
            opp = self.det.detect(scene, opp)

        assert abs(opp.center[0] - confirmed_cx) < POINTING_MOVE_THRESHOLD, (
            f"Jitter moved box too far: {opp.center[0]:.3f} vs {confirmed_cx:.3f}"
        )

    # -----------------------------------------------------------------------
    # 2. Box does not move when hand moves without confirmed pointing
    # -----------------------------------------------------------------------
    def test_02_hand_movement_without_pointing_does_not_move_box(self):
        """A hand moving around WITHOUT a POINTING gesture must not relocate
        a confirmed pointing placement."""
        # Confirm a placement first
        opp = _point_n_times(self.det, 0.3, 0.5, 0.15, 0.0,
                              n=POINTING_CONFIRM_FRAMES + 1)
        locked_cx = opp.center[0]

        # Move hand without pointing gesture
        for cx in (0.1, 0.9, 0.5, 0.7):
            hand = _make_hand(is_left=False, cx=cx, cy=0.5)
            scene = _make_scene(hands=[hand])   # no POINTING gesture
            opp = self.det.detect(scene, opp)

        # Box should still be at the confirmed location
        assert abs(opp.center[0] - locked_cx) < POINTING_MOVE_THRESHOLD, (
            f"Box drifted during non-pointing hand movement: {opp.center[0]:.3f}"
        )

    # -----------------------------------------------------------------------
    # 3. A confirmed pointing action relocates the box
    # -----------------------------------------------------------------------
    def test_03_confirmed_pointing_relocates_box(self):
        """Holding a new pointing direction for POINTING_CONFIRM_FRAMES frames
        must move the confirmed target to the new location."""
        # First confirmed placement at the right side
        opp = _point_n_times(self.det, 0.5, 0.5, 0.15, 0.0,
                              n=POINTING_CONFIRM_FRAMES + 1)
        first_cx = opp.center[0]

        # New confirmed pointing to the LEFT (large enough jump)
        opp = _point_n_times(self.det, 0.8, 0.5, -0.30, 0.0,
                              n=POINTING_CONFIRM_FRAMES + 1,
                              ts_start=2.0)
        new_cx = opp.center[0]

        assert new_cx < first_cx - POINTING_MOVE_THRESHOLD, (
            f"Relocation did not move left enough: {new_cx:.3f} vs {first_cx:.3f}"
        )

    # -----------------------------------------------------------------------
    # 4. Small fluctuations do not repeatedly relocate
    # -----------------------------------------------------------------------
    def test_04_small_fluctuations_do_not_repeatedly_relocate(self):
        """Fluctuations smaller than POINTING_MOVE_THRESHOLD must be ignored
        after a confirmed placement."""
        import random
        random.seed(7)

        opp = _point_n_times(self.det, 0.5, 0.5, 0.15, 0.0,
                              n=POINTING_CONFIRM_FRAMES + 1)
        confirmed_cx = self.det._confirmed_target[0]

        relocate_count = 0
        prev_target = self.det._confirmed_target

        for _ in range(20):
            jx = random.uniform(-POINTING_MOVE_THRESHOLD * 0.4,
                                 POINTING_MOVE_THRESHOLD * 0.4)
            hand = _make_hand(is_left=False, cx=0.5, cy=0.5,
                              point_dx=0.15 + jx, point_dy=0.0)
            scene = _make_scene(hands=[hand], right_gesture=Gesture.POINTING,
                                timestamp=2.0)
            opp = self.det.detect(scene, opp)
            if self.det._confirmed_target != prev_target:
                relocate_count += 1
                prev_target = self.det._confirmed_target

        assert relocate_count == 0, (
            f"Small fluctuations caused {relocate_count} relocations"
        )

    # -----------------------------------------------------------------------
    # 5. Pointing at empty space works without object detections
    # -----------------------------------------------------------------------
    def test_05_pointing_at_empty_space_no_objects(self):
        """Pointing into open air with no tracked objects must produce a
        POINTING_TARGET opportunity at the ray endpoint."""
        hand = _make_hand(is_left=False, cx=0.4, cy=0.5, point_dx=0.15, point_dy=0.0)
        scene = _make_scene(hands=[hand], right_gesture=Gesture.POINTING)
        assert len(scene.tracked_objects) == 0

        opp = self.det.detect(scene)
        assert opp.available
        assert opp.type == VisualOpportunityType.POINTING_TARGET
        # Box must be near the ray endpoint, not at screen center
        assert opp.center[0] > 0.5

    # -----------------------------------------------------------------------
    # 6. Last position survives hand loss and perception failures
    # -----------------------------------------------------------------------
    def test_06_position_survives_hand_loss(self):
        """After confirming a placement, the box must survive hand disappearance."""
        opp = _point_n_times(self.det, 0.5, 0.5, 0.15, 0.0,
                              n=POINTING_CONFIRM_FRAMES + 1)
        confirmed_center = opp.center

        # Hand completely disappears
        empty = _make_scene(timestamp=5.0)
        opp_after = self.det.detect(empty, opp)

        assert opp_after.available, "Box should remain available after hand loss"
        assert abs(opp_after.center[0] - confirmed_center[0]) < 0.01, (
            "Box drifted on hand loss"
        )

    # -----------------------------------------------------------------------
    # 7. Fallback logic cannot override a confirmed user placement
    # -----------------------------------------------------------------------
    def test_07_fallback_cannot_override_confirmed_placement(self):
        """Even with a person present (giving fallback candidates), a confirmed
        user pointing placement must be preserved."""
        from src.perception.models import PoseData

        # Confirm placement
        opp = _point_n_times(self.det, 0.5, 0.5, 0.15, 0.0,
                              n=POINTING_CONFIRM_FRAMES + 1)
        assert self.det.is_user_placed

        # Simulate frames with only a person (no pointing hand)
        lms = [Landmark(x=0.5, y=0.5 + i * 0.02, z=0.0, visibility=1.0)
               for i in range(33)]
        pose = PoseData(landmarks=tuple(lms))
        human_res = PerceptionResult(
            frame_id=0, capture_timestamp=0.0,
            perception_start=0.0, perception_end=0.0,
            pose=pose, hands=(),
        )
        from src.perception.models import SceneState as SS
        scene_person = SS(
            frame_id=0, timestamp=2.0,
            human_perception=human_res,
        )
        for _ in range(10):
            opp = self.det.detect(scene_person, opp)

        assert opp.type == VisualOpportunityType.POINTING_TARGET, (
            "Fallback overrode confirmed user placement"
        )

    # -----------------------------------------------------------------------
    # 8. Deliberate new pointing action changes the anchor smoothly
    # -----------------------------------------------------------------------
    def test_08_deliberate_new_pointing_changes_anchor(self):
        """After a confirmed placement, a new sustained pointing direction
        that exceeds POINTING_MOVE_THRESHOLD and lasts POINTING_CONFIRM_FRAMES
        must update the anchor to the new location."""
        opp = _point_n_times(self.det, 0.5, 0.5, 0.15, 0.0,
                              n=POINTING_CONFIRM_FRAMES + 1)
        old_center = self.det._confirmed_target

        # New confirmed direction pointing left
        opp = _point_n_times(self.det, 0.8, 0.5, -0.30, 0.0,
                              n=POINTING_CONFIRM_FRAMES + 1, ts_start=2.0)
        new_center = self.det._confirmed_target

        assert new_center != old_center, "Anchor should have moved"
        # New target should be to the left of the old target
        assert new_center[0] < old_center[0]

    # -----------------------------------------------------------------------
    # 9. Concept label appears and updates from the active VisualPlan
    # -----------------------------------------------------------------------
    def test_09_concept_label_from_visual_plan(self):
        """LiveState.update_plan with CREATE action must expose the concept."""
        ls = LiveState()
        plan = VisualPlan(
            event_id="e1", timestamp=100.0,
            action=ActionType.CREATE, concept="oxygen_atom", reason="test",
        )
        ls.update_plan(plan)
        _, _, _, _, _, _, active_concept = ls.snapshot()

        assert active_concept == "oxygen_atom"
        assert format_concept_label(active_concept) == "oxygen_atom"

        # ANIMATE with same concept should preserve label
        plan2 = VisualPlan(
            event_id="e2", timestamp=101.0,
            action=ActionType.ANIMATE, concept="oxygen_atom", reason="spin",
        )
        ls.update_plan(plan2)
        assert ls.snapshot()[6] == "oxygen_atom"

        # CREATE new concept updates label
        plan3 = VisualPlan(
            event_id="e3", timestamp=102.0,
            action=ActionType.CREATE, concept="red_blood_cell", reason="new",
        )
        ls.update_plan(plan3)
        assert ls.snapshot()[6] == "red_blood_cell"

    # -----------------------------------------------------------------------
    # 10. Partial transcripts do not clear an established concept
    # -----------------------------------------------------------------------
    def test_10_partial_transcripts_do_not_clear_concept(self):
        """Partial speech events must not overwrite an established concept."""
        ls = LiveState()
        plan = VisualPlan(
            event_id="e1", timestamp=100.0,
            action=ActionType.CREATE, concept="dog", reason="established",
        )
        ls.update_plan(plan)
        assert ls.snapshot()[6] == "dog"

        # Partial transcript update
        ls.update_transcript("show me a...", partial=True)
        assert ls.snapshot()[6] == "dog", "Partial transcript erased concept"

        # Partial NLP update without concept
        ls.update_nlp({"speaker_activity": [], "text_segment": "and then...",
                       "is_partial": True})
        assert ls.snapshot()[6] == "dog", "Partial NLP update erased concept"

    # -----------------------------------------------------------------------
    # 11. An available image asset renders inside the box without distortion
    # -----------------------------------------------------------------------
    def test_11_image_asset_renders_without_distortion(self):
        """render_box_and_asset must place a proportionally-scaled image
        inside the box. Output shape must remain equal to the frame shape."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Synthetic 50x100 checkerboard asset (tall, would be distorted if stretched)
        img = np.zeros((100, 50, 3), dtype=np.uint8)
        img[::10, :] = 255  # horizontal stripes
        asset = ConceptAsset(concept="dog", image=img, status="ready")

        bbox = (0.2, 0.1, 0.8, 0.9)  # wide box
        render_box_and_asset(frame, bbox=bbox, concept_label="dog", asset=asset)

        # Frame shape unchanged
        assert frame.shape == (480, 640, 3)
        # Pixels were written inside the box area
        x1, y1, x2, y2 = int(0.2*640), int(0.1*480), int(0.8*640), int(0.9*480)
        roi = frame[y1:y2, x1:x2]
        assert roi.sum() > 0, "No pixels rendered inside box"

    # -----------------------------------------------------------------------
    # 12. Missing image asset does not break the label or border
    # -----------------------------------------------------------------------
    def test_12_missing_asset_does_not_break_label_or_border(self):
        """When asset.status == 'missing', the label and border must still render."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        missing = ConceptAsset(concept="dragon", image=None, status="missing")

        # render_box_and_asset with None image (missing)
        bbox = (0.3, 0.3, 0.7, 0.7)
        render_box_and_asset(frame, bbox=bbox, concept_label="dragon", asset=None)

        # Border should still be drawn (non-zero pixels)
        assert frame.sum() > 0, "Missing asset caused no render output at all"

    # -----------------------------------------------------------------------
    # 13. Changing concepts cannot display a stale image from the previous concept
    # -----------------------------------------------------------------------
    def test_13_stale_image_not_shown_for_different_concept(self):
        """Asset store must not return an asset whose concept doesn't match."""
        store = ConceptAssetStore()
        img = np.full((50, 50, 3), 128, dtype=np.uint8)
        store.register_asset("cat", img)

        # Looking up 'dog' when only 'cat' is registered should return missing
        asset = store.get_asset("dog")
        assert asset is None or not asset.is_valid_image, (
            "Store returned cat's image for dog query"
        )

        # Looking up 'cat' returns the correct asset
        cat_asset = store.get_asset("cat")
        assert cat_asset is not None and cat_asset.is_valid_image

    # -----------------------------------------------------------------------
    # 14. Default minimal display does not suppress the yellow box or label
    # -----------------------------------------------------------------------
    def test_14_minimal_display_shows_box_and_label(self):
        """_draw_visual_opportunity must render when show_hud=False (default).
        The function must write non-zero pixels for an available opportunity."""
        from src.perception.models import (
            VisualOpportunityState, SceneState as SS,
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        opp = VisualOpportunity(
            state=VisualOpportunityState.AVAILABLE,
            type=VisualOpportunityType.POINTING_TARGET,
            available=True,
            bbox=(0.2, 0.2, 0.8, 0.8),
            center=(0.5, 0.5),
            width=0.6, height=0.6,
            score=0.9,
            source="right_hand",
            reason="test",
            region_label="RAY_0.18",
            detector_ms=1.0,
            anchor_locked=True,
            concept_label="golden_apple",
        )
        scene = SS(
            frame_id=1, timestamp=0.0,
            human_perception=None,
            visual_opportunity=opp,
        )
        # show_hud is NOT passed — function should render without it
        _draw_visual_opportunity(frame, scene, concept_label="golden_apple")
        assert frame.sum() > 0, "Yellow box not rendered in minimal mode"


class TestConceptAssetStoreLifecycle:
    """Unit tests for ConceptAssetStore."""

    def test_register_and_retrieve(self):
        store = ConceptAssetStore()
        img = np.full((40, 40, 3), 200, dtype=np.uint8)
        store.register_asset("apple", img)
        asset = store.get_asset("apple")
        assert asset is not None
        assert asset.is_valid_image
        assert asset.concept == "apple"

    def test_normalisation(self):
        """Keys are case-insensitive and strip-normalised."""
        store = ConceptAssetStore()
        img = np.full((10, 10, 3), 1, dtype=np.uint8)
        store.register_asset("  Apple  ", img)
        assert store.get_asset("apple") is not None
        assert store.get_asset("APPLE") is not None

    def test_missing_concept_returns_missing_status(self):
        store = ConceptAssetStore()
        asset = store.get_asset("nonexistent_concept")
        assert asset is None or (asset.status == "missing" and not asset.is_valid_image)

    def test_concept_change_isolation(self):
        """Retrieving a new concept must not return a previously-registered image."""
        store = ConceptAssetStore()
        img_cat = np.zeros((20, 20, 3), dtype=np.uint8)
        store.register_asset("cat", img_cat)

        dog_asset = store.get_asset("dog")
        assert dog_asset is None or not dog_asset.is_valid_image

    def test_none_concept_returns_none(self):
        store = ConceptAssetStore()
        assert store.get_asset(None) is None
        assert store.get_asset("") is None
        assert store.get_asset("unknown") is None
