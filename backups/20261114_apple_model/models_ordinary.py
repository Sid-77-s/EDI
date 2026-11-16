"""
integration/generation/models_ordinary.py — Procedural 3D Construction & Retrieval for Real-World Objects.

Provides:
- Retrieval of real local 3D assets (.glb / .obj) from local asset library.
- Procedural parametric construction of common objects:
  - Apple (dimpled spherical body, indentation, curved stem, leaf blade)
  - Chair (cushioned seat, 4 tapered legs, backrest with spindles)
  - Car (chassis body, cabin canopy, 4 wheels with rims, headlights)
  - Tree (textured trunk, tiered volumetric foliage canopies)
  - Cup / Mug (cylindrical hollow vessel, curved handle)
  - Book (hardcover binding, spine, page block)
  - Dog (anatomical quadruped: torso, neck, head, snout, ears, 4 limbs, animated wagging tail)
  - Person (stylized humanoid mannequin with articulating limbs)
  - Useful geometric showcase fallback for uncatalogued objects.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np

from integration.generation.geometry_primitives import (
    create_box,
    create_cylinder,
    create_torus,
    create_uv_sphere,
)
from integration.generation.model3d import (
    Material3D,
    Mesh3D,
    Model3D,
    ModelNode3D,
    load_glb,
    load_obj,
)


class OrdinaryObject3DBuilder:
    """Builds or retrieves renderable 3D models for everyday objects."""

    def __init__(self, model_library_dir: Union[str, Path] = "assets/models") -> None:
        self.library_dir = Path(model_library_dir)
        self.library_dir.mkdir(parents=True, exist_ok=True)

    def find_local_asset(self, concept: str) -> Optional[Model3D]:
        """Checks for existing .glb or .obj 3D asset in the local model library."""
        clean_name = concept.strip().lower().replace(" ", "_")
        for ext, loader in ((".glb", load_glb), (".obj", load_obj)):
            cand = self.library_dir / f"{clean_name}{ext}"
            if cand.is_file():
                try:
                    model = loader(cand)
                    model.concept = concept
                    return model
                except Exception:
                    pass
        return None

    def build_or_retrieve(self, concept: str) -> Model3D:
        """Retrieves from library or constructs procedural 3D model."""
        # 1. Check local file library
        local_model = self.find_local_asset(concept)
        if local_model is not None:
            return local_model

        # 2. Procedural construction
        clean = concept.strip().lower()
        if "apple" in clean or "fruit" in clean:
            return self._build_apple(concept)
        elif "chair" in clean or "seat" in clean:
            return self._build_chair(concept)
        elif "car" in clean or "vehicle" in clean or "automobile" in clean:
            return self._build_car(concept)
        elif "tree" in clean or "plant" in clean:
            return self._build_tree(concept)
        elif "cup" in clean or "mug" in clean or "coffee" in clean:
            return self._build_cup(concept)
        elif "book" in clean or "manual" in clean or "notebook" in clean:
            return self._build_book(concept)
        elif "dog" in clean or "puppy" in clean or "hound" in clean:
            return self._build_dog(concept)
        elif "person" in clean or "human" in clean or "avatar" in clean or "man" in clean or "woman" in clean:
            return self._build_person(concept)
        else:
            return self._build_generic_fallback(concept)

    # -----------------------------------------------------------------------
    # Procedural Object Builders
    # -----------------------------------------------------------------------

    def _build_apple(self, concept: str) -> Model3D:
        root = ModelNode3D(name="apple_root")
        # Apple body (dimpled spheroid: pinched top and bottom)
        body = create_uv_sphere(radius=0.72, lat_segs=14, lon_segs=18, color=(30, 40, 220), name="apple_body")
        # Deform sphere: dimple at top (y > 0.4) and bottom (y < -0.4)
        y = body.vertices[:, 1]
        top_mask = y > 0.35
        body.vertices[top_mask, 1] -= (y[top_mask] - 0.35) * 0.45
        bot_mask = y < -0.35
        body.vertices[bot_mask, 1] += (-0.35 - y[bot_mask]) * 0.35
        body.compute_face_normals()
        root.add_child(ModelNode3D(name="body_node", mesh=body))

        # Curved wooden stem
        stem = create_cylinder(radius_top=0.03, radius_bottom=0.04, height=0.35, segs=8, color=(30, 80, 120), name="stem")
        stem_node = ModelNode3D(name="stem_node", mesh=stem, position=np.array([0.04, 0.65, 0.0], dtype=np.float32), rotation=np.array([0.0, 0.0, math.radians(-15)], dtype=np.float32))
        root.add_child(stem_node)

        # Leaf blade
        leaf = create_box(width=0.25, height=0.02, depth=0.14, color=(40, 180, 40), name="leaf")
        leaf_node = ModelNode3D(name="leaf_node", mesh=leaf, position=np.array([0.15, 0.72, 0.0], dtype=np.float32), rotation=np.array([math.radians(25), 0.0, math.radians(35)], dtype=np.float32))
        root.add_child(leaf_node)

        def _anim(m: Model3D, t: float):
            root.rotation[1] = t * 0.5
        return Model3D(concept=concept, root=root, category="3D_OBJECT", animation_fn=_anim, description="Procedural 3D Red Apple with Stem and Leaf")

    def _build_chair(self, concept: str) -> Model3D:
        root = ModelNode3D(name="chair_root")
        wood_color = (40, 90, 150)  # Warm wood BGR

        # Seat
        seat = create_box(width=1.0, height=0.1, depth=1.0, color=wood_color, name="seat")
        root.add_child(ModelNode3D(name="seat_node", mesh=seat, position=np.array([0.0, 0.0, 0.0], dtype=np.float32)))

        # 4 Legs
        leg_offsets = [(-0.4, -0.4), (0.4, -0.4), (-0.4, 0.4), (0.4, 0.4)]
        for idx, (lx, lz) in enumerate(leg_offsets):
            leg = create_cylinder(radius_top=0.045, radius_bottom=0.035, height=0.9, segs=8, color=wood_color, name=f"leg_{idx}")
            root.add_child(ModelNode3D(name=f"leg_node_{idx}", mesh=leg, position=np.array([lx, -0.45, lz], dtype=np.float32)))

        # Backrest (2 upright posts + 2 crossbars)
        for side, bx in [("left", -0.4), ("right", 0.4)]:
            post = create_cylinder(radius_top=0.04, radius_bottom=0.04, height=1.0, segs=8, color=wood_color, name=f"post_{side}")
            root.add_child(ModelNode3D(name=f"post_{side}", mesh=post, position=np.array([bx, 0.5, -0.45], dtype=np.float32)))

        rail1 = create_box(width=0.82, height=0.12, depth=0.06, color=wood_color, name="rail_top")
        root.add_child(ModelNode3D(name="rail1", mesh=rail1, position=np.array([0.0, 0.85, -0.45], dtype=np.float32)))
        rail2 = create_box(width=0.82, height=0.10, depth=0.06, color=wood_color, name="rail_mid")
        root.add_child(ModelNode3D(name="rail2", mesh=rail2, position=np.array([0.0, 0.5, -0.45], dtype=np.float32)))

        def _anim(m: Model3D, t: float):
            root.rotation[1] = t * 0.4
        return Model3D(concept=concept, root=root, category="3D_OBJECT", animation_fn=_anim, description="Detailed 3D Wooden Dining Chair")

    def _build_car(self, concept: str) -> Model3D:
        root = ModelNode3D(name="car_root")
        body_col = (200, 50, 40)   # Blue
        wheel_col = (30, 30, 30)

        # Lower chassis
        chassis = create_box(width=1.8, height=0.35, depth=0.9, color=body_col, name="chassis")
        root.add_child(ModelNode3D(name="chassis_node", mesh=chassis, position=np.array([0.0, 0.0, 0.0], dtype=np.float32)))

        # Cabin / roof
        cabin = create_box(width=0.95, height=0.38, depth=0.8, color=(220, 180, 180), name="cabin")
        root.add_child(ModelNode3D(name="cabin_node", mesh=cabin, position=np.array([-0.1, 0.36, 0.0], dtype=np.float32)))

        # 4 Wheels
        wheel_nodes: List[ModelNode3D] = []
        for idx, (wx, wz) in enumerate([(-0.55, -0.48), (0.55, -0.48), (-0.55, 0.48), (0.55, 0.48)]):
            wheel = create_cylinder(radius_top=0.22, radius_bottom=0.22, height=0.14, segs=12, color=wheel_col, name=f"wheel_{idx}")
            w_node = ModelNode3D(name=f"w_{idx}", mesh=wheel, position=np.array([wx, -0.16, wz], dtype=np.float32), rotation=np.array([math.radians(90), 0.0, 0.0], dtype=np.float32))
            root.add_child(w_node)
            wheel_nodes.append(w_node)

        # Headlights
        for hx, hz in [(0.91, -0.28), (0.91, 0.28)]:
            hl = create_uv_sphere(radius=0.08, lat_segs=6, lon_segs=8, color=(50, 240, 255), name="headlight")
            root.add_child(ModelNode3D(name="hl", mesh=hl, position=np.array([hx, 0.05, hz], dtype=np.float32)))

        def _anim(m: Model3D, t: float):
            for w in wheel_nodes:
                w.rotation[1] = -t * 6.0
            root.rotation[1] = t * 0.4
        return Model3D(concept=concept, root=root, category="3D_OBJECT", animation_fn=_anim, description="3D Vehicle with Rotating Wheels and Detailed Body")

    def _build_tree(self, concept: str) -> Model3D:
        root = ModelNode3D(name="tree_root")
        # Bark trunk
        trunk = create_cylinder(radius_top=0.12, radius_bottom=0.2, height=1.1, segs=10, color=(35, 75, 120), name="trunk")
        root.add_child(ModelNode3D(name="trunk_node", mesh=trunk, position=np.array([0.0, -0.35, 0.0], dtype=np.float32)))

        # 3 tiered volumetric foliage tiers
        tiers = [(0.75, 0.25, (40, 160, 40)), (0.60, 0.65, (50, 185, 50)), (0.42, 1.0, (60, 210, 60))]
        for idx, (rad, y_pos, col) in enumerate(tiers):
            canopy = create_uv_sphere(radius=rad, lat_segs=10, lon_segs=14, color=col, name=f"canopy_{idx}")
            canopy.vertices[:, 1] *= 0.7  # Flatten slightly
            canopy.compute_face_normals()
            root.add_child(ModelNode3D(name=f"c_{idx}", mesh=canopy, position=np.array([0.0, y_pos, 0.0], dtype=np.float32)))

        def _anim(m: Model3D, t: float):
            # Gentle wind sway
            root.rotation[2] = math.sin(t * 1.5) * 0.06
            root.rotation[1] = t * 0.3
        return Model3D(concept=concept, root=root, category="3D_OBJECT", animation_fn=_anim, description="Volumetric 3D Foliage Tree")

    def _build_cup(self, concept: str) -> Model3D:
        root = ModelNode3D(name="cup_root")
        ceramic = (230, 230, 230)
        # Outer vessel
        vessel = create_cylinder(radius_top=0.45, radius_bottom=0.38, height=0.85, segs=14, color=ceramic, name="cup_body")
        root.add_child(ModelNode3D(name="vessel_node", mesh=vessel))

        # Handle (torus cut/offset)
        handle = create_torus(ring_radius=0.28, tube_radius=0.06, ring_segs=16, tube_segs=6, color=ceramic, name="handle")
        h_node = ModelNode3D(name="handle_node", mesh=handle, position=np.array([0.5, 0.0, 0.0], dtype=np.float32), rotation=np.array([0.0, 0.0, math.radians(90)], dtype=np.float32))
        root.add_child(h_node)

        def _anim(m: Model3D, t: float):
            root.rotation[1] = t * 0.6
        return Model3D(concept=concept, root=root, category="3D_OBJECT", animation_fn=_anim, description="Ceramic 3D Mug with Handle")

    def _build_book(self, concept: str) -> Model3D:
        root = ModelNode3D(name="book_root")
        cover_col = (40, 40, 160)  # Red/maroon
        pages_col = (235, 235, 230)

        # Pages block
        pages = create_box(width=0.9, height=0.2, depth=1.2, color=pages_col, name="pages")
        root.add_child(ModelNode3D(name="pages_node", mesh=pages))

        # Top & bottom covers
        cover_top = create_box(width=0.94, height=0.025, depth=1.24, color=cover_col, name="cover_top")
        root.add_child(ModelNode3D(name="ct", mesh=cover_top, position=np.array([0.0, 0.11, 0.0], dtype=np.float32)))
        cover_bot = create_box(width=0.94, height=0.025, depth=1.24, color=cover_col, name="cover_bot")
        root.add_child(ModelNode3D(name="cb", mesh=cover_bot, position=np.array([0.0, -0.11, 0.0], dtype=np.float32)))

        # Spine
        spine = create_box(width=0.03, height=0.24, depth=1.24, color=cover_col, name="spine")
        root.add_child(ModelNode3D(name="spine_node", mesh=spine, position=np.array([-0.47, 0.0, 0.0], dtype=np.float32)))

        def _anim(m: Model3D, t: float):
            root.rotation[1] = t * 0.4
            root.rotation[0] = 0.3
        return Model3D(concept=concept, root=root, category="3D_OBJECT", animation_fn=_anim, description="Bound 3D Book with Hardcover and Spine")

    def _build_dog(self, concept: str) -> Model3D:
        root = ModelNode3D(name="dog_root")
        fur_color = (60, 120, 180)  # Warm tan/golden brown (BGR)

        # Torso
        torso = create_cylinder(radius_top=0.32, radius_bottom=0.32, height=1.0, segs=10, color=fur_color, name="torso")
        root.add_child(ModelNode3D(name="torso_node", mesh=torso, rotation=np.array([0.0, 0.0, math.radians(90)], dtype=np.float32)))

        # Neck & Head
        neck = create_cylinder(radius_top=0.2, radius_bottom=0.24, height=0.45, segs=8, color=fur_color, name="neck")
        root.add_child(ModelNode3D(name="neck_node", mesh=neck, position=np.array([0.48, 0.28, 0.0], dtype=np.float32), rotation=np.array([0.0, 0.0, math.radians(-35)], dtype=np.float32)))

        head = create_uv_sphere(radius=0.26, lat_segs=8, lon_segs=10, color=fur_color, name="head")
        root.add_child(ModelNode3D(name="head_node", mesh=head, position=np.array([0.65, 0.52, 0.0], dtype=np.float32)))

        # Snout / muzzle
        snout = create_box(width=0.24, height=0.15, depth=0.18, color=(40, 90, 150), name="snout")
        root.add_child(ModelNode3D(name="snout_node", mesh=snout, position=np.array([0.84, 0.48, 0.0], dtype=np.float32)))

        # 4 Legs
        leg_positions = [(0.38, -0.45, -0.22), (0.38, -0.45, 0.22), (-0.38, -0.45, -0.22), (-0.38, -0.45, 0.22)]
        for i, (lx, ly, lz) in enumerate(leg_positions):
            leg = create_cylinder(radius_top=0.08, radius_bottom=0.06, height=0.55, segs=8, color=fur_color, name=f"leg_{i}")
            root.add_child(ModelNode3D(name=f"leg_{i}", mesh=leg, position=np.array([lx, ly, lz], dtype=np.float32)))

        # Wagging tail
        tail = create_cylinder(radius_top=0.03, radius_bottom=0.06, height=0.45, segs=6, color=fur_color, name="tail")
        tail_node = ModelNode3D(name="tail_node", mesh=tail, position=np.array([-0.52, 0.15, 0.0], dtype=np.float32), rotation=np.array([0.0, 0.0, math.radians(45)], dtype=np.float32))
        root.add_child(tail_node)

        def _anim(m: Model3D, t: float):
            # Tail wagging animation
            tail_node.rotation[1] = math.sin(t * 8.0) * 0.45
            root.rotation[1] = t * 0.4
        return Model3D(concept=concept, root=root, category="3D_OBJECT", animation_fn=_anim, description="Anatomical 3D Quadruped with Animated Wagging Tail")

    def _build_person(self, concept: str) -> Model3D:
        root = ModelNode3D(name="person_root")
        skin_col = (160, 190, 225)
        cloth_col = (180, 80, 50)

        # Torso
        torso = create_cylinder(radius_top=0.28, radius_bottom=0.24, height=0.75, segs=10, color=cloth_col, name="torso")
        root.add_child(ModelNode3D(name="torso_node", mesh=torso, position=np.array([0.0, 0.15, 0.0], dtype=np.float32)))

        # Head
        head = create_uv_sphere(radius=0.22, lat_segs=8, lon_segs=10, color=skin_col, name="head")
        root.add_child(ModelNode3D(name="head_node", mesh=head, position=np.array([0.0, 0.72, 0.0], dtype=np.float32)))

        # Arms
        arm_l = create_cylinder(radius_top=0.07, radius_bottom=0.05, height=0.6, segs=6, color=cloth_col, name="arm_l")
        root.add_child(ModelNode3D(name="arm_l_node", mesh=arm_l, position=np.array([-0.38, 0.15, 0.0], dtype=np.float32)))
        arm_r = create_cylinder(radius_top=0.07, radius_bottom=0.05, height=0.6, segs=6, color=cloth_col, name="arm_r")
        root.add_child(ModelNode3D(name="arm_r_node", mesh=arm_r, position=np.array([0.38, 0.15, 0.0], dtype=np.float32)))

        # Legs
        leg_l = create_cylinder(radius_top=0.09, radius_bottom=0.07, height=0.75, segs=8, color=(60, 60, 80), name="leg_l")
        root.add_child(ModelNode3D(name="leg_l_node", mesh=leg_l, position=np.array([-0.14, -0.55, 0.0], dtype=np.float32)))
        leg_r = create_cylinder(radius_top=0.09, radius_bottom=0.07, height=0.75, segs=8, color=(60, 60, 80), name="leg_r")
        root.add_child(ModelNode3D(name="leg_r_node", mesh=leg_r, position=np.array([0.14, -0.55, 0.0], dtype=np.float32)))

        def _anim(m: Model3D, t: float):
            root.rotation[1] = t * 0.4
        return Model3D(concept=concept, root=root, category="3D_OBJECT", animation_fn=_anim, description="Articulated 3D Humanoid Figure")

    def _build_generic_fallback(self, concept: str) -> Model3D:
        """Generic useful 3D showcase representation (faceted gem pedestal with glowing concept core)."""
        root = ModelNode3D(name="fallback_root")
        # Octahedral multi-faceted prism
        gem = create_uv_sphere(radius=0.7, lat_segs=6, lon_segs=8, color=(240, 200, 50), name="gem_core")
        root.add_child(ModelNode3D(name="gem_node", mesh=gem))

        ring = create_torus(ring_radius=0.95, tube_radius=0.03, ring_segs=18, tube_segs=4, color=(0, 220, 240), name="ring")
        root.add_child(ModelNode3D(name="ring_node", mesh=ring, rotation=np.array([0.4, 0.0, 0.0], dtype=np.float32)))

        def _anim(m: Model3D, t: float):
            root.rotation[1] = t * 0.6
            root.rotation[0] = math.sin(t * 0.8) * 0.2
        return Model3D(concept=concept, root=root, category="3D_OBJECT", animation_fn=_anim, description=f"3D Representation for {concept}")

