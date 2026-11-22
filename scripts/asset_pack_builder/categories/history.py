"""
scripts/asset_pack_builder/categories/history.py — 10 History, Architecture, and Culture Models.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple
import numpy as np

from integration.generation.model3d import Material3D, Model3D, ModelNode3D
from scripts.asset_pack_builder.primitives import (
    make_box,
    make_cone,
    make_cylinder,
    make_ellipsoid,
    make_ring_disc,
    make_sphere,
    make_torus,
)


def _meta(name: str, aliases: List[str], desc: str, features: List[str], subcat: str = "monuments") -> Dict:
    return {
        "name": name,
        "aliases": aliases,
        "category": "History, Architecture and Culture",
        "subcategory": subcat,
        "description": desc,
        "distinguishing_features": features,
        "educational_purpose": f"Visualizes the 3D archaeological, architectural, and historical significance of {name}.",
        "license": "CC0-1.0",
        "author": "Magic Mirror Spatial AI Team",
        "source": "Magic Mirror 3D Asset Library",
        "attribution": "Curated for Magic Mirror under CC0 1.0 Universal",
        "download_date": "2026-11-22",
    }


def get_models() -> List[Tuple[Model3D, Dict]]:
    models = []

    # 1. Egyptian Pyramid
    r1 = ModelNode3D("root")
    m1_base = make_cone(radius=1.2, height=1.2, segs=4, color=(120, 160, 210), roughness=0.8, name="limestone_pyramid")  # Golden limestone BGR
    m1_cap = make_cone(radius=0.3, height=0.3, segs=4, color=(40, 180, 240), metallic=0.7, name="gilded_pyramidion")     # Gold pyramidion capstone
    r1.add_child(ModelNode3D("pyramid", mesh=m1_base, rotation=np.array([0, math.pi/4, 0], dtype=np.float32)))
    r1.add_child(ModelNode3D("cap", mesh=m1_cap, position=np.array([0, 0.45, 0], dtype=np.float32), rotation=np.array([0, math.pi/4, 0], dtype=np.float32)))
    models.append((Model3D(concept="Pyramid", root=r1),
                   _meta("Pyramid", ["great pyramid", "giza pyramid", "egyptian pyramid", "khufu pyramid"],
                         "Ancient Egyptian monumental masonry tomb with square base and four sloping triangular faces meeting at gilded pyramidion capstone.",
                         ["Square monumental stone masonry foundation", "Four converging equilateral triangular sloping faces", "Gilded electrum apex pyramidion capstone"], "monuments")))

    # 2. Greek Column
    r2 = ModelNode3D("root")
    m2_shaft = make_cylinder(radius_top=0.22, radius_bottom=0.25, height=1.8, segs=20, color=(235, 238, 242), roughness=0.5, name="fluted_shaft")
    m2_cap = make_box(dx=0.6, dy=0.18, dz=0.6, color=(235, 238, 242), name="capital")
    m2_base = make_box(dx=0.65, dy=0.15, dz=0.65, color=(225, 228, 232), name="plinth_base")
    r2.add_child(ModelNode3D("shaft", mesh=m2_shaft))
    r2.add_child(ModelNode3D("cap", mesh=m2_cap, position=np.array([0, 0.95, 0], dtype=np.float32)))
    r2.add_child(ModelNode3D("base", mesh=m2_base, position=np.array([0, -0.95, 0], dtype=np.float32)))
    models.append((Model3D(concept="Column", root=r2),
                   _meta("Column", ["greek column", "corinthian column", "classical pillar", "marble column"],
                         "Classical architectural vertical structural element consisting of circular fluted marble shaft, base plinth, and decorative capital.",
                         ["Fluted vertical cylindrical marble shaft", "Classical architectural proportion with entasis taper", "Ornamented capital and square plinth base"], "classical_architecture")))

    # 3. Castle Tower
    r3 = ModelNode3D("root")
    m3_body = make_cylinder(radius_top=0.6, radius_bottom=0.65, height=1.6, color=(140, 140, 145), roughness=0.9, name="stone_tower")
    m3_crown = make_cylinder(radius_top=0.72, radius_bottom=0.72, height=0.3, segs=16, color=(130, 130, 135), name="crenellated_parapet")
    r3.add_child(ModelNode3D("body", mesh=m3_body))
    r3.add_child(ModelNode3D("crown", mesh=m3_crown, position=np.array([0, 0.9, 0], dtype=np.float32)))
    models.append((Model3D(concept="Castle Tower", root=r3),
                   _meta("Castle Tower", ["keep", "castle turret", "medieval tower", "fortress keep"],
                         "Medieval defensive stone masonry tower fortification featuring thick walls and projecting machicolated crenellated battlements.",
                         ["Heavy cylindrical stone ashlar masonry core", "Projecting corbelled machicolation crown", "Crenellated defensive parapet battlements"], "medieval_architecture")))

    # 4. Roman Amphitheatre / Colosseum
    r4 = ModelNode3D("root")
    m4_wall = make_cylinder(radius_top=1.1, radius_bottom=1.1, height=0.7, segs=28, color=(170, 165, 150), roughness=0.8, name="arcade_facade")
    m4_arena = make_cylinder(radius_top=0.75, radius_bottom=0.75, height=0.1, color=(130, 170, 200), name="sand_arena")  # Sand floor
    r4.add_child(ModelNode3D("wall", mesh=m4_wall))
    r4.add_child(ModelNode3D("arena", mesh=m4_arena, position=np.array([0, -0.25, 0], dtype=np.float32)))
    models.append((Model3D(concept="Colosseum", root=r4),
                   _meta("Colosseum", ["roman amphitheatre", "amphitheatrum flavium", "flavian amphitheatre"],
                         "Elliptical amphitheatre in center of Rome, the largest ancient amphitheatre ever built, constructed of travertine stone.",
                         ["Elliptical multi-tiered arcade monument", "Central gladiatorial arena combat floor", "Monumental Roman travertine stone arches"], "monuments")))

    # 5. Monumental Obelisk
    r5 = ModelNode3D("root")
    m5_shaft = make_cylinder(radius_top=0.14, radius_bottom=0.22, height=2.0, segs=4, color=(130, 120, 110), roughness=0.6, name="granite_shaft")
    m5_tip = make_cone(radius=0.14, height=0.3, segs=4, color=(30, 180, 240), metallic=0.8, name="gilded_pyramidion")
    r5.add_child(ModelNode3D("shaft", mesh=m5_shaft, rotation=np.array([0, math.pi/4, 0], dtype=np.float32)))
    r5.add_child(ModelNode3D("tip", mesh=m5_tip, position=np.array([0, 1.15, 0], dtype=np.float32), rotation=np.array([0, math.pi/4, 0], dtype=np.float32)))
    models.append((Model3D(concept="Obelisk", root=r5),
                   _meta("Obelisk", ["monolithic obelisk", "egyptian obelisk", "stone monument"],
                         "Tall, four-sided, narrow tapering monument which ends in a pyramid-like shape or pyramidion at the top.",
                         ["Monolithic four-sided tapering granite pillar", "Gilded summit pyramidion apex", "Ancient Egyptian monumental solar symbol"], "monuments")))

    # 6. Ancient Amphora Vase
    r6 = ModelNode3D("root")
    m6_body = make_ellipsoid(rx=0.45, ry=0.75, rz=0.45, color=(70, 110, 175), roughness=0.8, name="terracotta_body")  # Terracotta BGR
    m6_neck = make_cylinder(radius_top=0.18, radius_bottom=0.18, height=0.45, color=(70, 110, 175), name="neck")
    m6_h1 = make_torus(r_major=0.28, r_minor=0.04, color=(70, 110, 175), name="handle_1")
    m6_h2 = make_torus(r_major=0.28, r_minor=0.04, color=(70, 110, 175), name="handle_2")
    r6.add_child(ModelNode3D("body", mesh=m6_body))
    r6.add_child(ModelNode3D("neck", mesh=m6_neck, position=np.array([0, 0.85, 0], dtype=np.float32)))
    r6.add_child(ModelNode3D("h1", mesh=m6_h1, position=np.array([-0.38, 0.65, 0], dtype=np.float32), rotation=np.array([math.pi/2, 0, 0], dtype=np.float32)))
    r6.add_child(ModelNode3D("h2", mesh=m6_h2, position=np.array([0.38, 0.65, 0], dtype=np.float32), rotation=np.array([math.pi/2, 0, 0], dtype=np.float32)))
    models.append((Model3D(concept="Amphora", root=r6),
                   _meta("Amphora", ["greek amphora", "ancient vase", "terracotta vessel", "urn"],
                         "Classical ceramic container with characteristic ovoid shape and two curved vertical handles for carrying wine and olive oil.",
                         ["Ovoid tapered terracotta ceramic body", "Two vertical curved arching handles", "Slender cylindrical pouring neck"], "archaeology")))

    # 7. Megalithic Trilithon Stone Monument
    r7 = ModelNode3D("root")
    m7_up1 = make_box(dx=0.35, dy=1.5, dz=0.4, color=(140, 145, 150), roughness=0.9, name="upright_stone_l")
    m7_up2 = make_box(dx=0.35, dy=1.5, dz=0.4, color=(140, 145, 150), roughness=0.9, name="upright_stone_r")
    m7_lintel = make_box(dx=1.4, dy=0.3, dz=0.45, color=(135, 140, 145), roughness=0.9, name="horizontal_lintel")
    r7.add_child(ModelNode3D("up1", mesh=m7_up1, position=np.array([-0.4, 0, 0], dtype=np.float32)))
    r7.add_child(ModelNode3D("up2", mesh=m7_up2, position=np.array([0.4, 0, 0], dtype=np.float32)))
    r7.add_child(ModelNode3D("lintel", mesh=m7_lintel, position=np.array([0, 0.85, 0], dtype=np.float32)))
    models.append((Model3D(concept="Stone Monument", root=r7),
                   _meta("Stone Monument", ["stonehenge", "trilithon", "megalith", "standing stone", "dolmen"],
                         "Prehistoric megalithic structure consisting of two large vertical upright megaliths supporting a third horizontal stone lintel across top.",
                         ["Two vertical upright megalithic stone pillars", "Horizontal spanning lintel capstone beam", "Neolithic dressed sarsen megalith architecture"], "archaeology")))

    # 8. Acheulean Handaxe
    r8 = ModelNode3D("root")
    m8_axe = make_cone(radius=0.35, height=1.0, segs=8, color=(90, 85, 80), roughness=0.9, name="flint_handaxe")
    r8.add_child(ModelNode3D("axe", mesh=m8_axe, scale=np.array([1.0, 1.0, 0.4], dtype=np.float32)))
    models.append((Model3D(concept="Handaxe", root=r8),
                   _meta("Handaxe", ["biface", "paleolithic tool", "flint tool", "acheulean axe"],
                         "Prehistoric stone tool with two faces that is the longest-used tool in human history, bifacially flaked into teardrop shape.",
                         ["Bifacial flaked teardrop silhouette", "Sharp worked lateral cutting perimeter edges", "Flint knapped lithic texture"], "archaeology")))

    # 9. Navigational Astrolabe
    r9 = ModelNode3D("root")
    m9_mater = make_cylinder(radius_top=0.75, radius_bottom=0.75, height=0.06, color=(40, 170, 220), metallic=0.9, name="brass_mater")  # Brass BGR
    m9_ring = make_ring_disc(r_inner=0.4, r_outer=0.7, segs=24, color=(35, 155, 205), name="openwork_rete")
    m9_arm = make_box(dx=1.4, dy=0.03, dz=0.06, color=(30, 140, 190), name="alidade_ruler")
    r9.add_child(ModelNode3D("mater", mesh=m9_mater))
    r9.add_child(ModelNode3D("rete", mesh=m9_ring, position=np.array([0, 0.035, 0], dtype=np.float32)))
    r9.add_child(ModelNode3D("alidade", mesh=m9_arm, position=np.array([0, 0.06, 0], dtype=np.float32), rotation=np.array([0, 0.6, 0], dtype=np.float32)))
    models.append((Model3D(concept="Astrolabe", root=r9),
                   _meta("Astrolabe", ["planispheric astrolabe", "navigational instrument", "brass astrolabe"],
                         "Ancient astronomical instrument that was a handheld model of the universe used by astronomers and navigators.",
                         ["Circular brass graduated mater disc rim", "Rotating openwork star-pointer rete disc", "Pivoting alidade sighting rule arm"], "historical_instruments")))

    # 10. Roman Arch
    r10 = ModelNode3D("root")
    m10_pier1 = make_box(dx=0.35, dy=1.3, dz=0.5, color=(200, 200, 205), roughness=0.8, name="pier_l")
    m10_pier2 = make_box(dx=0.35, dy=1.3, dz=0.5, color=(200, 200, 205), roughness=0.8, name="pier_r")
    m10_arch = make_torus(r_major=0.55, r_minor=0.18, color=(190, 190, 195), name="voussoirs_vault")
    m10_keystone = make_box(dx=0.2, dy=0.22, dz=0.4, color=(170, 170, 175), name="keystone")
    r10.add_child(ModelNode3D("pier1", mesh=m10_pier1, position=np.array([-0.55, -0.2, 0], dtype=np.float32)))
    r10.add_child(ModelNode3D("pier2", mesh=m10_pier2, position=np.array([0.55, -0.2, 0], dtype=np.float32)))
    r10.add_child(ModelNode3D("arch", mesh=m10_arch, position=np.array([0, 0.45, 0], dtype=np.float32), rotation=np.array([math.pi/2, 0, 0], dtype=np.float32)))
    r10.add_child(ModelNode3D("keystone", mesh=m10_keystone, position=np.array([0, 1.0, 0], dtype=np.float32)))
    models.append((Model3D(concept="Arch", root=r10),
                   _meta("Arch", ["roman arch", "triumphal arch", "stone archway", "semicircular arch"],
                         "True semicircular masonry arch constructed of wedge-shaped voussoir stones culminating in a central keystone.",
                         ["Two vertical supporting stone masonry piers", "Semicircular voussoir curved arch vault", "Central locking keystone wedge at crown"], "classical_architecture")))

    return models
