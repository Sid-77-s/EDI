"""
scripts/asset_pack_builder/categories/sports.py — 15 Sports Equipment and Ball Models.
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
    make_sphere,
    make_torus,
)


def _meta(name: str, aliases: List[str], desc: str, features: List[str], subcat: str = "balls") -> Dict:
    return {
        "name": name,
        "aliases": aliases,
        "category": "Sports Equipment and Balls",
        "subcategory": subcat,
        "description": desc,
        "distinguishing_features": features,
        "educational_purpose": f"Visualizes the 3D dimensions, materials, and regulation design of {name}.",
        "license": "CC0-1.0",
        "author": "Magic Mirror Spatial AI Team",
        "source": "Magic Mirror 3D Asset Library",
        "attribution": "Curated for Magic Mirror under CC0 1.0 Universal",
        "download_date": "2026-11-22",
    }


def get_models() -> List[Tuple[Model3D, Dict]]:
    models = []

    # 1. Football / Soccer Ball
    r1 = ModelNode3D("root")
    m1_core = make_sphere(radius=0.7, color=(245, 245, 250), roughness=0.35, name="ball_white_leather")
    r1.add_child(ModelNode3D("core", mesh=m1_core))
    # 6 black pentagonal patches distributed symmetrically
    patch_dirs = [[0, 0.7, 0], [0, -0.7, 0], [0.7, 0, 0], [-0.7, 0, 0], [0, 0, 0.7], [0, 0, -0.7]]
    for idx, d in enumerate(patch_dirs):
        p = make_cylinder(radius_top=0.22, radius_bottom=0.22, height=0.04, segs=5, color=(25, 25, 25), name=f"pentagon_{idx}")
        # orient towards d
        r1.add_child(ModelNode3D(f"patch_{idx}", mesh=p, position=np.array(d, dtype=np.float32) * 0.98))
    models.append((Model3D(concept="Football", root=r1),
                   _meta("Football", ["soccer ball", "association football", "match ball", "fifa ball"],
                         "Standard association soccer ball featuring classic 32-panel truncated icosahedron white leather with black pentagons.",
                         ["Spherical leather surface with white hexagonal ground", "Distinct black pentagonal patch distribution", "Regulation soccer ball proportions"], "balls")))

    # 2. Basketball
    r2 = ModelNode3D("root")
    m2_core = make_sphere(radius=0.75, color=(20, 100, 220), roughness=0.8, name="basketball_orange_pebble")  # Bright orange BGR
    m2_seam_xy = make_torus(r_major=0.755, r_minor=0.02, color=(15, 15, 15), name="seam_xy")
    m2_seam_yz = make_torus(r_major=0.755, r_minor=0.02, color=(15, 15, 15), name="seam_yz")
    r2.add_child(ModelNode3D("core", mesh=m2_core))
    r2.add_child(ModelNode3D("seam1", mesh=m2_seam_xy))
    r2.add_child(ModelNode3D("seam2", mesh=m2_seam_yz, rotation=np.array([math.pi/2, 0, 0], dtype=np.float32)))
    models.append((Model3D(concept="Basketball", root=r2),
                   _meta("Basketball", ["nba ball", "hoop ball", "basketball sphere"],
                         "Regulation basketball with pebbled high-friction orange composite leather surface and inlaid black channel seams.",
                         ["Pebbled orange high-grip composite leather", "Contrasting inset black channel seam ribs", "Regulation spherical size 7 geometry"], "balls")))

    # 3. Tennis Ball
    r3 = ModelNode3D("root")
    m3_core = make_sphere(radius=0.55, color=(50, 230, 205), roughness=0.9, name="optic_yellow_felt")  # Fluorescent yellow-green BGR
    m3_seam = make_torus(r_major=0.555, r_minor=0.02, color=(245, 245, 250), name="white_curved_seam")
    r3.add_child(ModelNode3D("core", mesh=m3_core))
    r3.add_child(ModelNode3D("seam", mesh=m3_seam, rotation=np.array([0.5, 0.5, 0], dtype=np.float32)))
    models.append((Model3D(concept="Tennis Ball", root=r3),
                   _meta("Tennis Ball", ["tennis sphere", "felt ball", "yellow tennis ball"],
                         "Pressurized regulation tennis ball covered in fibrous optic yellow felt with distinctive curved white vulcanized rubber seam.",
                         ["Optic yellow fluorescent fibrous felt cover", "Continuous curved vulcanized white rubber seam", "Pressurized hollow core bounce sphere"], "balls")))

    # 4. Cricket Ball
    r4 = ModelNode3D("root")
    m4_core = make_sphere(radius=0.58, color=(20, 20, 150), roughness=0.4, name="crimson_leather")  # Crimson red BGR
    m4_seam = make_torus(r_major=0.585, r_minor=0.025, color=(240, 240, 240), name="stitched_string_seam")
    r4.add_child(ModelNode3D("core", mesh=m4_core))
    r4.add_child(ModelNode3D("seam", mesh=m4_seam))
    models.append((Model3D(concept="Cricket Ball", root=r4),
                   _meta("Cricket Ball", ["leather cricket ball", "red ball", "match cricket ball"],
                         "Hard solid cork core sphere covered with polished deep red leather halves joined by a prominent raised white stitched seam.",
                         ["Polished red leather hemispherical quarters", "Raised equatorially stitched string seam", "Hard regulation cork and twine core"], "balls")))

    # 5. Baseball
    r5 = ModelNode3D("root")
    m5_core = make_sphere(radius=0.56, color=(245, 245, 245), roughness=0.6, name="white_horsehide")
    m5_seam1 = make_torus(r_major=0.565, r_minor=0.018, color=(30, 30, 210), name="red_stitch_1")  # Red stitches BGR
    m5_seam2 = make_torus(r_major=0.565, r_minor=0.018, color=(30, 30, 210), name="red_stitch_2")
    r5.add_child(ModelNode3D("core", mesh=m5_core))
    r5.add_child(ModelNode3D("seam1", mesh=m5_seam1, rotation=np.array([0.4, 0.4, 0], dtype=np.float32)))
    r5.add_child(ModelNode3D("seam2", mesh=m5_seam2, rotation=np.array([-0.4, -0.4, 0], dtype=np.float32)))
    models.append((Model3D(concept="Baseball", root=r5),
                   _meta("Baseball", ["hardball", "mlb baseball", "stitched baseball"],
                         "White cowhide or horsehide sphere bound tightly with two interlocking pieces featuring 108 double raised red stitches.",
                         ["White genuine cowhide leather panels", "Double-curved raised red waxed thread stitches", "Regulation cork and rubber pill core"], "balls")))

    # 6. Golf Ball
    r6 = ModelNode3D("root")
    m6_core = make_sphere(radius=0.5, color=(250, 250, 252), roughness=0.4, name="white_ionomer_cover")
    # Multiple small dimple rings
    r6.add_child(ModelNode3D("core", mesh=m6_core))
    for ring_r in [0.2, 0.35]:
        t = make_torus(r_major=ring_r, r_minor=0.015, color=(210, 215, 220), name=f"dimple_ring_{ring_r}")
        r6.add_child(ModelNode3D(f"d_{ring_r}", mesh=t, position=np.array([0, 0, 0.5], dtype=np.float32)))
    models.append((Model3D(concept="Golf Ball", root=r6),
                   _meta("Golf Ball", ["dimpled ball", "pga golf ball", "golf sphere"],
                         "Small, hard white ball designed for golf, covered in aerodynamic dimples to reduce aerodynamic drag and optimize lift.",
                         ["Aerodynamic dimpled surface indentations", "Pure white ionomer/urethane protective cover", "Compact regulation 42.7mm diameter"], "balls")))

    # 7. Table Tennis Paddle
    r7 = ModelNode3D("root")
    m7_blade = make_cylinder(radius_top=0.55, radius_bottom=0.55, height=0.05, segs=24, color=(30, 30, 210), name="rubber_red_face")
    m7_handle = make_cylinder(radius_top=0.08, radius_bottom=0.1, height=0.6, color=(140, 110, 70), name="wooden_handle")
    r7.add_child(ModelNode3D("blade", mesh=m7_blade, rotation=np.array([math.pi/2, 0, 0], dtype=np.float32)))
    r7.add_child(ModelNode3D("handle", mesh=m7_handle, position=np.array([0, -0.65, 0], dtype=np.float32)))
    models.append((Model3D(concept="Table Tennis Paddle", root=r7),
                   _meta("Table Tennis Paddle", ["ping pong paddle", "table tennis bat", "ping pong racket"],
                         "Laminated wooden paddle with circular red rubber pimpled/smooth hitting surface and flared ergonomic grip handle.",
                         ["Circular laminated wooden blade with rubber sheet", "Vivid competition red rubber surface", "Ergonomic flared wooden grip handle"], "equipment")))

    # 8. Shuttlecock
    r8 = ModelNode3D("root")
    m8_cork = make_sphere(radius=0.2, color=(240, 240, 245), name="cork_nose")
    m8_skirt = make_cone(radius=0.45, height=0.7, segs=16, color=(250, 250, 255), name="feather_skirt")
    r8.add_child(ModelNode3D("cork", mesh=m8_cork, position=np.array([0, -0.35, 0], dtype=np.float32)))
    r8.add_child(ModelNode3D("skirt", mesh=m8_skirt, position=np.array([0, 0.1, 0], dtype=np.float32)))
    models.append((Model3D(concept="Shuttlecock", root=r8),
                   _meta("Shuttlecock", ["birdie", "badminton shuttlecock", "feather shuttle"],
                         "High-drag projectile used in badminton consisting of rounded cork nose base and flared conical skirt of 16 goose feathers.",
                         ["Hemispherical white leather-covered cork nose", "Flared aerodynamic conical feather skirt", "High drag aerodynamic orientation"], "equipment")))

    # 9. Tennis Racket
    r9 = ModelNode3D("root")
    m9_hoop = make_torus(r_major=0.55, r_minor=0.035, segs_major=28, color=(40, 40, 45), name="graphite_hoop")
    m9_shaft = make_cylinder(radius_top=0.03, radius_bottom=0.03, height=0.5, color=(40, 40, 45), name="throat_shaft")
    m9_grip = make_cylinder(radius_top=0.045, radius_bottom=0.045, height=0.55, color=(240, 240, 245), name="white_overgrip")
    r9.add_child(ModelNode3D("hoop", mesh=m9_hoop, position=np.array([0, 0.65, 0], dtype=np.float32)))
    r9.add_child(ModelNode3D("shaft", mesh=m9_shaft, position=np.array([0, 0.1, 0], dtype=np.float32)))
    r9.add_child(ModelNode3D("grip", mesh=m9_grip, position=np.array([0, -0.45, 0], dtype=np.float32)))
    models.append((Model3D(concept="Tennis Racket", root=r9),
                   _meta("Tennis Racket", ["tennis racquet", "strung racket", "graphite racket"],
                         "Strung sports implement with oval aerodynamic graphite composite frame hoop, open throat, and wrapped grip handle.",
                         ["Oval strung head hoop frame", "Open aerodynamic bifurcated throat", "Cushioned non-slip overgrip handle"], "equipment")))

    # 10. Cricket Bat
    r10 = ModelNode3D("root")
    m10_blade = make_box(dx=0.35, dy=1.2, dz=0.1, color=(150, 120, 75), roughness=0.6, name="willow_blade")
    m10_handle = make_cylinder(radius_top=0.04, radius_bottom=0.04, height=0.6, color=(240, 240, 245), name="cane_grip")
    r10.add_child(ModelNode3D("blade", mesh=m10_blade, position=np.array([0, 0.3, 0], dtype=np.float32)))
    r10.add_child(ModelNode3D("handle", mesh=m10_handle, position=np.array([0, 1.15, 0], dtype=np.float32)))
    models.append((Model3D(concept="Cricket Bat", root=r10),
                   _meta("Cricket Bat", ["willow bat", "cricket paddle", "english willow"],
                         "Wood bat consisting of cane handle attached to flat-fronted, ridge-backed English willow blade.",
                         ["Flat front striking face with rear spine ridge", "Natural English willow wood grain finish", "Round cane handle wrapped in white rubber grip"], "equipment")))

    # 11. Baseball Bat
    r11 = ModelNode3D("root")
    m11_bat = make_cylinder(radius_top=0.08, radius_bottom=0.035, height=1.5, color=(160, 130, 85), roughness=0.4, name="wooden_bat")
    m11_knob = make_cylinder(radius_top=0.055, radius_bottom=0.055, height=0.05, color=(150, 120, 80), name="handle_knob")
    r11.add_child(ModelNode3D("bat", mesh=m11_bat))
    r11.add_child(ModelNode3D("knob", mesh=m11_knob, position=np.array([0, -0.75, 0], dtype=np.float32)))
    models.append((Model3D(concept="Baseball Bat", root=r11),
                   _meta("Baseball Bat", ["wooden bat", "ash bat", "baseball club", "slugger"],
                         "Smooth tapered wooden club used in baseball to hit the ball, with thick barrel hitting zone and flared handle knob.",
                         ["Tapered cylindrical contour from thin handle to thick barrel", "Rounded end cap hitting surface", "Flared safety knob at handle base"], "equipment")))

    # 12. Hockey Stick
    r12 = ModelNode3D("root")
    m12_shaft = make_box(dx=0.06, dy=1.4, dz=0.04, color=(35, 35, 40), name="composite_shaft")
    m12_blade = make_box(dx=0.35, dy=0.08, dz=0.04, color=(35, 35, 40), name="curved_blade")
    r12.add_child(ModelNode3D("shaft", mesh=m12_shaft, position=np.array([0, 0.4, 0], dtype=np.float32)))
    r12.add_child(ModelNode3D("blade", mesh=m12_blade, position=np.array([0.15, -0.35, 0], dtype=np.float32)))
    models.append((Model3D(concept="Hockey Stick", root=r12),
                   _meta("Hockey Stick", ["field hockey stick", "ice hockey stick", "hockey shaft"],
                         "Long sporting shaft with angled lower curved blade used to drive a puck or ball into the opponent's goal.",
                         ["Long straight composite rectangular shaft", "Sharp perpendicular curved lower blade", "Grip taper at proximal butt end"], "equipment")))

    # 13. Dumbbell
    r13 = ModelNode3D("root")
    m13_bar = make_cylinder(radius_top=0.04, radius_bottom=0.04, height=0.6, color=(190, 190, 195), roughness=0.3, metallic=0.8, name="steel_bar")
    m13_plate_l = make_cylinder(radius_top=0.25, radius_bottom=0.25, height=0.15, segs=6, color=(30, 30, 35), name="hex_weight_l")
    m13_plate_r = make_cylinder(radius_top=0.25, radius_bottom=0.25, height=0.15, segs=6, color=(30, 30, 35), name="hex_weight_r")
    r13.add_child(ModelNode3D("bar", mesh=m13_bar, rotation=np.array([0, 0, math.pi/2], dtype=np.float32)))
    r13.add_child(ModelNode3D("weight_l", mesh=m13_plate_l, position=np.array([-0.35, 0, 0], dtype=np.float32), rotation=np.array([0, 0, math.pi/2], dtype=np.float32)))
    r13.add_child(ModelNode3D("weight_r", mesh=m13_plate_r, position=np.array([0.35, 0, 0], dtype=np.float32), rotation=np.array([0, 0, math.pi/2], dtype=np.float32)))
    models.append((Model3D(concept="Dumbbell", root=r13),
                   _meta("Dumbbell", ["hand weight", "free weight", "hex dumbbell", "gym weight"],
                         "Short barbell weight consisting of central knurled steel handle bar capped at each end with equal hexagonal rubber weights.",
                         ["Central knurled steel hand grip bar", "Twin hexagonal rubber-encased weight heads", "Balanced free-weight resistance design"], "gym_equipment")))

    # 14. Barbell
    r14 = ModelNode3D("root")
    m14_bar = make_cylinder(radius_top=0.035, radius_bottom=0.035, height=2.2, color=(200, 205, 215), roughness=0.2, metallic=0.9, name="olympic_bar")
    m14_disc_l = make_cylinder(radius_top=0.45, radius_bottom=0.45, height=0.1, color=(30, 30, 35), name="bumper_plate_l")
    m14_disc_r = make_cylinder(radius_top=0.45, radius_bottom=0.45, height=0.1, color=(30, 30, 35), name="bumper_plate_r")
    r14.add_child(ModelNode3D("bar", mesh=m14_bar, rotation=np.array([0, 0, math.pi/2], dtype=np.float32)))
    r14.add_child(ModelNode3D("disc_l", mesh=m14_disc_l, position=np.array([-0.85, 0, 0], dtype=np.float32), rotation=np.array([0, 0, math.pi/2], dtype=np.float32)))
    r14.add_child(ModelNode3D("disc_r", mesh=m14_disc_r, position=np.array([0.85, 0, 0], dtype=np.float32), rotation=np.array([0, 0, math.pi/2], dtype=np.float32)))
    models.append((Model3D(concept="Barbell", root=r14),
                   _meta("Barbell", ["olympic barbell", "weightlifting bar", "weight plate bar"],
                         "Full-length Olympic steel bar with rotating end sleeves holding large circular black Olympic bumper weight plates.",
                         ["Long straight knurled steel Olympic bar", "Large circular Olympic bumper weight plates", "Rotating sleeve collar assemblies"], "gym_equipment")))

    # 15. Sports Helmet
    r15 = ModelNode3D("root")
    m15_shell = make_ellipsoid(rx=0.6, ry=0.55, rz=0.65, color=(220, 40, 40), roughness=0.3, name="helmet_shell")  # Red shell
    m15_visor = make_box(dx=0.55, dy=0.12, dz=0.4, color=(20, 20, 25), name="sun_visor")
    r15.add_child(ModelNode3D("shell", mesh=m15_shell))
    r15.add_child(ModelNode3D("visor", mesh=m15_visor, position=np.array([0, 0.1, 0.5], dtype=np.float32), rotation=np.array([0.2, 0, 0], dtype=np.float32)))
    models.append((Model3D(concept="Sports Helmet", root=r15),
                   _meta("Sports Helmet", ["protective helmet", "cycling helmet", "baseball helmet", "headgear"],
                         "Hard protective impact-absorbing outer helmet shell with aerodynamic ventilation contours and front sun visor.",
                         ["Curved impact-resistant outer polycarbonate shell", "Anterior sun visor / protective brim", "Cranial protective ergonomic contour"], "equipment")))

    return models
