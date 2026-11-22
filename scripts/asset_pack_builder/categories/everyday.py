"""
scripts/asset_pack_builder/categories/everyday.py — 25 Everyday Objects and Furniture Models.
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


def _meta(name: str, aliases: List[str], desc: str, features: List[str], subcat: str = "furniture") -> Dict:
    return {
        "name": name,
        "aliases": aliases,
        "category": "Everyday Objects and Furniture",
        "subcategory": subcat,
        "description": desc,
        "distinguishing_features": features,
        "educational_purpose": f"Visualizes the 3D ergonomic design and utility of {name}.",
        "license": "CC0-1.0",
        "author": "Magic Mirror Spatial AI Team",
        "source": "Magic Mirror 3D Asset Library",
        "attribution": "Curated for Magic Mirror under CC0 1.0 Universal",
        "download_date": "2026-11-22",
    }


def get_models() -> List[Tuple[Model3D, Dict]]:
    models = []

    # 1. Chair
    r1 = ModelNode3D("root")
    m1_seat = make_box(dx=0.8, dy=0.08, dz=0.8, color=(140, 90, 50), name="seat_pan")  # Wood brown
    m1_back = make_box(dx=0.8, dy=0.8, dz=0.08, color=(140, 90, 50), name="backrest")
    r1.add_child(ModelNode3D("seat", mesh=m1_seat))
    r1.add_child(ModelNode3D("back", mesh=m1_back, position=np.array([0, 0.44, -0.36], dtype=np.float32)))
    for dx, dz in [(-0.35, 0.35), (0.35, 0.35), (-0.35, -0.35), (0.35, -0.35)]:
        leg = make_cylinder(radius_top=0.04, radius_bottom=0.03, height=0.7, color=(120, 75, 40), name=f"chair_leg_{dx}_{dz}")
        r1.add_child(ModelNode3D(f"leg_{dx}_{dz}", mesh=leg, position=np.array([dx, -0.35, dz], dtype=np.float32)))
    models.append((Model3D(concept="Chair", root=r1),
                   _meta("Chair", ["wooden chair", "dining chair", "seat"],
                         "Four-legged seating furniture with flat seat pan, supporting legs, and upright lumbar backrest.",
                         ["Horizontal supportive seat cushion pan", "Four vertical ground-contact legs", "Upright lumbar backrest support"], "furniture")))

    # 2. Table
    r2 = ModelNode3D("root")
    m2_top = make_box(dx=1.8, dy=0.08, dz=1.2, color=(160, 110, 65), roughness=0.5, name="tabletop")
    r2.add_child(ModelNode3D("top", mesh=m2_top, position=np.array([0, 0.4, 0], dtype=np.float32)))
    for dx, dz in [(-0.8, 0.5), (0.8, 0.5), (-0.8, -0.5), (0.8, -0.5)]:
        leg = make_cylinder(radius_top=0.05, radius_bottom=0.05, height=0.8, color=(130, 85, 45), name=f"tbl_leg_{dx}_{dz}")
        r2.add_child(ModelNode3D(f"leg_{dx}_{dz}", mesh=leg, position=np.array([dx, -0.04, dz], dtype=np.float32)))
    models.append((Model3D(concept="Table", root=r2),
                   _meta("Table", ["dining table", "wooden table", "desk table"],
                         "Piece of furniture with flat horizontal work or dining surface supported by four perimeter corner legs.",
                         ["Broad flat rectangular work tabletop", "Four perimeter vertical supporting legs", "Stable dining/work height proportion"], "furniture")))

    # 3. Sofa
    r3 = ModelNode3D("root")
    m3_seat = make_box(dx=1.8, dy=0.35, dz=0.8, color=(180, 100, 60), name="sofa_cushions")  # Warm fabric
    m3_back = make_box(dx=1.8, dy=0.7, dz=0.25, color=(170, 90, 55), name="sofa_back")
    m3_arm_l = make_box(dx=0.25, dy=0.55, dz=0.85, color=(160, 85, 50), name="arm_l")
    m3_arm_r = make_box(dx=0.25, dy=0.55, dz=0.85, color=(160, 85, 50), name="arm_r")
    r3.add_child(ModelNode3D("seat", mesh=m3_seat))
    r3.add_child(ModelNode3D("back", mesh=m3_back, position=np.array([0, 0.45, -0.35], dtype=np.float32)))
    r3.add_child(ModelNode3D("arm_l", mesh=m3_arm_l, position=np.array([-0.95, 0.15, 0], dtype=np.float32)))
    r3.add_child(ModelNode3D("arm_r", mesh=m3_arm_r, position=np.array([0.95, 0.15, 0], dtype=np.float32)))
    models.append((Model3D(concept="Sofa", root=r3),
                   _meta("Sofa", ["couch", "davenport", "living room sofa", "settee"],
                         "Upholstered furniture piece for seating multiple persons, with cushioned bench, high backrest, and padded side armrests.",
                         ["Upholstered multi-person seat bench cushion", "Padded upright backrest support", "Twin flanking upholstered armrests"], "furniture")))

    # 4. Bed
    r4 = ModelNode3D("root")
    m4_mattress = make_box(dx=1.4, dy=0.35, dz=2.0, color=(240, 245, 250), name="mattress")
    m4_headboard = make_box(dx=1.45, dy=0.8, dz=0.1, color=(120, 80, 50), name="headboard")
    m4_pillow = make_box(dx=0.9, dy=0.12, dz=0.4, color=(245, 250, 255), name="pillow")
    r4.add_child(ModelNode3D("mattress", mesh=m4_mattress))
    r4.add_child(ModelNode3D("headboard", mesh=m4_headboard, position=np.array([0, 0.35, -1.0], dtype=np.float32)))
    r4.add_child(ModelNode3D("pillow", mesh=m4_pillow, position=np.array([0, 0.22, -0.7], dtype=np.float32)))
    models.append((Model3D(concept="Bed", root=r4),
                   _meta("Bed", ["mattress bed", "single bed", "sleeping bed"],
                         "Piece of furniture used for sleeping, consisting of a supportive mattress, pillow, and raised wooden headboard.",
                         ["Horizontal sleep mattress box", "Elevated vertical wooden headboard", "Soft resting head pillow cushion"], "furniture")))

    # 5. Office Desk
    r5 = ModelNode3D("root")
    m5_top = make_box(dx=1.6, dy=0.06, dz=0.9, color=(80, 80, 85), name="desk_surface")
    m5_drawers = make_box(dx=0.4, dy=0.7, dz=0.8, color=(60, 60, 65), name="drawer_pedestal")
    m5_leg = make_box(dx=0.06, dy=0.7, dz=0.8, color=(40, 40, 45), name="desk_leg_left")
    r5.add_child(ModelNode3D("top", mesh=m5_top, position=np.array([0, 0.38, 0], dtype=np.float32)))
    r5.add_child(ModelNode3D("drawers", mesh=m5_drawers, position=np.array([0.55, 0, 0], dtype=np.float32)))
    r5.add_child(ModelNode3D("leg", mesh=m5_leg, position=np.array([-0.75, 0, 0], dtype=np.float32)))
    models.append((Model3D(concept="Desk", root=r5),
                   _meta("Desk", ["office desk", "workstation", "computer desk", "study desk"],
                         "Office workstation table with flat desktop work plane and integrated side drawer pedestal cabinet.",
                         ["Spacious rectangular desktop surface", "Side pedestal filing drawer unit", "Asymmetric functional workstation geometry"], "furniture")))

    # 6. Coffee Mug
    r6 = ModelNode3D("root")
    m6_cup = make_cylinder(radius_top=0.4, radius_bottom=0.38, height=0.8, color=(240, 245, 250), name="ceramic_mug")
    m6_handle = make_torus(r_major=0.28, r_minor=0.06, color=(240, 245, 250), name="d_handle")
    r6.add_child(ModelNode3D("cup", mesh=m6_cup))
    r6.add_child(ModelNode3D("handle", mesh=m6_handle, position=np.array([0.42, 0, 0], dtype=np.float32), rotation=np.array([math.pi/2, 0, 0], dtype=np.float32)))
    models.append((Model3D(concept="Coffee Mug", root=r6),
                   _meta("Coffee Mug", ["mug", "coffee cup", "tea cup", "ceramic mug"],
                         "Cylindrical ceramic vessel used for drinking hot beverages with lateral D-shaped handle loop.",
                         ["Cylindrical glazed ceramic beverage vessel", "Lateral curved D-shaped finger grip handle", "Hollow top drinking cavity"], "kitchenware")))

    # 7. Water Bottle
    r7 = ModelNode3D("root")
    m7_body = make_cylinder(radius_top=0.3, radius_bottom=0.3, height=1.1, color=(220, 140, 40), opacity=0.8, name="bottle_cylinder")  # Cyan/blue
    m7_neck = make_cone(radius=0.3, height=0.3, color=(220, 140, 40), opacity=0.8, name="bottle_neck")
    m7_cap = make_cylinder(radius_top=0.12, radius_bottom=0.12, height=0.15, color=(30, 30, 35), name="screw_cap")
    r7.add_child(ModelNode3D("body", mesh=m7_body))
    r7.add_child(ModelNode3D("neck", mesh=m7_neck, position=np.array([0, 0.7, 0], dtype=np.float32)))
    r7.add_child(ModelNode3D("cap", mesh=m7_cap, position=np.array([0, 0.9, 0], dtype=np.float32)))
    models.append((Model3D(concept="Water Bottle", root=r7),
                   _meta("Water Bottle", ["bottle", "drink bottle", "flask", "reusable bottle"],
                         "Portable cylindrical container for liquids with tapered shoulder neck and sealed threaded screw cap.",
                         ["Cylindrical liquid containment body", "Tapered shoulder narrowing to neck", "Hermetic threaded top screw cap"], "kitchenware")))

    # 8. Dinner Plate
    m8 = make_cylinder(radius_top=0.75, radius_bottom=0.6, height=0.08, segs=28, color=(245, 248, 252), roughness=0.2, name="ceramic_plate")
    models.append((Model3D(concept="Plate", root=ModelNode3D("root", mesh=m8)),
                   _meta("Plate", ["dinner plate", "dish", "ceramic plate"],
                         "Broad, mainly flat vessel on which food is served, with raised rim edge to prevent spillage.",
                         ["Circular shallow ceramic disc", "Raised concave outer rim edge", "Glazed smooth porcelain surface"], "kitchenware")))

    # 9. Soup Spoon
    r9 = ModelNode3D("root")
    m9_bowl = make_ellipsoid(rx=0.16, ry=0.06, rz=0.26, color=(220, 225, 235), roughness=0.2, metallic=0.9, name="spoon_bowl")
    m9_handle = make_cylinder(radius_top=0.02, radius_bottom=0.02, height=0.8, color=(220, 225, 235), metallic=0.9, name="spoon_handle")
    r9.add_child(ModelNode3D("bowl", mesh=m9_bowl, position=np.array([0, 0, 0.45], dtype=np.float32)))
    r9.add_child(ModelNode3D("handle", mesh=m9_handle, rotation=np.array([math.pi/2, 0, 0], dtype=np.float32)))
    models.append((Model3D(concept="Spoon", root=r9),
                   _meta("Spoon", ["soup spoon", "eating utensil", "tablespoon", "metal spoon"],
                         "Utensil consisting of a small, shallow oval bowl on a long handle, used in preparing, serving, or eating liquid food.",
                         ["Concave oval scoop bowl", "Elongated flat handle shank", "Polished stainless steel finish"], "kitchenware")))

    # 10. Dinner Fork
    r10 = ModelNode3D("root")
    m10_handle = make_cylinder(radius_top=0.025, radius_bottom=0.025, height=0.8, color=(220, 225, 235), metallic=0.9, name="fork_handle")
    m10_head = make_box(dx=0.18, dy=0.02, dz=0.25, color=(220, 225, 235), metallic=0.9, name="fork_tines")
    r10.add_child(ModelNode3D("handle", mesh=m10_handle, rotation=np.array([math.pi/2, 0, 0], dtype=np.float32)))
    r10.add_child(ModelNode3D("tines", mesh=m10_head, position=np.array([0, 0, 0.48], dtype=np.float32)))
    models.append((Model3D(concept="Fork", root=r10),
                   _meta("Fork", ["dinner fork", "table fork", "eating utensil"],
                         "Cutlery item with long handle terminating in a head that branches into several parallel prongs or tines.",
                         ["Parallel piercing tines / prongs", "Ergonomic slender handle shank", "Reflective stainless steel cutlery finish"], "kitchenware")))

    # 11. Table Knife
    r11 = ModelNode3D("root")
    m11_handle = make_box(dx=0.05, dy=0.08, dz=0.5, color=(140, 90, 50), name="knife_handle")
    m11_blade = make_box(dx=0.015, dy=0.07, dz=0.6, color=(220, 225, 235), metallic=0.9, name="knife_blade")
    r11.add_child(ModelNode3D("handle", mesh=m11_handle, position=np.array([0, 0, -0.3], dtype=np.float32)))
    r11.add_child(ModelNode3D("blade", mesh=m11_blade, position=np.array([0, 0, 0.28], dtype=np.float32)))
    models.append((Model3D(concept="Knife", root=r11),
                   _meta("Knife", ["table knife", "dinner knife", "butter knife", "cutlery knife"],
                         "Utensil with a handle and flat blade with a cutting edge used for cutting and preparing food.",
                         ["Straight steel cutting blade with bevel edge", "Solid ergonomic hand grip scales", "Balanced table cutlery proportions"], "kitchenware")))

    # 12. Smartphone
    r12 = ModelNode3D("root")
    m12_body = make_box(dx=0.55, dy=1.1, dz=0.06, color=(30, 30, 35), roughness=0.3, name="phone_chassis")
    m12_screen = make_box(dx=0.5, dy=1.02, dz=0.02, color=(10, 10, 15), roughness=0.1, name="glass_screen")
    m12_cam = make_cylinder(radius_top=0.05, radius_bottom=0.05, height=0.03, color=(50, 50, 55), name="camera_bump")
    r12.add_child(ModelNode3D("body", mesh=m12_body))
    r12.add_child(ModelNode3D("screen", mesh=m12_screen, position=np.array([0, 0, 0.03], dtype=np.float32)))
    r12.add_child(ModelNode3D("cam", mesh=m12_cam, position=np.array([0.16, 0.4, -0.035], dtype=np.float32), rotation=np.array([math.pi/2, 0, 0], dtype=np.float32)))
    models.append((Model3D(concept="Smartphone", root=r12),
                   _meta("Smartphone", ["phone", "mobile phone", "cell phone", "touchscreen phone"],
                         "Handheld personal computer with touchscreen interface, slim slab form factor, and multi-lens camera module.",
                         ["Slim rectangular rounded-corner chassis slab", "Full-face glossy touchscreen glass display", "Rear camera lens array bump"], "consumer_electronics")))

    # 13. Laptop Computer
    r13 = ModelNode3D("root")
    m13_base = make_box(dx=1.3, dy=0.06, dz=0.9, color=(160, 165, 175), roughness=0.3, metallic=0.7, name="laptop_base")
    m13_lid = make_box(dx=1.3, dy=0.85, dz=0.04, color=(160, 165, 175), roughness=0.3, metallic=0.7, name="screen_lid")
    m13_screen = make_box(dx=1.2, dy=0.75, dz=0.02, color=(15, 15, 20), roughness=0.1, name="display_panel")
    r13.add_child(ModelNode3D("base", mesh=m13_base))
    r13.add_child(ModelNode3D("lid", mesh=m13_lid, position=np.array([0, 0.45, -0.42], dtype=np.float32), rotation=np.array([-0.25, 0, 0], dtype=np.float32)))
    r13.add_child(ModelNode3D("screen", mesh=m13_screen, position=np.array([0, 0.45, -0.4], dtype=np.float32), rotation=np.array([-0.25, 0, 0], dtype=np.float32)))
    models.append((Model3D(concept="Laptop", root=r13),
                   _meta("Laptop", ["notebook computer", "laptop computer", "macbook", "pc laptop"],
                         "Portable personal computer with clamshell form factor consisting of keyboard base unit and hinged display screen.",
                         ["Clamshell folding hinge design", "Keyboard and trackpad lower base deck", "Thin upright widescreen LCD panel"], "consumer_electronics")))

    # 14. Computer Keyboard
    r14 = ModelNode3D("root")
    m14_base = make_box(dx=1.4, dy=0.05, dz=0.5, color=(40, 40, 45), name="keyboard_plate")
    m14_keys = make_box(dx=1.32, dy=0.03, dz=0.42, color=(25, 25, 30), name="keycaps")
    r14.add_child(ModelNode3D("base", mesh=m14_base))
    r14.add_child(ModelNode3D("keys", mesh=m14_keys, position=np.array([0, 0.04, 0], dtype=np.float32)))
    models.append((Model3D(concept="Keyboard", root=r14),
                   _meta("Keyboard", ["computer keyboard", "typing keyboard", "qwerty keyboard"],
                         "Typewriter-style input device which uses an arrangement of buttons or keys to act as mechanical levers or electronic switches.",
                         ["Low-profile rectangular base deck", "Grid of typewriter keycaps", "Ergonomic tilted typing plane"], "computer_peripherals")))

    # 15. Computer Mouse
    r15 = ModelNode3D("root")
    m15_body = make_ellipsoid(rx=0.28, ry=0.18, rz=0.45, color=(35, 35, 40), roughness=0.4, name="mouse_dome")
    m15_wheel = make_cylinder(radius_top=0.04, radius_bottom=0.04, height=0.03, color=(180, 180, 190), name="scroll_wheel")
    r15.add_child(ModelNode3D("body", mesh=m15_body))
    r15.add_child(ModelNode3D("wheel", mesh=m15_wheel, position=np.array([0, 0.16, 0.2], dtype=np.float32), rotation=np.array([0, 0, math.pi/2], dtype=np.float32)))
    models.append((Model3D(concept="Computer Mouse", root=r15),
                   _meta("Computer Mouse", ["mouse", "optical mouse", "wireless mouse", "track mouse"],
                         "Hand-held pointing device that detects two-dimensional motion relative to a surface, with ergonomic palm dome and scroll wheel.",
                         ["Ergonomic palm-contoured dome shell", "Split primary left/right clicking buttons", "Central tactile rotating scroll wheel"], "computer_peripherals")))

    # 16. Over-Ear Headphones
    r16 = ModelNode3D("root")
    m16_band = make_torus(r_major=0.55, r_minor=0.04, segs_major=24, color=(30, 30, 35), name="headband_arch")
    m16_cup_l = make_cylinder(radius_top=0.22, radius_bottom=0.22, height=0.15, color=(25, 25, 30), name="earcup_l")
    m16_cup_r = make_cylinder(radius_top=0.22, radius_bottom=0.22, height=0.15, color=(25, 25, 30), name="earcup_r")
    r16.add_child(ModelNode3D("band", mesh=m16_band))
    r16.add_child(ModelNode3D("cup_l", mesh=m16_cup_l, position=np.array([-0.55, -0.15, 0], dtype=np.float32), rotation=np.array([0, 0, math.pi/2], dtype=np.float32)))
    r16.add_child(ModelNode3D("cup_r", mesh=m16_cup_r, position=np.array([0.55, -0.15, 0], dtype=np.float32), rotation=np.array([0, 0, math.pi/2], dtype=np.float32)))
    models.append((Model3D(concept="Headphones", root=r16),
                   _meta("Headphones", ["headset", "earphones", "audio headphones", "cans"],
                         "Pair of small loudspeaker drivers worn on or around the head over user's ears, joined by a springy headband.",
                         ["Padded flexible cranial headband arch", "Twin circumaural acoustic earcups", "Cushioned noise-isolating earpads"], "audio_peripherals")))

    # 17. Wristwatch
    r17 = ModelNode3D("root")
    m17_case = make_cylinder(radius_top=0.35, radius_bottom=0.35, height=0.1, color=(200, 205, 215), roughness=0.2, metallic=0.9, name="watch_case")
    m17_strap1 = make_box(dx=0.28, dy=0.04, dz=0.6, color=(40, 30, 25), name="leather_strap_top")
    m17_strap2 = make_box(dx=0.28, dy=0.04, dz=0.6, color=(40, 30, 25), name="leather_strap_bot")
    r17.add_child(ModelNode3D("case", mesh=m17_case))
    r17.add_child(ModelNode3D("s1", mesh=m17_strap1, position=np.array([0, 0, 0.6], dtype=np.float32)))
    r17.add_child(ModelNode3D("s2", mesh=m17_strap2, position=np.array([0, 0, -0.6], dtype=np.float32)))
    models.append((Model3D(concept="Watch", root=r17),
                   _meta("Watch", ["wristwatch", "timepiece", "chronometer", "wrist watch"],
                         "Timepiece intended to be carried or worn by a person on wrist, with circular dial case, bezel, and leather strap bands.",
                         ["Circular metallic watchcase with dial face", "Twin flexible leather wrist attachment straps", "Lateral winding/adjustment crown knob"], "accessories")))

    # 18. Backpack
    r18 = ModelNode3D("root")
    m18_main = make_box(dx=0.8, dy=1.1, dz=0.5, color=(40, 90, 160), roughness=0.8, name="main_pack")  # Blue canvas
    m18_pocket = make_box(dx=0.65, dy=0.55, dz=0.2, color=(30, 75, 140), name="front_pocket")
    r18.add_child(ModelNode3D("main", mesh=m18_main))
    r18.add_child(ModelNode3D("pocket", mesh=m18_pocket, position=np.array([0, -0.2, 0.32], dtype=np.float32)))
    models.append((Model3D(concept="Backpack", root=r18),
                   _meta("Backpack", ["rucksack", "knapsack", "bookbag", "school bag"],
                         "Cloth sack carried on one's back and secured with two straps that go over shoulders, with zippered front compartments.",
                         ["Voluminous zippered main canvas compartment", "Protruding front utility storage pocket", "Ergonomic dual shoulder harness orientation"], "travel_accessories")))

    # 19. Hardcover Book
    r19 = ModelNode3D("root")
    m19_pages = make_box(dx=0.7, dy=0.9, dz=0.18, color=(245, 245, 240), name="page_block")
    m19_cover = make_box(dx=0.74, dy=0.94, dz=0.22, color=(40, 50, 150), name="hardcover_spine")  # Blue book
    r19.add_child(ModelNode3D("cover", mesh=m19_cover))
    r19.add_child(ModelNode3D("pages", mesh=m19_pages, position=np.array([0.02, 0, 0], dtype=np.float32)))
    models.append((Model3D(concept="Book", root=r19),
                   _meta("Book", ["hardcover book", "novel", "textbook", "bound volume"],
                         "Bound collection of printed paper pages encased within protective rigid hardcover binding with squared spine.",
                         ["Stiff protective cloth/board hardcover binding", "Squared bound paper leaf edge block", "Distinct longitudinal binding spine"], "stationery")))

    # 20. Ballpoint Pen
    r20 = ModelNode3D("root")
    m20_shaft = make_cylinder(radius_top=0.035, radius_bottom=0.035, height=1.2, color=(20, 20, 180), name="pen_barrel")  # Blue pen
    m20_tip = make_cone(radius=0.035, height=0.15, color=(200, 205, 215), metallic=0.9, name="writing_tip")
    m20_clip = make_box(dx=0.015, dy=0.35, dz=0.04, color=(200, 205, 215), metallic=0.9, name="pocket_clip")
    r20.add_child(ModelNode3D("shaft", mesh=m20_shaft))
    r20.add_child(ModelNode3D("tip", mesh=m20_tip, position=np.array([0, -0.65, 0], dtype=np.float32)))
    r20.add_child(ModelNode3D("clip", mesh=m20_clip, position=np.array([0, 0.4, 0.05], dtype=np.float32)))
    models.append((Model3D(concept="Pen", root=r20),
                   _meta("Pen", ["ballpoint pen", "biro", "ink pen", "writing pen"],
                         "Writing instrument that dispenses ink over a metal ball at its point, with slender cylindrical barrel and pocket clip.",
                         ["Slender cylindrical plastic/metal grip barrel", "Tapered metallic writing point nib", "Top pocket retention spring clip"], "stationery")))

    # 21. Desk Lamp
    r21 = ModelNode3D("root")
    m21_base = make_cylinder(radius_top=0.35, radius_bottom=0.35, height=0.06, color=(40, 40, 45), name="lamp_base")
    m21_stem = make_cylinder(radius_top=0.025, radius_bottom=0.025, height=0.9, color=(180, 185, 195), name="gooseneck_stem")
    m21_shade = make_cone(radius=0.3, height=0.35, color=(40, 40, 45), name="lamp_shade")
    r21.add_child(ModelNode3D("base", mesh=m21_base))
    r21.add_child(ModelNode3D("stem", mesh=m21_stem, position=np.array([0, 0.45, 0], dtype=np.float32)))
    r21.add_child(ModelNode3D("shade", mesh=m21_shade, position=np.array([0, 0.95, 0.1], dtype=np.float32), rotation=np.array([0.4, 0, 0], dtype=np.float32)))
    models.append((Model3D(concept="Desk Lamp", root=r21),
                   _meta("Desk Lamp", ["table lamp", "task light", "reading lamp", "study light"],
                         "Adjustable directional luminaire with weighted circular base, articulated neck column, and conical reflector shade.",
                         ["Stable weighted circular counterbase", "Articulated elevating support gooseneck column", "Conical downward directional reflector hood"], "lighting")))

    # 22. Electric Desk Fan
    r22 = ModelNode3D("root")
    m22_base = make_cylinder(radius_top=0.32, radius_bottom=0.32, height=0.08, color=(240, 245, 250), name="fan_base")
    m22_neck = make_cylinder(radius_top=0.04, radius_bottom=0.04, height=0.6, color=(240, 245, 250), name="fan_neck")
    m22_cage = make_torus(r_major=0.45, r_minor=0.04, color=(220, 225, 235), name="wire_cage_rim")
    m22_blade = make_box(dx=0.75, dy=0.02, dz=0.15, color=(190, 210, 230), name="fan_blades")
    r22.add_child(ModelNode3D("base", mesh=m22_base))
    r22.add_child(ModelNode3D("neck", mesh=m22_neck, position=np.array([0, 0.35, 0], dtype=np.float32)))
    r22.add_child(ModelNode3D("cage", mesh=m22_cage, position=np.array([0, 0.8, 0], dtype=np.float32), rotation=np.array([math.pi/2, 0, 0], dtype=np.float32)))
    r22.add_child(ModelNode3D("blade", mesh=m22_blade, position=np.array([0, 0.8, 0], dtype=np.float32)))
    models.append((Model3D(concept="Fan", root=r22),
                   _meta("Fan", ["desk fan", "table fan", "cooling fan", "electric fan"],
                         "Powered machine used to create flow of air for cooling, with circular safety guard cage and rotating impeller blades.",
                         ["Circular protective wire guard grille", "Internal rotating multi-fin airflow impeller", "Elevated tabletop pedestal stand"], "appliances")))

    # 23. Wall Clock
    r23 = ModelNode3D("root")
    m23_dial = make_cylinder(radius_top=0.6, radius_bottom=0.6, height=0.06, segs=28, color=(245, 248, 252), name="clock_dial")
    m23_rim = make_torus(r_major=0.62, r_minor=0.04, color=(30, 30, 35), name="outer_bezel")
    m23_h_hand = make_box(dx=0.04, dy=0.25, dz=0.02, color=(15, 15, 15), name="hour_hand")
    m23_m_hand = make_box(dx=0.03, dy=0.42, dz=0.02, color=(15, 15, 15), name="minute_hand")
    r23.add_child(ModelNode3D("dial", mesh=m23_dial, rotation=np.array([math.pi/2, 0, 0], dtype=np.float32)))
    r23.add_child(ModelNode3D("rim", mesh=m23_rim))
    r23.add_child(ModelNode3D("h_hand", mesh=m23_h_hand, position=np.array([0.08, 0.08, 0.04], dtype=np.float32), rotation=np.array([0, 0, -0.6], dtype=np.float32)))
    r23.add_child(ModelNode3D("m_hand", mesh=m23_m_hand, position=np.array([0, 0.18, 0.04], dtype=np.float32)))
    models.append((Model3D(concept="Clock", root=r23),
                   _meta("Clock", ["wall clock", "analog clock", "timepiece", "dial clock"],
                         "Instrument used to measure and indicate time with circular numbered analog dial face and rotating hour/minute hands.",
                         ["Circular analog dial face with contrast bezel", "Coaxial rotating hour and minute indicator hands", "Wall-mounted chronometric proportion"], "appliances")))

    # 24. Plant Pot
    r24 = ModelNode3D("root")
    m24_pot = make_cylinder(radius_top=0.45, radius_bottom=0.32, height=0.65, color=(60, 95, 170), roughness=0.8, name="terracotta_pot")  # Terracotta BGR
    m24_soil = make_cylinder(radius_top=0.42, radius_bottom=0.42, height=0.06, color=(30, 40, 50), name="potting_soil")
    m24_stem = make_cylinder(radius_top=0.025, radius_bottom=0.025, height=0.4, color=(40, 160, 50), name="green_shoot")
    r24.add_child(ModelNode3D("pot", mesh=m24_pot))
    r24.add_child(ModelNode3D("soil", mesh=m24_soil, position=np.array([0, 0.3, 0], dtype=np.float32)))
    r24.add_child(ModelNode3D("stem", mesh=m24_stem, position=np.array([0, 0.5, 0], dtype=np.float32)))
    models.append((Model3D(concept="Plant Pot", root=r24),
                   _meta("Plant Pot", ["flowerpot", "planter", "indoor plant", "potted plant"],
                         "Container in which flowers and other plants are cultivated and displayed, made of terracotta with soil and shoot.",
                         ["Tapered conical terracotta pottery body", "Dark interior nutrient soil bed surface", "Emerging central photosynthetic green sprout"], "home_decor")))

    # 25. Umbrella
    r25 = ModelNode3D("root")
    m25_canopy = make_cone(radius=0.85, height=0.35, segs=16, color=(35, 35, 40), roughness=0.5, name="umbrella_canopy")
    m25_shaft = make_cylinder(radius_top=0.02, radius_bottom=0.02, height=1.1, color=(180, 185, 195), name="shaft")
    m25_handle = make_torus(r_major=0.1, r_minor=0.025, color=(110, 75, 40), name="j_hook_handle")
    r25.add_child(ModelNode3D("canopy", mesh=m25_canopy, position=np.array([0, 0.4, 0], dtype=np.float32)))
    r25.add_child(ModelNode3D("shaft", mesh=m25_shaft, position=np.array([0, -0.1, 0], dtype=np.float32)))
    r25.add_child(ModelNode3D("handle", mesh=m25_handle, position=np.array([0.08, -0.65, 0], dtype=np.float32)))
    models.append((Model3D(concept="Umbrella", root=r25),
                   _meta("Umbrella", ["parasol", "rain umbrella", "brolly"],
                         "Collapsible canopy designed to protect against rain or sunlight, supported by ribs on a central shaft with curved hook handle.",
                         ["Domed protective waterproof fabric canopy", "Central supporting metal runner shaft", "Curved ergonomic J-hook hand grip handle"], "accessories")))

    return models
