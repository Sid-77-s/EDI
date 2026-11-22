"""
scripts/asset_pack_builder/categories/animals.py — 20 Animal and Wildlife Models.
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
)


def _meta(name: str, aliases: List[str], desc: str, features: List[str], subcat: str = "mammals") -> Dict:
    return {
        "name": name,
        "aliases": aliases,
        "category": "Animals and Wildlife",
        "subcategory": subcat,
        "description": desc,
        "distinguishing_features": features,
        "educational_purpose": f"Visualizes the 3D biological morphology and physical characteristics of {name}.",
        "license": "CC0-1.0",
        "author": "Magic Mirror Spatial AI Team",
        "source": "Magic Mirror 3D Asset Library",
        "attribution": "Curated for Magic Mirror under CC0 1.0 Universal",
        "download_date": "2026-11-22",
    }


def get_models() -> List[Tuple[Model3D, Dict]]:
    models = []

    # 1. Dog
    r1 = ModelNode3D("root")
    m1_body = make_ellipsoid(rx=0.35, ry=0.35, rz=0.7, color=(60, 110, 160), name="dog_body")  # Golden brown BGR
    m1_head = make_sphere(radius=0.25, color=(60, 110, 160), name="dog_head")
    m1_snout = make_box(dx=0.16, dy=0.14, dz=0.22, color=(40, 80, 120), name="dog_snout")
    m1_tail = make_cylinder(radius_top=0.03, radius_bottom=0.05, height=0.45, color=(60, 110, 160), name="dog_tail")
    r1.add_child(ModelNode3D("body", mesh=m1_body))
    r1.add_child(ModelNode3D("head", mesh=m1_head, position=np.array([0, 0.35, 0.65], dtype=np.float32)))
    r1.add_child(ModelNode3D("snout", mesh=m1_snout, position=np.array([0, 0.28, 0.88], dtype=np.float32)))
    r1.add_child(ModelNode3D("tail", mesh=m1_tail, position=np.array([0, 0.25, -0.75], dtype=np.float32), rotation=np.array([0.6, 0, 0], dtype=np.float32)))
    for dx, dz in [(-0.22, 0.4), (0.22, 0.4), (-0.22, -0.4), (0.22, -0.4)]:
        leg = make_cylinder(radius_top=0.06, radius_bottom=0.05, height=0.55, color=(50, 95, 140), name=f"leg_{dx}_{dz}")
        r1.add_child(ModelNode3D(f"leg_{dx}_{dz}", mesh=leg, position=np.array([dx, -0.4, dz], dtype=np.float32)))
    models.append((Model3D(concept="Dog", root=r1),
                   _meta("Dog", ["domestic dog", "canine", "canis familiaris", "puppy"],
                         "Domesticated carnivorous quadruped mammal with characteristic muzzle snout, erect/floppy ears, and tail.",
                         ["Quadruped mammalian anatomy", "Elongated canine snout and jaw", "Upright wagging tail"], "mammals")))

    # 2. Cat
    r2 = ModelNode3D("root")
    m2_body = make_ellipsoid(rx=0.28, ry=0.28, rz=0.55, color=(120, 120, 125), name="cat_body")  # Grey feline
    m2_head = make_sphere(radius=0.22, color=(120, 120, 125), name="cat_head")
    m2_ear_l = make_cone(radius=0.08, height=0.15, color=(100, 100, 105), name="ear_l")
    m2_ear_r = make_cone(radius=0.08, height=0.15, color=(100, 100, 105), name="ear_r")
    m2_tail = make_cylinder(radius_top=0.03, radius_bottom=0.03, height=0.5, color=(120, 120, 125), name="cat_tail")
    r2.add_child(ModelNode3D("body", mesh=m2_body))
    r2.add_child(ModelNode3D("head", mesh=m2_head, position=np.array([0, 0.3, 0.52], dtype=np.float32)))
    r2.add_child(ModelNode3D("ear_l", mesh=m2_ear_l, position=np.array([-0.12, 0.52, 0.52], dtype=np.float32)))
    r2.add_child(ModelNode3D("ear_r", mesh=m2_ear_r, position=np.array([0.12, 0.52, 0.52], dtype=np.float32)))
    r2.add_child(ModelNode3D("tail", mesh=m2_tail, position=np.array([0, 0.22, -0.65], dtype=np.float32), rotation=np.array([0.8, 0, 0], dtype=np.float32)))
    for dx, dz in [(-0.18, 0.3), (0.18, 0.3), (-0.18, -0.3), (0.18, -0.3)]:
        leg = make_cylinder(radius_top=0.05, radius_bottom=0.04, height=0.45, color=(110, 110, 115), name=f"leg_{dx}_{dz}")
        r2.add_child(ModelNode3D(f"leg_{dx}_{dz}", mesh=leg, position=np.array([dx, -0.32, dz], dtype=np.float32)))
    models.append((Model3D(concept="Cat", root=r2),
                   _meta("Cat", ["domestic cat", "feline", "felis catus", "kitten"],
                         "Small carnivorous mammal with sleek body, upright pointed triangular ears, and long curved balancing tail.",
                         ["Compact feline quadruped body", "Upright pointed triangular ears", "Curved balancing caudal tail"], "mammals")))

    # 3. Horse
    r3 = ModelNode3D("root")
    m3_body = make_ellipsoid(rx=0.38, ry=0.42, rz=0.85, color=(40, 65, 110), name="horse_body")  # Chestnut brown BGR
    m3_neck = make_cylinder(radius_top=0.18, radius_bottom=0.25, height=0.7, color=(40, 65, 110), name="horse_neck")
    m3_head = make_ellipsoid(rx=0.16, ry=0.2, rz=0.35, color=(40, 65, 110), name="horse_head")
    r3.add_child(ModelNode3D("body", mesh=m3_body))
    r3.add_child(ModelNode3D("neck", mesh=m3_neck, position=np.array([0, 0.45, 0.7], dtype=np.float32), rotation=np.array([0.6, 0, 0], dtype=np.float32)))
    r3.add_child(ModelNode3D("head", mesh=m3_head, position=np.array([0, 0.8, 0.95], dtype=np.float32), rotation=np.array([0.3, 0, 0], dtype=np.float32)))
    for dx, dz in [(-0.25, 0.55), (0.25, 0.55), (-0.25, -0.55), (0.25, -0.55)]:
        leg = make_cylinder(radius_top=0.07, radius_bottom=0.05, height=0.9, color=(35, 55, 95), name=f"leg_{dx}_{dz}")
        r3.add_child(ModelNode3D(f"leg_{dx}_{dz}", mesh=leg, position=np.array([dx, -0.6, dz], dtype=np.float32)))
    models.append((Model3D(concept="Horse", root=r3),
                   _meta("Horse", ["equine", "stallion", "equus caballus", "mare"],
                         "Large, single-toed, ungulate mammal with long muscular legs, arched neck, and elongated equine skull.",
                         ["Equine arched neck carriage", "Slender athletic ungulate legs", "Elongated equine head and muzzle"], "mammals")))

    # 4. Cow
    r4 = ModelNode3D("root")
    m4_body = make_box(dx=0.75, dy=0.7, dz=1.4, color=(235, 235, 235), name="cow_body")  # White base
    m4_spot = make_box(dx=0.78, dy=0.45, dz=0.6, color=(30, 30, 30), name="cow_black_patch")
    m4_head = make_box(dx=0.4, dy=0.4, dz=0.45, color=(230, 230, 230), name="cow_head")
    m4_horn1 = make_cone(radius=0.06, height=0.22, color=(180, 180, 190), name="horn_l")
    m4_horn2 = make_cone(radius=0.06, height=0.22, color=(180, 180, 190), name="horn_r")
    r4.add_child(ModelNode3D("body", mesh=m4_body))
    r4.add_child(ModelNode3D("spot", mesh=m4_spot, position=np.array([0, 0.1, 0.1], dtype=np.float32)))
    r4.add_child(ModelNode3D("head", mesh=m4_head, position=np.array([0, 0.25, 0.85], dtype=np.float32)))
    r4.add_child(ModelNode3D("horn_l", mesh=m4_horn1, position=np.array([-0.22, 0.5, 0.85], dtype=np.float32), rotation=np.array([0, 0, 0.5], dtype=np.float32)))
    r4.add_child(ModelNode3D("horn_r", mesh=m4_horn2, position=np.array([0.22, 0.5, 0.85], dtype=np.float32), rotation=np.array([0, 0, -0.5], dtype=np.float32)))
    for dx, dz in [(-0.3, 0.45), (0.3, 0.45), (-0.3, -0.45), (0.3, -0.45)]:
        leg = make_cylinder(radius_top=0.08, radius_bottom=0.07, height=0.7, color=(220, 220, 220), name=f"leg_{dx}_{dz}")
        r4.add_child(ModelNode3D(f"leg_{dx}_{dz}", mesh=leg, position=np.array([dx, -0.55, dz], dtype=np.float32)))
    models.append((Model3D(concept="Cow", root=r4),
                   _meta("Cow", ["bovine", "cattle", "bos taurus", "dairy cow"],
                         "Large domesticated bovine herbivore with distinctive piebald black-and-white patched coat and horns.",
                         ["Heavy barrel torso with black-and-white patch markings", "Bilateral cranial horns", "Sturdy columnar limbs"], "mammals")))

    # 5. Lion
    r5 = ModelNode3D("root")
    m5_body = make_ellipsoid(rx=0.38, ry=0.4, rz=0.85, color=(45, 110, 180), name="lion_body")  # Tawny gold BGR
    m5_mane = make_sphere(radius=0.42, color=(30, 60, 110), roughness=0.9, name="lion_mane")    # Dark mane
    m5_head = make_sphere(radius=0.26, color=(45, 110, 180), name="lion_face")
    r5.add_child(ModelNode3D("body", mesh=m5_body))
    r5.add_child(ModelNode3D("mane", mesh=m5_mane, position=np.array([0, 0.35, 0.65], dtype=np.float32)))
    r5.add_child(ModelNode3D("head", mesh=m5_head, position=np.array([0, 0.35, 0.82], dtype=np.float32)))
    for dx, dz in [(-0.25, 0.45), (0.25, 0.45), (-0.25, -0.45), (0.25, -0.45)]:
        leg = make_cylinder(radius_top=0.08, radius_bottom=0.06, height=0.65, color=(40, 100, 165), name=f"leg_{dx}_{dz}")
        r5.add_child(ModelNode3D(f"leg_{dx}_{dz}", mesh=leg, position=np.array([dx, -0.45, dz], dtype=np.float32)))
    models.append((Model3D(concept="Lion", root=r5),
                   _meta("Lion", ["panthera leo", "king of beasts", "male lion", "big cat"],
                         "Apex feline predator of the savanna with muscular build and prominent magnificent male mane encircling head.",
                         ["Prominent dark masculine facial mane ruff", "Tawny golden coat coloration", "Powerful predatory feline torso"], "mammals")))

    # 6. Tiger
    r6 = ModelNode3D("root")
    m6_body = make_ellipsoid(rx=0.38, ry=0.4, rz=0.85, color=(30, 95, 215), name="tiger_body")  # Bright orange BGR
    m6_stripe = make_box(dx=0.8, dy=0.42, dz=0.1, color=(25, 25, 25), name="tiger_stripe_band")
    m6_head = make_sphere(radius=0.26, color=(30, 95, 215), name="tiger_head")
    r6.add_child(ModelNode3D("body", mesh=m6_body))
    r6.add_child(ModelNode3D("stripe", mesh=m6_stripe, position=np.array([0, 0.05, 0.1], dtype=np.float32)))
    r6.add_child(ModelNode3D("head", mesh=m6_head, position=np.array([0, 0.35, 0.75], dtype=np.float32)))
    for dx, dz in [(-0.25, 0.45), (0.25, 0.45), (-0.25, -0.45), (0.25, -0.45)]:
        leg = make_cylinder(radius_top=0.08, radius_bottom=0.06, height=0.65, color=(28, 85, 195), name=f"leg_{dx}_{dz}")
        r6.add_child(ModelNode3D(f"leg_{dx}_{dz}", mesh=leg, position=np.array([dx, -0.45, dz], dtype=np.float32)))
    models.append((Model3D(concept="Tiger", root=r6),
                   _meta("Tiger", ["panthera tigris", "bengal tiger", "siberian tiger"],
                         "Largest extant feline species characterized by reddish-orange coat with dark vertical predatory camouflage stripes.",
                         ["Reddish-orange ground coloration", "Transverse black camouflage stripes", "Muscular apex predator frame"], "mammals")))

    # 7. Elephant
    r7 = ModelNode3D("root")
    m7_body = make_ellipsoid(rx=0.6, ry=0.65, rz=1.0, color=(130, 130, 135), roughness=0.8, name="elephant_body")
    m7_head = make_sphere(radius=0.45, color=(130, 130, 135), name="elephant_head")
    m7_trunk = make_cylinder(radius_top=0.14, radius_bottom=0.07, height=0.9, color=(125, 125, 130), name="trunk")
    m7_tusk1 = make_cone(radius=0.06, height=0.5, color=(235, 240, 245), name="tusk_l")
    m7_tusk2 = make_cone(radius=0.06, height=0.5, color=(235, 240, 245), name="tusk_r")
    r7.add_child(ModelNode3D("body", mesh=m7_body))
    r7.add_child(ModelNode3D("head", mesh=m7_head, position=np.array([0, 0.35, 0.95], dtype=np.float32)))
    r7.add_child(ModelNode3D("trunk", mesh=m7_trunk, position=np.array([0, -0.15, 1.3], dtype=np.float32), rotation=np.array([-0.3, 0, 0], dtype=np.float32)))
    r7.add_child(ModelNode3D("tusk_l", mesh=m7_tusk1, position=np.array([-0.2, 0.1, 1.25], dtype=np.float32), rotation=np.array([0.5, 0, 0.2], dtype=np.float32)))
    r7.add_child(ModelNode3D("tusk_r", mesh=m7_tusk2, position=np.array([0.2, 0.1, 1.25], dtype=np.float32), rotation=np.array([0.5, 0, -0.2], dtype=np.float32)))
    for dx, dz in [(-0.38, 0.5), (0.38, 0.5), (-0.38, -0.5), (0.38, -0.5)]:
        leg = make_cylinder(radius_top=0.15, radius_bottom=0.13, height=0.85, color=(120, 120, 125), name=f"leg_{dx}_{dz}")
        r7.add_child(ModelNode3D(f"leg_{dx}_{dz}", mesh=leg, position=np.array([dx, -0.65, dz], dtype=np.float32)))
    models.append((Model3D(concept="Elephant", root=r7),
                   _meta("Elephant", ["pachyderm", "loxodonta", "african elephant", "asian elephant"],
                         "Largest living terrestrial animal with massive grey body, long prehensile proboscis trunk, and ivory tusks.",
                         ["Prehensile cranial proboscis trunk", "Bilateral curved ivory tusks", "Massive columnar pillar legs"], "mammals")))

    # 8. Giraffe
    r8 = ModelNode3D("root")
    m8_body = make_ellipsoid(rx=0.35, ry=0.4, rz=0.7, color=(40, 130, 200), name="giraffe_body")  # Yellow-tan BGR
    m8_neck = make_cylinder(radius_top=0.12, radius_bottom=0.18, height=1.4, color=(40, 130, 200), name="neck")
    m8_head = make_sphere(radius=0.18, color=(40, 130, 200), name="giraffe_head")
    r8.add_child(ModelNode3D("body", mesh=m8_body, position=np.array([0, 0.2, 0], dtype=np.float32)))
    r8.add_child(ModelNode3D("neck", mesh=m8_neck, position=np.array([0, 0.95, 0.4], dtype=np.float32), rotation=np.array([0.2, 0, 0], dtype=np.float32)))
    r8.add_child(ModelNode3D("head", mesh=m8_head, position=np.array([0, 1.7, 0.55], dtype=np.float32)))
    for dx, dz in [(-0.22, 0.35), (0.22, 0.35), (-0.22, -0.35), (0.22, -0.35)]:
        leg = make_cylinder(radius_top=0.06, radius_bottom=0.045, height=1.2, color=(35, 115, 180), name=f"leg_{dx}_{dz}")
        r8.add_child(ModelNode3D(f"leg_{dx}_{dz}", mesh=leg, position=np.array([dx, -0.6, dz], dtype=np.float32)))
    models.append((Model3D(concept="Giraffe", root=r8),
                   _meta("Giraffe", ["giraffa camelopardalis", "tallest mammal", "spotted giraffe"],
                         "Tallest living terrestrial animal characterized by extraordinarily long cervical neck, tall legs, and coat patterning.",
                         ["Extraordinarily elongated cervical neck", "Tall slender stilt-like limbs", "Reticulated coat color pattern"], "mammals")))

    # 9. Zebra
    r9 = ModelNode3D("root")
    m9_body = make_ellipsoid(rx=0.38, ry=0.42, rz=0.85, color=(240, 240, 240), name="zebra_body")
    m9_stripe = make_box(dx=0.8, dy=0.45, dz=0.12, color=(20, 20, 20), name="stripe_bar")
    m9_head = make_ellipsoid(rx=0.16, ry=0.2, rz=0.35, color=(235, 235, 235), name="zebra_head")
    r9.add_child(ModelNode3D("body", mesh=m9_body))
    r9.add_child(ModelNode3D("stripe", mesh=m9_stripe, position=np.array([0, 0.05, 0], dtype=np.float32)))
    r9.add_child(ModelNode3D("head", mesh=m9_head, position=np.array([0, 0.65, 0.85], dtype=np.float32), rotation=np.array([0.3, 0, 0], dtype=np.float32)))
    for dx, dz in [(-0.25, 0.45), (0.25, 0.45), (-0.25, -0.45), (0.25, -0.45)]:
        leg = make_cylinder(radius_top=0.07, radius_bottom=0.05, height=0.8, color=(220, 220, 220), name=f"leg_{dx}_{dz}")
        r9.add_child(ModelNode3D(f"leg_{dx}_{dz}", mesh=leg, position=np.array([dx, -0.55, dz], dtype=np.float32)))
    models.append((Model3D(concept="Zebra", root=r9),
                   _meta("Zebra", ["equus quagga", "wild horse", "striped zebra"],
                         "African wild equids famous for distinctive black-and-white vertical striped camouflage patterns.",
                         ["Striking black and white striped patterning", "Equid quadruped morphology", "Erect dorsal mane ridge"], "mammals")))

    # 10. Bear
    r10 = ModelNode3D("root")
    m10_body = make_ellipsoid(rx=0.5, ry=0.5, rz=0.8, color=(25, 45, 75), roughness=0.85, name="bear_body")  # Dark brown BGR
    m10_head = make_sphere(radius=0.32, color=(25, 45, 75), name="bear_head")
    m10_snout = make_box(dx=0.2, dy=0.16, dz=0.25, color=(35, 60, 95), name="bear_snout")
    r10.add_child(ModelNode3D("body", mesh=m10_body))
    r10.add_child(ModelNode3D("head", mesh=m10_head, position=np.array([0, 0.35, 0.75], dtype=np.float32)))
    r10.add_child(ModelNode3D("snout", mesh=m10_snout, position=np.array([0, 0.28, 0.98], dtype=np.float32)))
    for dx, dz in [(-0.3, 0.45), (0.3, 0.45), (-0.3, -0.45), (0.3, -0.45)]:
        leg = make_cylinder(radius_top=0.12, radius_bottom=0.1, height=0.6, color=(22, 40, 70), name=f"leg_{dx}_{dz}")
        r10.add_child(ModelNode3D(f"leg_{dx}_{dz}", mesh=leg, position=np.array([dx, -0.45, dz], dtype=np.float32)))
    models.append((Model3D(concept="Bear", root=r10),
                   _meta("Bear", ["grizzly bear", "brown bear", "ursus arctos", "ursine"],
                         "Heavy ursid carnivoran with stocky physique, heavy brown fur, non-retractile claws, and broad snout.",
                         ["Stocky robust muscular barrel torso", "Broad round ursid head and snout", "Stout columnar plantigrade limbs"], "mammals")))

    # 11. Penguin
    r11 = ModelNode3D("root")
    m11_body = make_cylinder(radius_top=0.3, radius_bottom=0.4, height=1.0, color=(30, 30, 35), name="penguin_back")
    m11_belly = make_ellipsoid(rx=0.26, ry=0.42, rz=0.15, color=(245, 245, 250), name="penguin_white_belly")
    m11_beak = make_cone(radius=0.06, height=0.25, color=(20, 140, 240), name="beak")  # Orange beak
    m11_flipper_l = make_box(dx=0.06, dy=0.6, dz=0.18, color=(30, 30, 35), name="flipper_l")
    m11_flipper_r = make_box(dx=0.06, dy=0.6, dz=0.18, color=(30, 30, 35), name="flipper_r")
    r11.add_child(ModelNode3D("body", mesh=m11_body))
    r11.add_child(ModelNode3D("belly", mesh=m11_belly, position=np.array([0, -0.05, 0.32], dtype=np.float32)))
    r11.add_child(ModelNode3D("beak", mesh=m11_beak, position=np.array([0, 0.45, 0.4], dtype=np.float32), rotation=np.array([math.pi/2, 0, 0], dtype=np.float32)))
    r11.add_child(ModelNode3D("flipper_l", mesh=m11_flipper_l, position=np.array([-0.36, 0.05, 0], dtype=np.float32), rotation=np.array([0, 0, -0.2], dtype=np.float32)))
    r11.add_child(ModelNode3D("flipper_r", mesh=m11_flipper_r, position=np.array([0.36, 0.05, 0], dtype=np.float32), rotation=np.array([0, 0, 0.2], dtype=np.float32)))
    models.append((Model3D(concept="Penguin", root=r11),
                   _meta("Penguin", ["emperor penguin", "spheniscidae", "antarctic penguin", "seabird"],
                         "Upright aquatic flightless bird with countershaded white breast and black plumage, flipper wings, and tapered beak.",
                         ["Upright bipedal tuxedo countershading", "Rigid paddle-like flipper wings", "Anterior tapered orange bill"], "birds")))

    # 12. Eagle
    r12 = ModelNode3D("root")
    m12_body = make_ellipsoid(rx=0.25, ry=0.25, rz=0.5, color=(35, 55, 80), name="eagle_body")
    m12_head = make_sphere(radius=0.18, color=(240, 245, 250), name="eagle_white_head")  # Bald eagle white head
    m12_beak = make_cone(radius=0.06, height=0.18, color=(20, 180, 235), name="hooked_beak")
    m12_wing_l = make_box(dx=1.1, dy=0.04, dz=0.35, color=(35, 55, 80), name="wing_l")
    m12_wing_r = make_box(dx=1.1, dy=0.04, dz=0.35, color=(35, 55, 80), name="wing_r")
    r12.add_child(ModelNode3D("body", mesh=m12_body))
    r12.add_child(ModelNode3D("head", mesh=m12_head, position=np.array([0, 0.15, 0.45], dtype=np.float32)))
    r12.add_child(ModelNode3D("beak", mesh=m12_beak, position=np.array([0, 0.12, 0.62], dtype=np.float32), rotation=np.array([math.pi/2, 0, 0], dtype=np.float32)))
    r12.add_child(ModelNode3D("wing_l", mesh=m12_wing_l, position=np.array([-0.9, 0.1, 0], dtype=np.float32)))
    r12.add_child(ModelNode3D("wing_r", mesh=m12_wing_r, position=np.array([0.9, 0.1, 0], dtype=np.float32)))
    models.append((Model3D(concept="Eagle", root=r12),
                   _meta("Eagle", ["bald eagle", "raptor", "bird of prey", "aquila"],
                         "Large bird of prey with wide outstretched soaring wings, pure white head plumage, and hooked raptorial beak.",
                         ["Wide outstretched aerofoil wingspan", "Contrasting white cranial plumage", "Curved hooked predatory beak"], "birds")))

    # 13. Owl
    r13 = ModelNode3D("root")
    m13_body = make_cylinder(radius_top=0.32, radius_bottom=0.35, height=0.75, color=(70, 95, 125), name="owl_body")
    m13_head = make_sphere(radius=0.32, color=(80, 110, 140), name="owl_head")
    m13_eye_l = make_sphere(radius=0.1, color=(30, 200, 240), name="eye_l")  # Yellow eyes
    m13_eye_r = make_sphere(radius=0.1, color=(30, 200, 240), name="eye_r")
    r13.add_child(ModelNode3D("body", mesh=m13_body, position=np.array([0, -0.2, 0], dtype=np.float32)))
    r13.add_child(ModelNode3D("head", mesh=m13_head, position=np.array([0, 0.35, 0], dtype=np.float32)))
    r13.add_child(ModelNode3D("eye_l", mesh=m13_eye_l, position=np.array([-0.14, 0.4, 0.28], dtype=np.float32)))
    r13.add_child(ModelNode3D("eye_r", mesh=m13_eye_r, position=np.array([0.14, 0.4, 0.28], dtype=np.float32)))
    models.append((Model3D(concept="Owl", root=r13),
                   _meta("Owl", ["strigiformes", "barn owl", "nocturnal bird", "horned owl"],
                         "Nocturnal bird of prey with upright stance, large forward-facing circular facial discs, and large yellow eyes.",
                         ["Upright compact cylindrical posture", "Prominent forward-facing binocular eyes", "Facial disc plumage architecture"], "birds")))

    # 14. Dolphin
    r14 = ModelNode3D("root")
    m14_body = make_ellipsoid(rx=0.3, ry=0.32, rz=1.0, color=(160, 130, 90), roughness=0.3, name="dolphin_body")  # Blue-grey
    m14_dorsal = make_cone(radius=0.12, height=0.35, color=(140, 110, 80), name="dorsal_fin")
    m14_fluke = make_box(dx=0.6, dy=0.04, dz=0.22, color=(140, 110, 80), name="tail_fluke")
    r14.add_child(ModelNode3D("body", mesh=m14_body))
    r14.add_child(ModelNode3D("dorsal", mesh=m14_dorsal, position=np.array([0, 0.38, 0], dtype=np.float32), rotation=np.array([-0.4, 0, 0], dtype=np.float32)))
    r14.add_child(ModelNode3D("fluke", mesh=m14_fluke, position=np.array([0, 0, -1.0], dtype=np.float32)))
    models.append((Model3D(concept="Dolphin", root=r14),
                   _meta("Dolphin", ["delphinus", "bottlenose dolphin", "marine cetacean", "porpoise"],
                         "Streamlined marine mammal with sleek hydrodynamic body, falcate dorsal fin, and horizontal caudal tail flukes.",
                         ["Hydrodynamic fusiform marine body", "Falcate curved dorsal fin", "Horizontal propulsion tail flukes"], "marine_life")))

    # 15. Shark
    r15 = ModelNode3D("root")
    m15_body = make_ellipsoid(rx=0.32, ry=0.35, rz=1.1, color=(100, 105, 115), roughness=0.4, name="shark_body")
    m15_fin = make_cone(radius=0.15, height=0.5, color=(90, 95, 105), name="dorsal_fin")
    m15_tail = make_box(dx=0.04, dy=0.65, dz=0.3, color=(90, 95, 105), name="caudal_tail")
    r15.add_child(ModelNode3D("body", mesh=m15_body))
    r15.add_child(ModelNode3D("fin", mesh=m15_fin, position=np.array([0, 0.45, 0.1], dtype=np.float32), rotation=np.array([-0.3, 0, 0], dtype=np.float32)))
    r15.add_child(ModelNode3D("tail", mesh=m15_tail, position=np.array([0, 0.1, -1.15], dtype=np.float32), rotation=np.array([0.3, 0, 0], dtype=np.float32)))
    models.append((Model3D(concept="Shark", root=r15),
                   _meta("Shark", ["selachimorpha", "great white shark", "predatory fish"],
                         "Apex marine cartilaginous predator with torpedo-shaped fuselage, prominent sharp dorsal fin, and vertical tail fin.",
                         ["Fusiform torpedo hydrodynamic body", "Prominent sharp triangular dorsal fin", "Heterocercal vertical caudal tail fin"], "marine_life")))

    # 16. Whale
    r16 = ModelNode3D("root")
    m16_body = make_ellipsoid(rx=0.55, ry=0.5, rz=1.3, color=(80, 70, 60), roughness=0.4, name="whale_body")  # Deep blue/grey
    m16_fluke = make_box(dx=0.9, dy=0.06, dz=0.35, color=(70, 60, 50), name="tail_flukes")
    r16.add_child(ModelNode3D("body", mesh=m16_body))
    r16.add_child(ModelNode3D("fluke", mesh=m16_fluke, position=np.array([0, 0, -1.35], dtype=np.float32)))
    models.append((Model3D(concept="Whale", root=r16),
                   _meta("Whale", ["blue whale", "cetacean", "humpback whale", "baleen whale"],
                         "Massive aquatic cetacean mammal with colossal streamlined body and broad horizontal fluke propulsion blades.",
                         ["Massive colossal cetacean fuselage", "Wide horizontal caudal fluke blades", "Ventral throat pleats contour"], "marine_life")))

    # 17. Octopus
    r17 = ModelNode3D("root")
    m17_head = make_sphere(radius=0.45, color=(40, 80, 180), name="mantle_head")  # Coral red BGR
    r17.add_child(ModelNode3D("mantle", mesh=m17_head, position=np.array([0, 0.35, 0], dtype=np.float32)))
    for t_idx in range(8):
        ang = 2.0 * math.pi * t_idx / 8
        arm = make_cylinder(radius_top=0.07, radius_bottom=0.02, height=1.0, color=(35, 75, 170), name=f"tent_{t_idx}")
        tx, tz = 0.35 * math.cos(ang), 0.35 * math.sin(ang)
        r17.add_child(ModelNode3D(f"tent_{t_idx}", mesh=arm, position=np.array([tx, -0.3, tz], dtype=np.float32), rotation=np.array([0.5 * math.sin(ang), 0, 0.5 * math.cos(ang)], dtype=np.float32)))
    models.append((Model3D(concept="Octopus", root=r17),
                   _meta("Octopus", ["octopod", "cephalopod", "eight-armed mollusc"],
                         "Soft-bodied eight-limbed cephalopod mollusc with large bulbous mantle head and radial prehensile tentacles.",
                         ["Bulbous cranial mantle sac", "Eight radial flexible grasping tentacles", "Bilaterally symmetric cephalopod"], "marine_life")))

    # 18. Butterfly
    r18 = ModelNode3D("root")
    m18_body = make_cylinder(radius_top=0.04, radius_bottom=0.04, height=0.7, color=(30, 30, 35), name="butterfly_body")
    m18_wing_l = make_ellipsoid(rx=0.7, ry=0.015, rz=0.45, color=(210, 100, 30), name="wing_left")  # Cyan/blue wings BGR
    m18_wing_r = make_ellipsoid(rx=0.7, ry=0.015, rz=0.45, color=(210, 100, 30), name="wing_right")
    r18.add_child(ModelNode3D("body", mesh=m18_body, rotation=np.array([math.pi/2, 0, 0], dtype=np.float32)))
    r18.add_child(ModelNode3D("wing_l", mesh=m18_wing_l, position=np.array([-0.65, 0.1, 0], dtype=np.float32), rotation=np.array([0, 0, 0.25], dtype=np.float32)))
    r18.add_child(ModelNode3D("wing_r", mesh=m18_wing_r, position=np.array([0.65, 0.1, 0], dtype=np.float32), rotation=np.array([0, 0, -0.25], dtype=np.float32)))
    models.append((Model3D(concept="Butterfly", root=r18),
                   _meta("Butterfly", ["lepidoptera", "monarch butterfly", "winged insect"],
                         "Four-winged insect of the order Lepidoptera with segmented body and broad, brightly patterned membrane wings.",
                         ["Slender segmented insect body", "Broad paired membranous aerofoil wings", "Bilateral scale pigmentation display"], "insects")))

    # 19. Bee
    r19 = ModelNode3D("root")
    m19_abdomen = make_ellipsoid(rx=0.3, ry=0.3, rz=0.5, color=(30, 180, 230), name="bee_abdomen")  # Yellow/black stripes
    m19_stripe = make_box(dx=0.62, dy=0.62, dz=0.15, color=(20, 20, 20), name="bee_stripe")
    m19_wings = make_box(dx=0.8, dy=0.01, dz=0.25, color=(230, 240, 250), opacity=0.7, name="bee_wings")
    r19.add_child(ModelNode3D("abdomen", mesh=m19_abdomen))
    r19.add_child(ModelNode3D("stripe", mesh=m19_stripe, position=np.array([0, 0, 0.05], dtype=np.float32)))
    r19.add_child(ModelNode3D("wings", mesh=m19_wings, position=np.array([0, 0.32, 0.1], dtype=np.float32)))
    models.append((Model3D(concept="Bee", root=r19),
                   _meta("Bee", ["honeybee", "bumblebee", "apis mellifera", "pollinator"],
                         "Social winged flying insect known for honey production, pollination, and vivid yellow-and-black abdominal bands.",
                         ["Yellow and black striped abdomen", "Translucent vibrating flight wings", "Hymenopteran waist constriction"], "insects")))

    # 20. Turtle
    r20 = ModelNode3D("root")
    m20_shell = make_ellipsoid(rx=0.6, ry=0.32, rz=0.75, color=(40, 95, 60), roughness=0.6, name="carapace_shell")  # Olive green BGR
    m20_head = make_sphere(radius=0.16, color=(50, 120, 70), name="turtle_head")
    r20.add_child(ModelNode3D("shell", mesh=m20_shell))
    r20.add_child(ModelNode3D("head", mesh=m20_head, position=np.array([0, 0.05, 0.85], dtype=np.float32)))
    for dx, dz in [(-0.55, 0.4), (0.55, 0.4), (-0.5, -0.4), (0.5, -0.4)]:
        flipper = make_ellipsoid(rx=0.22, ry=0.06, rz=0.14, color=(50, 120, 70), name=f"flip_{dx}_{dz}")
        r20.add_child(ModelNode3D(f"flip_{dx}_{dz}", mesh=flipper, position=np.array([dx, -0.1, dz], dtype=np.float32)))
    models.append((Model3D(concept="Turtle", root=r20),
                   _meta("Turtle", ["testudines", "sea turtle", "tortoise", "chelonian"],
                         "Reptile with domed bony carapace shell protecting body, outstretched flippers, and projecting head.",
                         ["Domed protective bony carapace", "Projecting cranial neck and head", "Paddle-like swimming flippers"], "reptiles")))

    return models
