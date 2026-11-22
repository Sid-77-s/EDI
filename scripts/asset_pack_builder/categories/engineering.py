"""
scripts/asset_pack_builder/categories/engineering.py — 15 Engineering, Machines, and Technology Models.
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


def _meta(name: str, aliases: List[str], desc: str, features: List[str], subcat: str = "mechanical") -> Dict:
    return {
        "name": name,
        "aliases": aliases,
        "category": "Engineering, Machines and Technology",
        "subcategory": subcat,
        "description": desc,
        "distinguishing_features": features,
        "educational_purpose": f"Visualizes the 3D mechanical and electronic engineering architecture of {name}.",
        "license": "CC0-1.0",
        "author": "Magic Mirror Spatial AI Team",
        "source": "Magic Mirror 3D Asset Library",
        "attribution": "Curated for Magic Mirror under CC0 1.0 Universal",
        "download_date": "2026-11-22",
    }


def get_models() -> List[Tuple[Model3D, Dict]]:
    models = []

    # 1. Industrial Robotic Arm
    r1 = ModelNode3D("root")
    m1_base = make_cylinder(radius_top=0.4, radius_bottom=0.45, height=0.25, color=(210, 140, 30), name="arm_base")  # Industrial yellow/orange BGR
    m1_link1 = make_cylinder(radius_top=0.1, radius_bottom=0.1, height=0.8, color=(210, 140, 30), name="lower_arm")
    m1_link2 = make_cylinder(radius_top=0.08, radius_bottom=0.08, height=0.7, color=(50, 50, 55), name="upper_arm")
    m1_gripper_l = make_box(dx=0.04, dy=0.2, dz=0.04, color=(180, 180, 190), name="jaw_l")
    m1_gripper_r = make_box(dx=0.04, dy=0.2, dz=0.04, color=(180, 180, 190), name="jaw_r")
    r1.add_child(ModelNode3D("base", mesh=m1_base))
    r1.add_child(ModelNode3D("link1", mesh=m1_link1, position=np.array([0, 0.45, 0], dtype=np.float32), rotation=np.array([0, 0, 0.25], dtype=np.float32)))
    r1.add_child(ModelNode3D("link2", mesh=m1_link2, position=np.array([0.2, 1.05, 0], dtype=np.float32), rotation=np.array([0, 0, -0.5], dtype=np.float32)))
    r1.add_child(ModelNode3D("jaw_l", mesh=m1_gripper_l, position=np.array([0.5, 1.35, -0.06], dtype=np.float32)))
    r1.add_child(ModelNode3D("jaw_r", mesh=m1_gripper_r, position=np.array([0.5, 1.35, 0.06], dtype=np.float32)))
    models.append((Model3D(concept="Robotic Arm", root=r1),
                   _meta("Robotic Arm", ["industrial robot", "articulated arm", "6-axis robot", "manipulator"],
                         "Articulated multi-axis industrial manipulator with revolving pedestal base, shoulder, forearm, and parallel-jaw end effector gripper.",
                         ["Articulated multi-axis robotic joints", "Vivid safety industrial orange finish", "Precision parallel mechanical gripper end effector"], "mechanical")))

    # 2. Turbofan Jet Engine
    r2 = ModelNode3D("root")
    m2_nacelle = make_cylinder(radius_top=0.75, radius_bottom=0.65, height=1.6, color=(170, 175, 185), roughness=0.3, metallic=0.6, name="fan_cowl")
    m2_spinner = make_cone(radius=0.22, height=0.45, color=(30, 30, 35), name="spinner_cone")
    m2_exhaust = make_cone(radius=0.45, height=0.5, color=(70, 70, 75), name="exhaust_plug")
    r2.add_child(ModelNode3D("nacelle", mesh=m2_nacelle, rotation=np.array([math.pi/2, 0, 0], dtype=np.float32)))
    r2.add_child(ModelNode3D("spinner", mesh=m2_spinner, position=np.array([0, 0, 0.75], dtype=np.float32), rotation=np.array([math.pi/2, 0, 0], dtype=np.float32)))
    r2.add_child(ModelNode3D("exhaust", mesh=m2_exhaust, position=np.array([0, 0, -0.9], dtype=np.float32), rotation=np.array([-math.pi/2, 0, 0], dtype=np.float32)))
    models.append((Model3D(concept="Jet Engine", root=r2),
                   _meta("Jet Engine", ["turbofan", "aircraft engine", "gas turbine", "jet turbine"],
                         "Aviation high-bypass turbofan engine with aerodynamic nacelle cowl, forward fan spinner cone, and exhaust plug nozzle.",
                         ["Aerodynamic cylindrical bypass nacelle casing", "Forward rotating aerodynamic spinner cone", "High-bypass propulsion nozzle geometry"], "aviation_engineering")))

    # 3. Planetary Gear Assembly
    r3 = ModelNode3D("root")
    m3_sun = make_cylinder(radius_top=0.25, radius_bottom=0.25, height=0.15, segs=12, color=(30, 180, 240), name="sun_gear")  # Gold
    m3_ring = make_ring_disc(r_inner=0.8, r_outer=0.98, segs=28, color=(140, 145, 155), name="ring_gear")
    r3.add_child(ModelNode3D("sun", mesh=m3_sun))
    r3.add_child(ModelNode3D("ring", mesh=m3_ring))
    for p_idx in range(3):
        ang = 2.0 * math.pi * p_idx / 3
        planet = make_cylinder(radius_top=0.22, radius_bottom=0.22, height=0.15, segs=10, color=(180, 100, 50), name=f"planet_{p_idx}")
        px, pz = 0.52 * math.cos(ang), 0.52 * math.sin(ang)
        r3.add_child(ModelNode3D(f"planet_{p_idx}", mesh=planet, position=np.array([px, 0, pz], dtype=np.float32)))
    models.append((Model3D(concept="Planetary Gear", root=r3),
                   _meta("Planetary Gear", ["epicyclic gear", "gear train", "sun and planet gear", "planetary transmission"],
                         "Epicyclic gear system consisting of central sun gear, three revolving planet gears, and outer annular ring gear.",
                         ["Central drive sun gear cylinder", "Three symmetrically meshed planet gears", "Outer concentric annular internal ring gear"], "mechanical")))

    # 4. CPU Chip
    r4 = ModelNode3D("root")
    m4_sub = make_box(dx=1.2, dy=0.06, dz=1.2, color=(40, 140, 60), name="pcb_substrate")  # Green PCB substrate
    m4_ihs = make_box(dx=0.85, dy=0.08, dz=0.85, color=(200, 205, 215), roughness=0.2, metallic=0.9, name="ihs_spreader")  # Nickel IHS
    m4_mark = make_cone(radius=0.06, height=0.02, color=(40, 210, 240), name="pin_1_notch")  # Gold triangle
    r4.add_child(ModelNode3D("sub", mesh=m4_sub))
    r4.add_child(ModelNode3D("ihs", mesh=m4_ihs, position=np.array([0, 0.06, 0], dtype=np.float32)))
    r4.add_child(ModelNode3D("mark", mesh=m4_mark, position=np.array([-0.5, 0.04, -0.5], dtype=np.float32)))
    models.append((Model3D(concept="CPU", root=r4),
                   _meta("CPU", ["microprocessor", "central processing unit", "processor chip", "computer chip"],
                         "Semiconductor microprocessor package with green FR4 substrate, metallic nickel-plated integrated heat spreader (IHS), and pin-1 marker.",
                         ["Square multi-layer PCB carrier substrate", "Metallic nickel-plated integrated heat spreader (IHS)", "Gold corner index alignment mark"], "electronics")))

    # 5. Circuit Board PCB
    r5 = ModelNode3D("root")
    m5_board = make_box(dx=1.4, dy=0.04, dz=1.0, color=(30, 120, 50), roughness=0.6, name="fr4_board")
    m5_chip = make_box(dx=0.35, dy=0.06, dz=0.35, color=(20, 20, 25), name="qfp_chip")
    m5_cap1 = make_cylinder(radius_top=0.08, radius_bottom=0.08, height=0.18, color=(40, 45, 180), name="cap_1")
    m5_cap2 = make_cylinder(radius_top=0.08, radius_bottom=0.08, height=0.18, color=(40, 45, 180), name="cap_2")
    r5.add_child(ModelNode3D("board", mesh=m5_board))
    r5.add_child(ModelNode3D("chip", mesh=m5_chip, position=np.array([-0.2, 0.04, 0], dtype=np.float32)))
    r5.add_child(ModelNode3D("c1", mesh=m5_cap1, position=np.array([0.35, 0.1, 0.2], dtype=np.float32)))
    r5.add_child(ModelNode3D("c2", mesh=m5_cap2, position=np.array([0.35, 0.1, -0.2], dtype=np.float32)))
    models.append((Model3D(concept="Circuit Board", root=r5),
                   _meta("Circuit Board", ["PCB", "printed circuit board", "motherboard", "electronics board"],
                         "Printed circuit board (PCB) featuring characteristic green solder mask, soldered integrated circuit IC chips, and cylindrical electrolytic capacitors.",
                         ["Green solder mask laminate substrate plate", "Surface mount black IC chip packages", "Vertical cylindrical electrolytic capacitors"], "electronics")))

    # 6. 3D Printer
    r6 = ModelNode3D("root")
    m6_frame = make_box(dx=1.1, dy=1.2, dz=0.1, color=(35, 35, 40), name="gantry_arch")
    m6_bed = make_box(dx=0.8, dy=0.04, dz=0.8, color=(180, 180, 185), name="heated_bed")
    m6_head = make_box(dx=0.2, dy=0.18, dz=0.18, color=(210, 40, 40), name="printhead_extruder")
    r6.add_child(ModelNode3D("frame", mesh=m6_frame, position=np.array([0, 0.6, -0.3], dtype=np.float32)))
    r6.add_child(ModelNode3D("bed", mesh=m6_bed, position=np.array([0, 0.1, 0], dtype=np.float32)))
    r6.add_child(ModelNode3D("head", mesh=m6_head, position=np.array([0, 0.55, 0], dtype=np.float32)))
    models.append((Model3D(concept="3D Printer", root=r6),
                   _meta("3D Printer", ["additive manufacturing", "fdm printer", "rapid prototyper"],
                         "Fused deposition modeling (FDM) 3D printer with vertical Cartesian gantry frame, square heated build plate bed, and toolhead extruder.",
                         ["Cartesian axis aluminum gantry frame", "Square horizontal heated build plate", "Precision hotend extruder carriage"], "mechanical")))

    # 7. Wind Turbine
    r7 = ModelNode3D("root")
    m7_tower = make_cylinder(radius_top=0.06, radius_bottom=0.14, height=2.4, color=(240, 245, 250), name="tower_mast")
    m7_nacelle = make_box(dx=0.22, dy=0.22, dz=0.55, color=(240, 245, 250), name="generator_nacelle")
    m7_hub = make_cone(radius=0.12, height=0.25, color=(240, 245, 250), name="rotor_hub")
    r7.add_child(ModelNode3D("tower", mesh=m7_tower))
    r7.add_child(ModelNode3D("nacelle", mesh=m7_nacelle, position=np.array([0, 1.25, 0], dtype=np.float32)))
    r7.add_child(ModelNode3D("hub", mesh=m7_hub, position=np.array([0, 1.25, 0.35], dtype=np.float32), rotation=np.array([math.pi/2, 0, 0], dtype=np.float32)))
    for b_idx in range(3):
        ang = 2.0 * math.pi * b_idx / 3
        blade = make_box(dx=0.08, dy=1.2, dz=0.02, color=(240, 245, 250), name=f"blade_{b_idx}")
        r7.add_child(ModelNode3D(f"blade_{b_idx}", mesh=blade, position=np.array([0.55 * math.sin(ang), 1.25 + 0.55 * math.cos(ang), 0.35], dtype=np.float32), rotation=np.array([0, 0, -ang], dtype=np.float32)))
    models.append((Model3D(concept="Wind Turbine", root=r7),
                   _meta("Wind Turbine", ["wind generator", "wind mill", "renewable energy turbine"],
                         "Renewable wind power electricity generator featuring tall conical tubular steel tower, nacelle housing, and 3 aerodynamic rotor blades.",
                         ["Tall slender tapered tubular steel tower", "Overhead generator nacelle pod", "Three-bladed aerodynamic rotor assembly"], "renewable_energy")))

    # 8. Solar Panel
    r8 = ModelNode3D("root")
    m8_frame = make_box(dx=1.4, dy=0.06, dz=1.8, color=(180, 185, 190), roughness=0.3, metallic=0.7, name="aluminum_frame")
    m8_cells = make_box(dx=1.3, dy=0.08, dz=1.7, color=(160, 50, 30), roughness=0.1, name="photovoltaic_cells")  # Deep blue/cyan PV cells BGR
    r8.add_child(ModelNode3D("frame", mesh=m8_frame, rotation=np.array([0.45, 0, 0], dtype=np.float32)))
    r8.add_child(ModelNode3D("cells", mesh=m8_cells, rotation=np.array([0.45, 0, 0], dtype=np.float32)))
    models.append((Model3D(concept="Solar Panel", root=r8),
                   _meta("Solar Panel", ["photovoltaic module", "pv panel", "solar module", "solar array"],
                         "Photovoltaic module designed to absorb sun rays as source of energy, with protective extruded aluminum frame and blue crystalline cells.",
                         ["Extruded anodized aluminum perimeter frame", "Deep blue anti-reflective crystalline silicon cell matrix", "Tilted incident angle mounting orientation"], "renewable_energy")))

    # 9. Electric Motor
    r9 = ModelNode3D("root")
    m9_stator = make_cylinder(radius_top=0.45, radius_bottom=0.45, height=0.9, color=(50, 110, 160), name="stator_frame")  # Blue motor housing
    m9_shaft = make_cylinder(radius_top=0.08, radius_bottom=0.08, height=1.4, color=(200, 205, 215), metallic=0.9, name="rotor_shaft")
    m9_jbox = make_box(dx=0.3, dy=0.2, dz=0.3, color=(40, 40, 45), name="junction_box")
    r9.add_child(ModelNode3D("stator", mesh=m9_stator, rotation=np.array([math.pi/2, 0, 0], dtype=np.float32)))
    r9.add_child(ModelNode3D("shaft", mesh=m9_shaft, rotation=np.array([math.pi/2, 0, 0], dtype=np.float32)))
    r9.add_child(ModelNode3D("jbox", mesh=m9_jbox, position=np.array([0, 0.45, 0], dtype=np.float32)))
    models.append((Model3D(concept="Electric Motor", root=r9),
                   _meta("Electric Motor", ["induction motor", "AC motor", "electromotor", "drive motor"],
                         "Electromechanical machine converting electrical energy into mechanical energy, with cast iron stator body, drive shaft, and conduit terminal box.",
                         ["Cylindrical ribbed stator housing", "Protruding hardened steel drive rotor shaft", "Top mounted electrical terminal junction box"], "electrical_engineering")))

    # 10. Hydraulic Cylinder
    r10 = ModelNode3D("root")
    m10_barrel = make_cylinder(radius_top=0.25, radius_bottom=0.25, height=1.2, color=(35, 35, 40), name="cylinder_barrel")
    m10_rod = make_cylinder(radius_top=0.12, radius_bottom=0.12, height=1.0, color=(230, 235, 245), roughness=0.1, metallic=0.9, name="chrome_piston_rod")
    r10.add_child(ModelNode3D("barrel", mesh=m10_barrel))
    r10.add_child(ModelNode3D("rod", mesh=m10_rod, position=np.array([0, 0.9, 0], dtype=np.float32)))
    models.append((Model3D(concept="Hydraulic Cylinder", root=r10),
                   _meta("Hydraulic Cylinder", ["hydraulic actuator", "ram", "pneumatic piston", "piston actuator"],
                         "Mechanical actuator used to give unidirectional force through unidirectional stroke, with heavy barrel and extending hard-chromed rod.",
                         ["Heavy seamless steel cylinder barrel tube", "Extending polished hard-chrome plated piston rod", "High-pressure fluid actuation geometry"], "mechanical")))

    # 11. Water Pump
    r11 = ModelNode3D("root")
    m11_volute = make_torus(r_major=0.45, r_minor=0.2, color=(40, 120, 180), name="volute_casing")
    m11_inlet = make_cylinder(radius_top=0.18, radius_bottom=0.18, height=0.5, color=(40, 120, 180), name="suction_inlet")
    m11_outlet = make_cylinder(radius_top=0.15, radius_bottom=0.15, height=0.5, color=(40, 120, 180), name="discharge_outlet")
    r11.add_child(ModelNode3D("volute", mesh=m11_volute))
    r11.add_child(ModelNode3D("inlet", mesh=m11_inlet, position=np.array([0, 0, 0.4], dtype=np.float32), rotation=np.array([math.pi/2, 0, 0], dtype=np.float32)))
    r11.add_child(ModelNode3D("outlet", mesh=m11_outlet, position=np.array([0.5, 0.4, 0], dtype=np.float32)))
    models.append((Model3D(concept="Water Pump", root=r11),
                   _meta("Water Pump", ["centrifugal pump", "fluid pump", "impeller pump"],
                         "Centrifugal water pump with spiral volute casing, axial suction inlet flange, and tangential vertical discharge outlet.",
                         ["Spiral hydrodynamic volute casing", "Central axial suction eye inlet", "Tangential discharge pressure flange"], "fluid_mechanics")))

    # 12. Steam Turbine
    r12 = ModelNode3D("root")
    m12_casing = make_cylinder(radius_top=0.6, radius_bottom=0.75, height=1.4, color=(130, 135, 140), name="split_casing")
    m12_shaft = make_cylinder(radius_top=0.1, radius_bottom=0.1, height=1.8, color=(200, 205, 215), metallic=0.9, name="turbine_rotor")
    r12.add_child(ModelNode3D("casing", mesh=m12_casing, rotation=np.array([math.pi/2, 0, 0], dtype=np.float32)))
    r12.add_child(ModelNode3D("shaft", mesh=m12_shaft, rotation=np.array([math.pi/2, 0, 0], dtype=np.float32)))
    models.append((Model3D(concept="Steam Turbine", root=r12),
                   _meta("Steam Turbine", ["power turbine", "rotary steam engine", "turbogenerator"],
                         "Turbomachine extracting thermal energy from pressurized steam to perform mechanical work on rotating bladed shaft stages.",
                         ["Heavy cast steel split casing shell", "Concentric high-speed rotor shaft axis", "Multi-stage thermodynamic expansion architecture"], "mechanical")))

    # 13. DSLR Camera
    r13 = ModelNode3D("root")
    m13_body = make_box(dx=1.0, dy=0.7, dz=0.5, color=(30, 30, 35), roughness=0.7, name="camera_body")
    m13_lens = make_cylinder(radius_top=0.3, radius_bottom=0.3, height=0.6, color=(25, 25, 30), name="zoom_lens")
    m13_prism = make_box(dx=0.35, dy=0.2, dz=0.35, color=(30, 30, 35), name="pentaprism_hump")
    r13.add_child(ModelNode3D("body", mesh=m13_body))
    r13.add_child(ModelNode3D("lens", mesh=m13_lens, position=np.array([0, 0, 0.45], dtype=np.float32), rotation=np.array([math.pi/2, 0, 0], dtype=np.float32)))
    r13.add_child(ModelNode3D("prism", mesh=m13_prism, position=np.array([0, 0.4, 0], dtype=np.float32)))
    models.append((Model3D(concept="Camera", root=r13),
                   _meta("Camera", ["dslr", "digital camera", "slr camera", "photo camera"],
                         "Digital single-lens reflex (DSLR) camera with ergonomic grip body, prominent top pentaprism optical viewfinder, and barrel lens.",
                         ["Ergonomic matte black magnesium chassis body", "Top pentaprism optical viewfinder hump", "Cylindrical optical glass zoom lens barrel"], "optics_electronics")))

    # 14. Studio Microphone
    r14 = ModelNode3D("root")
    m14_handle = make_cylinder(radius_top=0.08, radius_bottom=0.08, height=0.9, color=(40, 40, 45), name="mic_handle")
    m14_grille = make_sphere(radius=0.22, color=(180, 185, 195), roughness=0.4, metallic=0.8, name="capsule_grille")  # Silver wire mesh
    r14.add_child(ModelNode3D("handle", mesh=m14_handle, position=np.array([0, -0.3, 0], dtype=np.float32)))
    r14.add_child(ModelNode3D("grille", mesh=m14_grille, position=np.array([0, 0.35, 0], dtype=np.float32)))
    models.append((Model3D(concept="Microphone", root=r14),
                   _meta("Microphone", ["mic", "dynamic mic", "studio condenser", "vocal mic"],
                         "Acoustic-to-electric transducer with cylindrical handheld body and spherical steel mesh pop filter capsule grille.",
                         ["Cylindrical die-cast zinc handheld handle", "Spherical woven steel mesh acoustic capsule grille", "Acoustic transducer diaphragm architecture"], "audio_engineering")))

    # 15. Audio Speaker
    r15 = ModelNode3D("root")
    m15_cab = make_box(dx=0.8, dy=1.2, dz=0.7, color=(25, 25, 30), roughness=0.6, name="mdf_enclosure")
    m15_woofer = make_cylinder(radius_top=0.28, radius_bottom=0.28, height=0.04, color=(15, 15, 15), name="woofer_cone")
    m15_tweeter = make_sphere(radius=0.1, color=(190, 190, 195), metallic=0.7, name="dome_tweeter")
    r15.add_child(ModelNode3D("cab", mesh=m15_cab))
    r15.add_child(ModelNode3D("woofer", mesh=m15_woofer, position=np.array([0, -0.2, 0.36], dtype=np.float32), rotation=np.array([math.pi/2, 0, 0], dtype=np.float32)))
    r15.add_child(ModelNode3D("tweeter", mesh=m15_tweeter, position=np.array([0, 0.35, 0.36], dtype=np.float32)))
    models.append((Model3D(concept="Speaker", root=r15),
                   _meta("Speaker", ["loudspeaker", "studio monitor", "audio speaker", "woofer"],
                         "Two-way acoustic loudspeaker monitor featuring cuboid resonance enclosure, large low-frequency woofer cone, and dome tweeter.",
                         ["Solid acoustic cabinet enclosure", "Large cone-shaped low frequency woofer driver", "High frequency silk dome tweeter aperture"], "audio_engineering")))

    return models
