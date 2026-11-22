"""
scripts/asset_pack_builder/categories/vehicles.py — 20 Vehicles and Transportation Models.
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


def _meta(name: str, aliases: List[str], desc: str, features: List[str], subcat: str = "road") -> Dict:
    return {
        "name": name,
        "aliases": aliases,
        "category": "Vehicles and Transportation",
        "subcategory": subcat,
        "description": desc,
        "distinguishing_features": features,
        "educational_purpose": f"Visualizes the 3D engineering design and vehicular architecture of {name}.",
        "license": "CC0-1.0",
        "author": "Magic Mirror Spatial AI Team",
        "source": "Magic Mirror 3D Asset Library",
        "attribution": "Curated for Magic Mirror under CC0 1.0 Universal",
        "download_date": "2026-11-22",
    }


def get_models() -> List[Tuple[Model3D, Dict]]:
    models = []

    # 1. Formula 1 Car
    r1 = ModelNode3D("root")
    m1_chassis = make_box(dx=0.5, dy=0.2, dz=2.0, color=(30, 30, 220), name="f1_monocoque")  # Racing red BGR
    m1_nose = make_cone(radius=0.25, height=0.8, segs=16, color=(30, 30, 220), name="f1_nose")
    m1_fw = make_box(dx=1.4, dy=0.06, dz=0.3, color=(30, 30, 35), name="front_wing")
    m1_rw = make_box(dx=1.1, dy=0.25, dz=0.1, color=(30, 30, 35), name="rear_wing")
    m1_halo = make_torus(r_major=0.25, r_minor=0.03, color=(30, 30, 35), name="cockpit_halo")
    r1.add_child(ModelNode3D("chassis", mesh=m1_chassis))
    r1.add_child(ModelNode3D("nose", mesh=m1_nose, position=np.array([0, 0, 1.2], dtype=np.float32), rotation=np.array([math.pi/2, 0, 0], dtype=np.float32)))
    r1.add_child(ModelNode3D("fw", mesh=m1_fw, position=np.array([0, -0.05, 1.45], dtype=np.float32)))
    r1.add_child(ModelNode3D("rw", mesh=m1_rw, position=np.array([0, 0.35, -0.95], dtype=np.float32)))
    r1.add_child(ModelNode3D("halo", mesh=m1_halo, position=np.array([0, 0.18, 0.1], dtype=np.float32), rotation=np.array([0.4, 0, 0], dtype=np.float32)))
    for dx, dz in [(-0.65, 0.8), (0.65, 0.8), (-0.68, -0.7), (0.68, -0.7)]:
        w = make_cylinder(radius_top=0.32, radius_bottom=0.32, height=0.25, color=(20, 20, 20), name=f"f1_tire_{dx}_{dz}")
        r1.add_child(ModelNode3D(f"tire_{dx}_{dz}", mesh=w, position=np.array([dx, -0.05, dz], dtype=np.float32), rotation=np.array([0, 0, math.pi/2], dtype=np.float32)))
    models.append((Model3D(concept="Formula 1 Car", root=r1),
                   _meta("Formula 1 Car", ["f1 car", "grand prix racer", "formula one", "racing single seater"],
                         "Single-seat, open-cockpit, open-wheel formula racing car with prominent front/rear aerofoil downforce wings and halo safety frame.",
                         ["Open-wheel racing configuration with 4 exposed slick tires", "Wide front and high rear downforce aerofoil wings", "Distinctive protective cockpit halo arc"], "road")))

    # 2. Sports Car
    r2 = ModelNode3D("root")
    m2_body = make_box(dx=0.9, dy=0.25, dz=1.8, color=(25, 25, 215), roughness=0.3, name="car_lower_body")  # Glossy red BGR
    m2_cabin = make_box(dx=0.75, dy=0.24, dz=0.8, color=(30, 35, 40), roughness=0.1, name="cockpit_cabin")
    m2_spoiler = make_box(dx=0.8, dy=0.04, dz=0.18, color=(20, 20, 20), name="rear_spoiler")
    r2.add_child(ModelNode3D("body", mesh=m2_body))
    r2.add_child(ModelNode3D("cabin", mesh=m2_cabin, position=np.array([0, 0.22, -0.1], dtype=np.float32)))
    r2.add_child(ModelNode3D("spoiler", mesh=m2_spoiler, position=np.array([0, 0.25, -0.85], dtype=np.float32)))
    for dx, dz in [(-0.48, 0.55), (0.48, 0.55), (-0.48, -0.55), (0.48, -0.55)]:
        w = make_cylinder(radius_top=0.25, radius_bottom=0.25, height=0.16, color=(25, 25, 25), name=f"wheel_{dx}_{dz}")
        r2.add_child(ModelNode3D(f"wheel_{dx}_{dz}", mesh=w, position=np.array([dx, -0.12, dz], dtype=np.float32), rotation=np.array([0, 0, math.pi/2], dtype=np.float32)))
    models.append((Model3D(concept="Sports Car", root=r2),
                   _meta("Sports Car", ["supercar", "fast car", "sports coupe", "performance car"],
                         "Low-slung, aerodynamic high-performance passenger automobile with low ground clearance and rear aerofoil spoiler.",
                         ["Low-profile aerodynamic sports coupe silhouette", "Glossy automotive paintwork finish", "Raised rear downforce aerofoil spoiler"], "road")))

    # 3. Sedan
    r3 = ModelNode3D("root")
    m3_body = make_box(dx=0.9, dy=0.28, dz=2.0, color=(160, 160, 165), roughness=0.4, name="sedan_chassis")  # Silver metallic
    m3_cabin = make_box(dx=0.8, dy=0.3, dz=1.0, color=(35, 40, 45), name="sedan_cabin")
    r3.add_child(ModelNode3D("body", mesh=m3_body))
    r3.add_child(ModelNode3D("cabin", mesh=m3_cabin, position=np.array([0, 0.26, -0.05], dtype=np.float32)))
    for dx, dz in [(-0.48, 0.6), (0.48, 0.6), (-0.48, -0.6), (0.48, -0.6)]:
        w = make_cylinder(radius_top=0.24, radius_bottom=0.24, height=0.16, color=(25, 25, 25), name=f"wheel_{dx}_{dz}")
        r3.add_child(ModelNode3D(f"wheel_{dx}_{dz}", mesh=w, position=np.array([dx, -0.14, dz], dtype=np.float32), rotation=np.array([0, 0, math.pi/2], dtype=np.float32)))
    models.append((Model3D(concept="Sedan", root=r3),
                   _meta("Sedan", ["passenger car", "saloon car", "family car", "automobile"],
                         "Three-box passenger car configuration with separate compartments for engine hood, passenger cabin, and boot trunk.",
                         ["Three-box classic automobile profile", "Elevated four-door passenger compartment", "Four standard road wheels with silver rims"], "road")))

    # 4. Truck
    r4 = ModelNode3D("root")
    m4_cab = make_box(dx=1.0, dy=0.8, dz=0.8, color=(210, 40, 40), name="semi_cab")  # Blue cab BGR
    m4_bed = make_box(dx=1.1, dy=0.5, dz=1.6, color=(180, 180, 185), name="flatbed")
    r4.add_child(ModelNode3D("cab", mesh=m4_cab, position=np.array([0, 0.2, 0.8], dtype=np.float32)))
    r4.add_child(ModelNode3D("bed", mesh=m4_bed, position=np.array([0, 0.05, -0.4], dtype=np.float32)))
    for dx, dz in [(-0.55, 0.8), (0.55, 0.8), (-0.55, -0.3), (0.55, -0.3), (-0.55, -0.9), (0.55, -0.9)]:
        w = make_cylinder(radius_top=0.32, radius_bottom=0.32, height=0.2, color=(20, 20, 20), name=f"tw_{dx}_{dz}")
        r4.add_child(ModelNode3D(f"tw_{dx}_{dz}", mesh=w, position=np.array([dx, -0.35, dz], dtype=np.float32), rotation=np.array([0, 0, math.pi/2], dtype=np.float32)))
    models.append((Model3D(concept="Truck", root=r4),
                   _meta("Truck", ["semi truck", "lorry", "cargo truck", "heavy hauler"],
                         "Heavy commercial motor vehicle with upright forward operator cabin and elongated rear cargo load bed on 6 heavy wheels.",
                         ["Upright forward commercial operator cabin", "Long rear heavy-duty cargo haulage bed", "Six heavy-duty radial truck wheels"], "road")))

    # 5. City Bus
    r5 = ModelNode3D("root")
    m5_bus = make_box(dx=1.1, dy=0.9, dz=2.6, color=(30, 210, 240), roughness=0.4, name="bus_body")  # Yellow transit bus BGR
    m5_win = make_box(dx=1.12, dy=0.35, dz=2.3, color=(35, 40, 45), name="bus_windows")
    r5.add_child(ModelNode3D("body", mesh=m5_bus))
    r5.add_child(ModelNode3D("windows", mesh=m5_win, position=np.array([0, 0.15, 0], dtype=np.float32)))
    for dx, dz in [(-0.58, 0.85), (0.58, 0.85), (-0.58, -0.85), (0.58, -0.85)]:
        w = make_cylinder(radius_top=0.32, radius_bottom=0.32, height=0.18, color=(20, 20, 20), name=f"bw_{dx}_{dz}")
        r5.add_child(ModelNode3D(f"bw_{dx}_{dz}", mesh=w, position=np.array([dx, -0.45, dz], dtype=np.float32), rotation=np.array([0, 0, math.pi/2], dtype=np.float32)))
    models.append((Model3D(concept="Bus", root=r5),
                   _meta("Bus", ["city bus", "transit bus", "coach", "public transit"],
                         "Public transit passenger vehicle with long rectangular body, continuous side glazing window bands, and high passenger capacity.",
                         ["Long rectangular mass transit monocoque body", "Continuous horizontal panoramic window glazing", "Vivid high-visibility transit coach livery"], "road")))

    # 6. Ambulance
    r6 = ModelNode3D("root")
    m6_body = make_box(dx=1.0, dy=0.85, dz=2.2, color=(245, 245, 250), name="ambulance_body")
    m6_bar = make_box(dx=0.5, dy=0.12, dz=0.18, color=(30, 30, 220), name="lightbar")  # Red emergency light
    m6_cross_h = make_box(dx=0.35, dy=0.1, dz=0.02, color=(30, 30, 210), name="cross_h")
    m6_cross_v = make_box(dx=0.1, dy=0.35, dz=0.02, color=(30, 30, 210), name="cross_v")
    r6.add_child(ModelNode3D("body", mesh=m6_body))
    r6.add_child(ModelNode3D("bar", mesh=m6_bar, position=np.array([0, 0.5, 0.4], dtype=np.float32)))
    r6.add_child(ModelNode3D("cross_h", mesh=m6_cross_h, position=np.array([0.51, 0.05, 0], dtype=np.float32), rotation=np.array([0, math.pi/2, 0], dtype=np.float32)))
    r6.add_child(ModelNode3D("cross_v", mesh=m6_cross_v, position=np.array([0.51, 0.05, 0], dtype=np.float32), rotation=np.array([0, math.pi/2, 0], dtype=np.float32)))
    for dx, dz in [(-0.52, 0.65), (0.52, 0.65), (-0.52, -0.65), (0.52, -0.65)]:
        w = make_cylinder(radius_top=0.28, radius_bottom=0.28, height=0.16, color=(25, 25, 25), name=f"aw_{dx}_{dz}")
        r6.add_child(ModelNode3D(f"aw_{dx}_{dz}", mesh=w, position=np.array([dx, -0.42, dz], dtype=np.float32), rotation=np.array([0, 0, math.pi/2], dtype=np.float32)))
    models.append((Model3D(concept="Ambulance", root=r6),
                   _meta("Ambulance", ["emergency vehicle", "paramedic van", "medical transport"],
                         "Medically equipped emergency vehicle with white high-roof body, roof flashing emergency lightbar, and red cross insignia.",
                         ["White emergency medical van coachwork", "Roof-mounted emergency strobe lightbar", "Contrasting medical Red Cross lateral insignia"], "emergency")))

    # 7. Police Car
    r7 = ModelNode3D("root")
    m7_body = make_box(dx=0.9, dy=0.28, dz=2.0, color=(245, 245, 250), name="police_sedan")
    m7_cabin = make_box(dx=0.8, dy=0.3, dz=1.0, color=(25, 25, 30), name="police_cabin")  # Black cabin
    m7_bar_b = make_box(dx=0.25, dy=0.1, dz=0.12, color=(220, 40, 40), name="siren_blue")
    m7_bar_r = make_box(dx=0.25, dy=0.1, dz=0.12, color=(30, 30, 220), name="siren_red")
    r7.add_child(ModelNode3D("body", mesh=m7_body))
    r7.add_child(ModelNode3D("cabin", mesh=m7_cabin, position=np.array([0, 0.26, -0.05], dtype=np.float32)))
    r7.add_child(ModelNode3D("siren_b", mesh=m7_bar_b, position=np.array([-0.18, 0.46, -0.05], dtype=np.float32)))
    r7.add_child(ModelNode3D("siren_r", mesh=m7_bar_r, position=np.array([0.18, 0.46, -0.05], dtype=np.float32)))
    for dx, dz in [(-0.48, 0.6), (0.48, 0.6), (-0.48, -0.6), (0.48, -0.6)]:
        w = make_cylinder(radius_top=0.24, radius_bottom=0.24, height=0.16, color=(25, 25, 25), name=f"pw_{dx}_{dz}")
        r7.add_child(ModelNode3D(f"pw_{dx}_{dz}", mesh=w, position=np.array([dx, -0.14, dz], dtype=np.float32), rotation=np.array([0, 0, math.pi/2], dtype=np.float32)))
    models.append((Model3D(concept="Police Car", root=r7),
                   _meta("Police Car", ["police cruiser", "patrol car", "cop car", "law enforcement vehicle"],
                         "Law enforcement patrol cruiser with black-and-white livery and roof-mounted dual red and blue emergency strobe beacons.",
                         ["High-contrast black-and-white police livery", "Roof emergency dual red and blue strobe lightbar", "Reinforced sedan pursuit chassis"], "emergency")))

    # 8. Motorcycle
    r8 = ModelNode3D("root")
    m8_frame = make_cylinder(radius_top=0.05, radius_bottom=0.05, height=1.2, color=(200, 40, 40), name="frame_spine")
    m8_tank = make_ellipsoid(rx=0.18, ry=0.18, rz=0.35, color=(200, 40, 40), name="fuel_tank")
    m8_eng = make_box(dx=0.25, dy=0.3, dz=0.4, color=(120, 120, 130), name="engine_block")
    r8.add_child(ModelNode3D("frame", mesh=m8_frame, position=np.array([0, 0.1, 0], dtype=np.float32), rotation=np.array([0.4, 0, 0], dtype=np.float32)))
    r8.add_child(ModelNode3D("tank", mesh=m8_tank, position=np.array([0, 0.35, 0.15], dtype=np.float32)))
    r8.add_child(ModelNode3D("engine", mesh=m8_eng, position=np.array([0, -0.05, 0], dtype=np.float32)))
    for dz in [0.75, -0.75]:
        w = make_cylinder(radius_top=0.35, radius_bottom=0.35, height=0.12, color=(25, 25, 25), name=f"moto_wheel_{dz}")
        r8.add_child(ModelNode3D(f"wheel_{dz}", mesh=w, position=np.array([0, -0.2, dz], dtype=np.float32), rotation=np.array([0, 0, math.pi/2], dtype=np.float32)))
    models.append((Model3D(concept="Motorcycle", root=r8),
                   _meta("Motorcycle", ["motorbike", "cycle", "motor racer", "chopper"],
                         "Two-wheeled single-track motorized vehicle powered by internal combustion engine with frame, fuel tank, and wheels.",
                         ["Two in-line road wheels on single track", "Central internal combustion engine block", "Ergonomic top fuel tank and saddle frame"], "road")))

    # 9. Bicycle
    r9 = ModelNode3D("root")
    m9_top = make_cylinder(radius_top=0.02, radius_bottom=0.02, height=0.9, color=(40, 160, 220), name="top_tube")
    m9_seat = make_box(dx=0.12, dy=0.04, dz=0.22, color=(30, 30, 35), name="saddle")
    m9_bars = make_cylinder(radius_top=0.02, radius_bottom=0.02, height=0.5, color=(190, 190, 200), name="handlebar")
    r9.add_child(ModelNode3D("tube", mesh=m9_top, position=np.array([0, 0.2, 0], dtype=np.float32), rotation=np.array([math.pi/2, 0, 0], dtype=np.float32)))
    r9.add_child(ModelNode3D("saddle", mesh=m9_seat, position=np.array([0, 0.3, -0.3], dtype=np.float32)))
    r9.add_child(ModelNode3D("bars", mesh=m9_bars, position=np.array([0, 0.45, 0.45], dtype=np.float32), rotation=np.array([0, 0, math.pi/2], dtype=np.float32)))
    for dz in [0.65, -0.65]:
        w = make_torus(r_major=0.38, r_minor=0.025, color=(30, 30, 35), name=f"bike_wheel_{dz}")
        r9.add_child(ModelNode3D(f"wheel_{dz}", mesh=w, position=np.array([0, -0.15, dz], dtype=np.float32), rotation=np.array([0, 0, math.pi/2], dtype=np.float32)))
    models.append((Model3D(concept="Bicycle", root=r9),
                   _meta("Bicycle", ["bike", "pushbike", "cycle", "pedal bicycle"],
                         "Human-powered pedal-driven single-track vehicle with two spoked wheels attached to diamond frame structure.",
                         ["Tubular diamond geometric bicycle frame", "Two slender spoked pneumatic rim wheels", "Elevated handlebar steering and saddle"], "road")))

    # 10. Passenger Airplane
    r10 = ModelNode3D("root")
    m10_fuse = make_cylinder(radius_top=0.35, radius_bottom=0.35, height=2.6, color=(245, 245, 250), name="fuselage")
    m10_nose = make_cone(radius=0.35, height=0.6, color=(245, 245, 250), name="airplane_nose")
    m10_wings = make_box(dx=3.2, dy=0.06, dz=0.6, color=(220, 225, 235), name="swept_wings")
    m10_tail = make_box(dx=0.06, dy=0.6, dz=0.45, color=(210, 40, 40), name="vertical_stabilizer")
    r10.add_child(ModelNode3D("fuse", mesh=m10_fuse, rotation=np.array([math.pi/2, 0, 0], dtype=np.float32)))
    r10.add_child(ModelNode3D("nose", mesh=m10_nose, position=np.array([0, 0, 1.6], dtype=np.float32), rotation=np.array([math.pi/2, 0, 0], dtype=np.float32)))
    r10.add_child(ModelNode3D("wings", mesh=m10_wings, position=np.array([0, 0, 0.2], dtype=np.float32)))
    r10.add_child(ModelNode3D("tail", mesh=m10_tail, position=np.array([0, 0.45, -1.1], dtype=np.float32)))
    models.append((Model3D(concept="Airplane", root=r10),
                   _meta("Airplane", ["passenger jet", "airliner", "commercial aircraft", "aeroplane"],
                         "Fixed-wing commercial passenger aircraft with pressurized cylindrical fuselage, main aerofoil wings, and empennage tail.",
                         ["Long cylindrical pressurized passenger fuselage", "Wide aerofoil lift wings", "Vertical tail stabilizer fin with airline livery"], "aviation")))

    # 11. Fighter Jet
    r11 = ModelNode3D("root")
    m11_fuse = make_cone(radius=0.3, height=2.4, color=(140, 145, 155), roughness=0.3, name="jet_fuselage")
    m11_delta = make_box(dx=2.0, dy=0.04, dz=0.9, color=(130, 135, 145), name="delta_wings")
    m11_canopy = make_ellipsoid(rx=0.15, ry=0.15, rz=0.4, color=(210, 180, 50), opacity=0.8, name="canopy_tint")
    r11.add_child(ModelNode3D("fuse", mesh=m11_fuse, rotation=np.array([-math.pi/2, 0, 0], dtype=np.float32)))
    r11.add_child(ModelNode3D("delta", mesh=m11_delta, position=np.array([0, 0, -0.2], dtype=np.float32)))
    r11.add_child(ModelNode3D("canopy", mesh=m11_canopy, position=np.array([0, 0.2, 0.35], dtype=np.float32)))
    models.append((Model3D(concept="Fighter Jet", root=r11),
                   _meta("Fighter Jet", ["supersonic fighter", "combat aircraft", "jet fighter", "stealth jet"],
                         "High-speed tactical military jet aircraft featuring sharp delta-wing planform, bubble canopy, and afterburning exhaust.",
                         ["Sleek aerodynamic supersonic delta wings", "Tinted bubble pilot cockpit canopy", "Needle-point supersonic radome nose"], "aviation")))

    # 12. Helicopter
    r12 = ModelNode3D("root")
    m12_fuse = make_ellipsoid(rx=0.45, ry=0.5, rz=0.9, color=(40, 60, 140), name="helo_cabin")
    m12_boom = make_cylinder(radius_top=0.08, radius_bottom=0.08, height=1.3, color=(40, 60, 140), name="tail_boom")
    m12_rotor = make_box(dx=2.6, dy=0.03, dz=0.14, color=(30, 30, 35), name="main_rotor_blades")
    r12.add_child(ModelNode3D("fuse", mesh=m12_fuse))
    r12.add_child(ModelNode3D("boom", mesh=m12_boom, position=np.array([0, 0.15, -1.0], dtype=np.float32), rotation=np.array([math.pi/2, 0, 0], dtype=np.float32)))
    r12.add_child(ModelNode3D("rotor", mesh=m12_rotor, position=np.array([0, 0.58, 0], dtype=np.float32)))
    models.append((Model3D(concept="Helicopter", root=r12),
                   _meta("Helicopter", ["chopper", "rotary aircraft", "whirlybird", "helo"],
                         "Rotorcraft in which lift and thrust are supplied by horizontally-spinning main rotors, with tail boom stabilizer.",
                         ["Aerodynamic aerodynamic cabin fuselage", "Large horizontal overhead main rotor blades", "Elongated tail boom with anti-torque rotor"], "aviation")))

    # 13. Drone Quadcopter
    r13 = ModelNode3D("root")
    m13_hub = make_box(dx=0.35, dy=0.15, dz=0.35, color=(40, 40, 45), name="drone_hub")
    m13_arm1 = make_cylinder(radius_top=0.03, radius_bottom=0.03, height=1.4, color=(180, 180, 190), name="arm_cross_1")
    m13_arm2 = make_cylinder(radius_top=0.03, radius_bottom=0.03, height=1.4, color=(180, 180, 190), name="arm_cross_2")
    r13.add_child(ModelNode3D("hub", mesh=m13_hub))
    r13.add_child(ModelNode3D("arm1", mesh=m13_arm1, rotation=np.array([0, math.pi/4, math.pi/2], dtype=np.float32)))
    r13.add_child(ModelNode3D("arm2", mesh=m13_arm2, rotation=np.array([0, -math.pi/4, math.pi/2], dtype=np.float32)))
    for dx, dz in [(-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5), (0.5, 0.5)]:
        prop = make_box(dx=0.35, dy=0.015, dz=0.06, color=(220, 40, 40), name=f"prop_{dx}_{dz}")
        r13.add_child(ModelNode3D(f"prop_{dx}_{dz}", mesh=prop, position=np.array([dx, 0.1, dz], dtype=np.float32)))
    models.append((Model3D(concept="Drone", root=r13),
                   _meta("Drone", ["quadcopter", "UAV", "multirotor", "camera drone"],
                         "Unmanned aerial vehicle with four horizontal rotor arms generating vertical lift and precision maneuvering.",
                         ["Central flight controller avionics hub", "Four radial cross arms in X-configuration", "Four independent vertical rotor propellers"], "aviation")))

    # 14. Hot Air Balloon
    r14 = ModelNode3D("root")
    m14_env = make_sphere(radius=0.9, color=(30, 140, 230), roughness=0.6, name="balloon_envelope")  # Bright orange BGR
    m14_basket = make_box(dx=0.4, dy=0.35, dz=0.4, color=(60, 95, 140), name="wicker_basket")
    r14.add_child(ModelNode3D("envelope", mesh=m14_env, position=np.array([0, 0.6, 0], dtype=np.float32), scale=np.array([1.0, 1.25, 1.0], dtype=np.float32)))
    r14.add_child(ModelNode3D("basket", mesh=m14_basket, position=np.array([0, -0.8, 0], dtype=np.float32)))
    models.append((Model3D(concept="Hot Air Balloon", root=r14),
                   _meta("Hot Air Balloon", ["aerostat", "hot air aerostat", "thermal balloon"],
                         "Lighter-than-air aerostat consisting of a large heated air fabric envelope bag and suspended passenger wicker basket.",
                         ["Large tear-shaped thermal air envelope", "Suspended passenger wicker gondola basket", "Vertical buoyancy aerostat principle"], "aviation")))

    # 15. Speedboat
    r15 = ModelNode3D("root")
    m15_hull = make_box(dx=0.8, dy=0.3, dz=1.8, color=(240, 245, 250), name="boat_hull")
    m15_bow = make_cone(radius=0.4, height=0.7, color=(240, 245, 250), name="v_bow")
    m15_screen = make_box(dx=0.7, dy=0.18, dz=0.04, color=(190, 160, 50), opacity=0.7, name="windshield")
    r15.add_child(ModelNode3D("hull", mesh=m15_hull))
    r15.add_child(ModelNode3D("bow", mesh=m15_bow, position=np.array([0, 0, 1.25], dtype=np.float32), rotation=np.array([math.pi/2, 0, 0], dtype=np.float32)))
    r15.add_child(ModelNode3D("screen", mesh=m15_screen, position=np.array([0, 0.22, 0.2], dtype=np.float32), rotation=np.array([-0.3, 0, 0], dtype=np.float32)))
    models.append((Model3D(concept="Speedboat", root=r15),
                   _meta("Speedboat", ["motorboat", "powerboat", "runabout", "speedboat craft"],
                         "High-speed watercraft featuring sharp V-shaped hydrodynamic planing hull, tinted wrap-around windshield, and stern motor.",
                         ["Sharp hydrodynamic V-hull entry bow", "Tinted aerodynamic cockpit windshield", "Planing hull surface for marine velocity"], "maritime")))

    # 16. Sailboat
    r16 = ModelNode3D("root")
    m16_hull = make_ellipsoid(rx=0.4, ry=0.25, rz=1.0, color=(240, 245, 250), name="sailboat_hull")
    m16_mast = make_cylinder(radius_top=0.03, radius_bottom=0.03, height=1.6, color=(160, 160, 165), name="mast_pole")
    m16_sail = make_box(dx=0.02, dy=1.2, dz=0.7, color=(250, 250, 255), name="mainsail")
    r16.add_child(ModelNode3D("hull", mesh=m16_hull))
    r16.add_child(ModelNode3D("mast", mesh=m16_mast, position=np.array([0, 0.85, 0.1], dtype=np.float32)))
    r16.add_child(ModelNode3D("sail", mesh=m16_sail, position=np.array([0, 0.85, -0.3], dtype=np.float32)))
    models.append((Model3D(concept="Sailboat", root=r16),
                   _meta("Sailboat", ["sailing yacht", "monohull", "sail vessel", "sloop"],
                         "Wind-propelled watercraft with displacement monohull, tall vertical mast spar, and broad triangular mainsail cloth.",
                         ["Sleek marine displacement monohull", "Tall vertical aluminum/carbon mast spar", "High-aspect triangular canvas mainsail"], "maritime")))

    # 17. Submarine
    r17 = ModelNode3D("root")
    m17_hull = make_cylinder(radius_top=0.38, radius_bottom=0.38, height=2.2, color=(40, 45, 50), roughness=0.5, name="pressure_hull")
    m17_nose = make_sphere(radius=0.38, color=(40, 45, 50), name="sonar_bow")
    m17_sail = make_box(dx=0.18, dy=0.4, dz=0.45, color=(40, 45, 50), name="conning_tower")
    r17.add_child(ModelNode3D("hull", mesh=m17_hull, rotation=np.array([math.pi/2, 0, 0], dtype=np.float32)))
    r17.add_child(ModelNode3D("bow", mesh=m17_nose, position=np.array([0, 0, 1.1], dtype=np.float32)))
    r17.add_child(ModelNode3D("sail", mesh=m17_sail, position=np.array([0, 0.45, 0.2], dtype=np.float32)))
    models.append((Model3D(concept="Submarine", root=r17),
                   _meta("Submarine", ["naval sub", "underwater vessel", "submersible", "u-boat"],
                         "Naval watercraft capable of independent underwater operation with cylindrical pressure hull and elevated conning tower sail.",
                         ["Elongated cylindrical pressure hull fuselage", "Elevated conning tower sail with periscope mast", "Submersible naval architecture"], "maritime")))

    # 18. Bullet Train
    r18 = ModelNode3D("root")
    m18_body = make_box(dx=0.8, dy=0.75, dz=2.4, color=(245, 245, 250), name="train_coach")
    m18_nose = make_cone(radius=0.4, height=0.9, color=(245, 245, 250), name="needle_nose")
    m18_stripe = make_box(dx=0.82, dy=0.1, dz=2.6, color=(210, 40, 40), name="speed_stripe")  # Blue racing stripe
    r18.add_child(ModelNode3D("body", mesh=m18_body))
    r18.add_child(ModelNode3D("nose", mesh=m18_nose, position=np.array([0, -0.05, 1.6], dtype=np.float32), rotation=np.array([math.pi/2, 0, 0], dtype=np.float32)))
    r18.add_child(ModelNode3D("stripe", mesh=m18_stripe, position=np.array([0, -0.1, 0.2], dtype=np.float32)))
    models.append((Model3D(concept="Bullet Train", root=r18),
                   _meta("Bullet Train", ["shinkansen", "high-speed train", "maglev", "tgy"],
                         "High-speed aerodynamic passenger railway locomotive featuring a streamlined long needle-nose to reduce tunnel boom.",
                         ["Aerodynamic elongated needle-nose front taper", "Smooth streamlined high-speed train carriage", "Distinctive high-speed livery striping"], "rail")))

    # 19. Cargo Ship
    r19 = ModelNode3D("root")
    m19_hull = make_box(dx=1.0, dy=0.5, dz=2.4, color=(35, 35, 40), name="container_hull")
    m19_deck = make_box(dx=0.75, dy=0.6, dz=0.5, color=(240, 240, 245), name="stern_bridge")
    m19_c1 = make_box(dx=0.8, dy=0.25, dz=0.6, color=(30, 160, 50), name="container_green")
    m19_c2 = make_box(dx=0.8, dy=0.25, dz=0.6, color=(30, 40, 210), name="container_red")
    r19.add_child(ModelNode3D("hull", mesh=m19_hull))
    r19.add_child(ModelNode3D("bridge", mesh=m19_deck, position=np.array([0, 0.45, -0.8], dtype=np.float32)))
    r19.add_child(ModelNode3D("c1", mesh=m19_c1, position=np.array([0, 0.35, 0.3], dtype=np.float32)))
    r19.add_child(ModelNode3D("c2", mesh=m19_c2, position=np.array([0, 0.35, -0.35], dtype=np.float32)))
    models.append((Model3D(concept="Cargo Ship", root=r19),
                   _meta("Cargo Ship", ["container ship", "freighter", "merchant vessel", "cargo vessel"],
                         "Commercial maritime merchant freighter with broad cargo hull carrying intermodal shipping containers and stern bridge.",
                         ["Broad heavy displacement merchant hull", "Multi-tiered stern navigation bridge deck", "Stacked intermodal ISO shipping containers"], "maritime")))

    # 20. Locomotive
    r20 = ModelNode3D("root")
    m20_body = make_box(dx=0.9, dy=0.8, dz=2.2, color=(30, 80, 150), name="diesel_loco")  # Blue engine
    m20_cab = make_box(dx=0.88, dy=0.45, dz=0.6, color=(40, 40, 45), name="operator_cab")
    m20_horn = make_cylinder(radius_top=0.06, radius_bottom=0.03, height=0.25, color=(200, 200, 210), name="roof_horn")
    r20.add_child(ModelNode3D("body", mesh=m20_body))
    r20.add_child(ModelNode3D("cab", mesh=m20_cab, position=np.array([0, 0.5, 0.6], dtype=np.float32)))
    r20.add_child(ModelNode3D("horn", mesh=m20_horn, position=np.array([0, 0.78, 0.6], dtype=np.float32), rotation=np.array([math.pi/2, 0, 0], dtype=np.float32)))
    models.append((Model3D(concept="Locomotive", root=r20),
                   _meta("Locomotive", ["train engine", "diesel locomotive", "railroad engine", "rail locomotive"],
                         "Railway vehicle that provides motive power for a train, with heavy diesel-electric hood chassis and elevated cab.",
                         ["Heavy diesel-electric hood engine body", "Elevated forward engineer observation cab", "Industrial railroad motive power proportions"], "rail")))

    return models
