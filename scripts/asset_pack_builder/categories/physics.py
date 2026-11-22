"""
scripts/asset_pack_builder/categories/physics.py — 20 Physics and Mathematics Models.
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
    make_helix,
    make_platonic_solid,
    make_ring_disc,
    make_sphere,
    make_torus,
)


def _meta(name: str, aliases: List[str], desc: str, features: List[str], subcat: str = "mechanics") -> Dict:
    return {
        "name": name,
        "aliases": aliases,
        "category": "Physics and Mathematics",
        "subcategory": subcat,
        "description": desc,
        "distinguishing_features": features,
        "educational_purpose": f"Visualizes the 3D physical and mathematical properties of {name}.",
        "license": "CC0-1.0",
        "author": "Magic Mirror Spatial AI Team",
        "source": "Magic Mirror 3D Asset Library",
        "attribution": "Curated for Magic Mirror under CC0 1.0 Universal",
        "download_date": "2026-11-22",
    }


def get_models() -> List[Tuple[Model3D, Dict]]:
    models = []

    # 1. Proton
    r1 = ModelNode3D("root")
    m1_core = make_sphere(radius=0.7, color=(40, 40, 220), roughness=0.3, name="proton_sphere")  # Bright red BGR
    m1_h = make_box(dx=0.5, dy=0.12, dz=0.12, color=(240, 240, 240), name="plus_h")
    m1_v = make_box(dx=0.12, dy=0.5, dz=0.12, color=(240, 240, 240), name="plus_v")
    r1.add_child(ModelNode3D("core", mesh=m1_core))
    r1.add_child(ModelNode3D("plus_h", mesh=m1_h, position=np.array([0, 0, 0.72], dtype=np.float32)))
    r1.add_child(ModelNode3D("plus_v", mesh=m1_v, position=np.array([0, 0, 0.72], dtype=np.float32)))
    models.append((Model3D(concept="Proton", root=r1),
                   _meta("Proton", ["p+", "proton particle", "positive nucleon"],
                         "Positively charged subatomic nucleon with distinctive red coloration and positive charge sign.",
                         ["Spherical subatomic nucleon", "Red standard color-coding", "White positive (+) charge marker"], "subatomic")))

    # 2. Neutron
    r2 = ModelNode3D("root")
    m2_core = make_sphere(radius=0.7, color=(160, 160, 160), roughness=0.5, name="neutron_sphere")  # Grey neutral
    m2_ring = make_torus(r_major=0.74, r_minor=0.04, color=(100, 100, 100), name="neutral_equator")
    r2.add_child(ModelNode3D("core", mesh=m2_core))
    r2.add_child(ModelNode3D("equator", mesh=m2_ring))
    models.append((Model3D(concept="Neutron", root=r2),
                   _meta("Neutron", ["n0", "neutron particle", "neutral nucleon"],
                         "Electrically neutral hadron found in atomic nuclei, composed of two down quarks and one up quark.",
                         ["Spherical neutral nucleon", "Grey neutral color-coding", "Distinct zero-charge equatorial guide"], "subatomic")))

    # 3. Electron
    r3 = ModelNode3D("root")
    m3_core = make_sphere(radius=0.4, color=(220, 120, 30), roughness=0.2, name="electron_sphere")  # Electric cyan-blue BGR
    m3_minus = make_box(dx=0.35, dy=0.08, dz=0.08, color=(245, 245, 245), name="minus_bar")
    r3.add_child(ModelNode3D("core", mesh=m3_core))
    r3.add_child(ModelNode3D("minus", mesh=m3_minus, position=np.array([0, 0, 0.42], dtype=np.float32)))
    models.append((Model3D(concept="Electron", root=r3),
                   _meta("Electron", ["e-", "beta particle", "elementary lepton"],
                         "Subatomic elementary particle with negative elementary electric charge, represented as small blue sphere with minus marker.",
                         ["Compact elementary particle radius", "Electric blue color-coding", "White negative (-) charge bar"], "subatomic")))

    # 4. Atomic Nucleus
    r4 = ModelNode3D("root")
    p_pos = [[0.3, 0.3, 0.3], [-0.3, -0.3, 0.3], [0.3, -0.3, -0.3], [-0.3, 0.3, -0.3]]
    n_pos = [[-0.3, -0.3, -0.3], [0.3, 0.3, -0.3], [-0.3, 0.3, 0.3], [0.3, -0.3, 0.3]]
    for idx, pos in enumerate(p_pos):
        m_p = make_sphere(radius=0.35, color=(40, 40, 210), name=f"nuc_p_{idx}")
        r4.add_child(ModelNode3D(f"p_{idx}", mesh=m_p, position=np.array(pos, dtype=np.float32)))
    for idx, pos in enumerate(n_pos):
        m_n = make_sphere(radius=0.35, color=(160, 160, 160), name=f"nuc_n_{idx}")
        r4.add_child(ModelNode3D(f"n_{idx}", mesh=m_n, position=np.array(pos, dtype=np.float32)))
    models.append((Model3D(concept="Atomic Nucleus", root=r4),
                   _meta("Atomic Nucleus", ["atomic core", "nuclear cluster", "nucleon cluster"],
                         "Dense central core of an atom consisting of tightly bound protons and neutrons held by the strong force.",
                         ["Tightly bound nucleon cluster", "Red protons and grey neutrons", "Strong nuclear force demonstration"], "subatomic")))

    # 5. Bar Magnet
    r5 = ModelNode3D("root")
    m5_n = make_box(dx=0.8, dy=0.35, dz=0.35, color=(30, 30, 220), name="north_pole")   # Red (North)
    m5_s = make_box(dx=0.8, dy=0.35, dz=0.35, color=(220, 40, 40), name="south_pole")   # Blue (South)
    r5.add_child(ModelNode3D("north", mesh=m5_n, position=np.array([0.4, 0, 0], dtype=np.float32)))
    r5.add_child(ModelNode3D("south", mesh=m5_s, position=np.array([-0.4, 0, 0], dtype=np.float32)))
    models.append((Model3D(concept="Bar Magnet", root=r5),
                   _meta("Bar Magnet", ["permanent magnet", "dipole magnet", "magnetic bar"],
                         "Rectangular permanent magnetic dipole featuring marked North (red) and South (blue) magnetic poles.",
                         ["Bicolor magnetic dipole", "Red North pole segment", "Blue South pole segment"], "electromagnetism")))

    # 6. Horseshoe Magnet
    r6 = ModelNode3D("root")
    m6_arch = make_cylinder(radius_top=0.7, radius_bottom=0.7, height=0.3, segs=24, color=(180, 180, 180), name="u_arch")
    m6_leg_n = make_box(dx=0.3, dy=0.8, dz=0.3, color=(30, 30, 220), name="n_leg")
    m6_leg_s = make_box(dx=0.3, dy=0.8, dz=0.3, color=(220, 40, 40), name="s_leg")
    r6.add_child(ModelNode3D("arch", mesh=m6_arch, position=np.array([0, 0.5, 0], dtype=np.float32), rotation=np.array([math.pi/2, 0, 0], dtype=np.float32)))
    r6.add_child(ModelNode3D("leg_n", mesh=m6_leg_n, position=np.array([0.5, 0, 0], dtype=np.float32)))
    r6.add_child(ModelNode3D("leg_s", mesh=m6_leg_s, position=np.array([-0.5, 0, 0], dtype=np.float32)))
    models.append((Model3D(concept="Horseshoe Magnet", root=r6),
                   _meta("Horseshoe Magnet", ["U magnet", "horseshoe dipole"],
                         "U-shaped permanent magnet concentrating magnetic flux between closely spaced opposite poles.",
                         ["Characteristic U-shape geometry", "Red North and blue South tips", "Concentrated air gap field"], "electromagnetism")))

    # 7. Magnetic Field
    r7 = ModelNode3D("root")
    m7_mag = make_box(dx=0.7, dy=0.25, dz=0.25, color=(180, 50, 50), name="center_magnet")
    r7.add_child(ModelNode3D("center_mag", mesh=m7_mag))
    for rx, rz in [(0.9, 0.4), (1.4, 0.7), (1.9, 1.1)]:
        t = make_torus(r_major=rx, r_minor=0.03, segs_major=24, color=(240, 200, 80), name=f"field_loop_{rx}")
        r7.add_child(ModelNode3D(f"loop_{rx}", mesh=t, scale=np.array([1.0, 1.0, rz/rx], dtype=np.float32)))
    models.append((Model3D(concept="Magnetic Field", root=r7),
                   _meta("Magnetic Field", ["magnetic flux lines", "dipole field lines", "magnetic demonstration"],
                         "Visual demonstration of magnetic field lines (flux loops) emerging from north pole and curving to south pole.",
                         ["Central dipole magnet", "Elliptical concentric magnetic flux loops", "Dipole field distribution"], "electromagnetism")))

    # 8. Pendulum
    r8 = ModelNode3D("root")
    m8_stand = make_cylinder(radius_top=0.05, radius_bottom=0.05, height=1.6, color=(140, 140, 140), name="stand")
    m8_wire = make_cylinder(radius_top=0.015, radius_bottom=0.015, height=1.0, color=(220, 220, 220), name="wire")
    m8_bob = make_sphere(radius=0.25, color=(40, 180, 230), roughness=0.3, metallic=0.7, name="bob")  # Brass/copper bob
    r8.add_child(ModelNode3D("stand", mesh=m8_stand, position=np.array([0, 0.8, 0], dtype=np.float32)))
    r8.add_child(ModelNode3D("wire", mesh=m8_wire, position=np.array([0.3, 0.6, 0], dtype=np.float32), rotation=np.array([0, 0, 0.25], dtype=np.float32)))
    r8.add_child(ModelNode3D("bob", mesh=m8_bob, position=np.array([0.43, 0.1, 0], dtype=np.float32)))
    models.append((Model3D(concept="Pendulum", root=r8),
                   _meta("Pendulum", ["simple pendulum", "harmonic oscillator", "clock pendulum"],
                         "Classic harmonic oscillator consisting of a mass bob suspended from a pivot by a light string.",
                         ["Vertical support pivot", "Inelastic suspension wire", "Spherical metallic pendulum bob"], "mechanics")))

    # 9. Spring
    r9 = ModelNode3D("root")
    m9_coil = make_helix(radius=0.3, pitch=0.25, turns=5.0, tube_radius=0.04, color=(170, 170, 180), name="coil")
    r9.add_child(ModelNode3D("coil", mesh=m9_coil))
    models.append((Model3D(concept="Spring", root=r9),
                   _meta("Spring", ["helical spring", "coil spring", "Hooke spring"],
                         "Mechanical elastic coil demonstrating Hooke's Law of elasticity (F = -kx).",
                         ["Continuous helical coil winding", "Uniform pitch and diameter", "Metallic wire cross-section"], "mechanics")))

    # 10. Gear Pair
    r10 = ModelNode3D("root")
    m10_g1 = make_cylinder(radius_top=0.7, radius_bottom=0.7, height=0.2, segs=16, color=(180, 150, 80), name="gear1")
    m10_g2 = make_cylinder(radius_top=0.4, radius_bottom=0.4, height=0.2, segs=12, color=(140, 140, 150), name="gear2")
    r10.add_child(ModelNode3D("gear_driver", mesh=m10_g1, position=np.array([-0.6, 0, 0], dtype=np.float32)))
    r10.add_child(ModelNode3D("gear_driven", mesh=m10_g2, position=np.array([0.5, 0, 0], dtype=np.float32)))
    models.append((Model3D(concept="Gears", root=r10),
                   _meta("Gears", ["gear pair", "interlocking gears", "spur gears", "transmission"],
                         "Pair of intermeshing toothed spur gears demonstrating mechanical advantage and rotational transmission.",
                         ["Driver and driven spur gear cylinders", "Segmented perimeter gear teeth", "Distinct gear reduction ratio"], "mechanics")))

    # 11. Pulley
    r11 = ModelNode3D("root")
    m11_wheel = make_cylinder(radius_top=0.6, radius_bottom=0.6, height=0.2, color=(120, 130, 150), name="wheel")
    m11_axle = make_cylinder(radius_top=0.1, radius_bottom=0.1, height=0.4, color=(200, 200, 210), name="axle")
    m11_bracket = make_box(dx=0.15, dy=1.2, dz=0.3, color=(80, 80, 90), name="bracket")
    r11.add_child(ModelNode3D("wheel", mesh=m11_wheel, rotation=np.array([math.pi/2, 0, 0], dtype=np.float32)))
    r11.add_child(ModelNode3D("axle", mesh=m11_axle, rotation=np.array([math.pi/2, 0, 0], dtype=np.float32)))
    r11.add_child(ModelNode3D("bracket", mesh=m11_bracket, position=np.array([0, 0.4, 0], dtype=np.float32)))
    models.append((Model3D(concept="Pulley", root=r11),
                   _meta("Pulley", ["simple machine pulley", "sheave", "pulley system"],
                         "Grooved wheel turned by a cord or rope, used to change direction of force and gain mechanical advantage.",
                         ["Grooved rim wheel sheave", "Central axle pin and suspension bracket", "Mechanical advantage demonstration"], "mechanics")))

    # 12. Lever
    r12 = ModelNode3D("root")
    m12_beam = make_box(dx=2.2, dy=0.08, dz=0.3, color=(160, 110, 60), name="beam")  # Wood beam
    m12_fulcrum = make_cone(radius=0.25, height=0.4, segs=16, color=(100, 100, 110), name="fulcrum")
    m12_load = make_box(dx=0.3, dy=0.3, dz=0.3, color=(50, 50, 180), name="load")
    r12.add_child(ModelNode3D("beam", mesh=m12_beam, position=np.array([0, 0.24, 0], dtype=np.float32), rotation=np.array([0, 0, -0.1], dtype=np.float32)))
    r12.add_child(ModelNode3D("fulcrum", mesh=m12_fulcrum, position=np.array([0, 0, 0], dtype=np.float32)))
    r12.add_child(ModelNode3D("load", mesh=m12_load, position=np.array([-0.8, 0.45, 0], dtype=np.float32)))
    models.append((Model3D(concept="Lever", root=r12),
                   _meta("Lever", ["simple lever", "class 1 lever", "fulcrum beam"],
                         "First-class lever demonstrating torque equilibrium where effort arm balances load arm over a pivot fulcrum.",
                         ["Rigid horizontal lever beam", "Triangular pivot fulcrum", "Counterbalancing load mass"], "mechanics")))

    # 13. Inclined Plane
    r13 = ModelNode3D("root")
    # Triangular wedge ramp + block
    m13_ramp = make_box(dx=1.8, dy=0.1, dz=0.6, color=(180, 150, 110), name="ramp_board")
    m13_base = make_box(dx=1.8, dy=0.1, dz=0.6, color=(100, 100, 110), name="ramp_base")
    m13_block = make_box(dx=0.35, dy=0.25, dz=0.35, color=(40, 140, 220), name="sliding_block")
    r13.add_child(ModelNode3D("ramp", mesh=m13_ramp, position=np.array([0, 0.4, 0], dtype=np.float32), rotation=np.array([0, 0, 0.35], dtype=np.float32)))
    r13.add_child(ModelNode3D("base", mesh=m13_base, position=np.array([0, -0.1, 0], dtype=np.float32)))
    r13.add_child(ModelNode3D("block", mesh=m13_block, position=np.array([0.2, 0.65, 0], dtype=np.float32), rotation=np.array([0, 0, 0.35], dtype=np.float32)))
    models.append((Model3D(concept="Inclined Plane", root=r13),
                   _meta("Inclined Plane", ["ramp", "slope", "simple machine plane"],
                         "Sloping ramp enabling heavy loads to be raised using less applied force over a longer distance.",
                         ["Tilted inclined plane surface", "Sliding friction block", "Horizontal ground reference"], "mechanics")))

    # 14. Optical Prism
    r14 = ModelNode3D("root")
    m14_prism = make_cylinder(radius_top=0.6, radius_bottom=0.6, height=1.0, segs=3, color=(240, 240, 250), opacity=0.7, name="prism_glass")
    m14_beam = make_cylinder(radius_top=0.03, radius_bottom=0.03, height=1.2, color=(255, 255, 255), name="light_ray")
    r14.add_child(ModelNode3D("prism", mesh=m14_prism))
    r14.add_child(ModelNode3D("ray", mesh=m14_beam, position=np.array([-0.8, 0, 0], dtype=np.float32), rotation=np.array([0, 0, math.pi/2], dtype=np.float32)))
    models.append((Model3D(concept="Prism", root=r14),
                   _meta("Prism", ["optical prism", "triangular prism", "refraction prism"],
                         "Transparent triangular optical prism refracting and dispersing incident light into constituent wavelengths.",
                         ["Triangular prism cross-section", "Translucent optical glass medium", "Incident light refraction beam"], "optics")))

    # 15. Wave
    r15 = ModelNode3D("root")
    # Multiple connected spheres creating sinusoidal wave
    for i in range(25):
        x = -1.2 + i * 0.1
        y = 0.45 * math.sin(x * 3.0)
        col = (int(50 + 8 * i), int(120 + 4 * i), 220)
        s = make_sphere(radius=0.08, color=col, name=f"wave_pt_{i}")
        r15.add_child(ModelNode3D(f"w_{i}", mesh=s, position=np.array([x, y, 0], dtype=np.float32)))
    models.append((Model3D(concept="Wave", root=r15),
                   _meta("Wave", ["sine wave", "transverse wave", "harmonic wave", "oscillation"],
                         "Sinusoidal transverse wave demonstrating wavelength, amplitude, crests, and troughs of harmonic propagation.",
                         ["Sinusoidal spatial profile", "Periodic crests and troughs", "Wavelength oscillation demonstration"], "waves")))

    # 16. Cube
    m16 = make_box(dx=1.0, dy=1.0, dz=1.0, color=(70, 150, 220), roughness=0.3, name="cube_mesh")
    models.append((Model3D(concept="Cube", root=ModelNode3D("root", mesh=m16)),
                   _meta("Cube", ["regular hexahedron", "geometric cube", "3d square"],
                         "Platonic solid with six congruent square faces, twelve straight edges, and eight vertices.",
                         ["Six congruent square faces", "Right angles (90 deg) throughout", "Platonic regular hexahedron"], "geometry")))

    # 17. Tetrahedron
    m17 = make_platonic_solid(solid_type="tetrahedron", radius=1.0, color=(220, 100, 60), name="tetra_mesh")
    models.append((Model3D(concept="Tetrahedron", root=ModelNode3D("root", mesh=m17)),
                   _meta("Tetrahedron", ["regular tetrahedron", "triangular pyramid", "4-faced polyhedron"],
                         "Simplest Platonic solid having four congruent equilateral triangular faces and four vertices.",
                         ["Four equilateral triangular faces", "Four vertices and six edges", "Self-dual Platonic solid"], "geometry")))

    # 18. Octahedron
    m18 = make_platonic_solid(solid_type="octahedron", radius=1.0, color=(60, 200, 120), name="octa_mesh")
    models.append((Model3D(concept="Octahedron", root=ModelNode3D("root", mesh=m18)),
                   _meta("Octahedron", ["regular octahedron", "8-faced polyhedron"],
                         "Platonic solid having eight congruent equilateral triangular faces and six vertices, dual of the cube.",
                         ["Eight equilateral triangular faces", "Six vertices and twelve edges", "Platonic octahedron geometry"], "geometry")))

    # 19. Icosahedron
    m19 = make_platonic_solid(solid_type="icosahedron", radius=1.0, color=(160, 70, 210), name="icosa_mesh")
    models.append((Model3D(concept="Icosahedron", root=ModelNode3D("root", mesh=m19)),
                   _meta("Icosahedron", ["regular icosahedron", "20-faced polyhedron"],
                         "Platonic solid possessing twenty congruent equilateral triangular faces and twelve vertices.",
                         ["Twenty equilateral triangular faces", "Twelve vertices with five faces meeting at each", "Golden ratio proportions"], "geometry")))

    # 20. 3D Coordinate Axes
    r20 = ModelNode3D("root")
    # X axis = Red, Y axis = Green, Z axis = Blue
    x_shaft = make_cylinder(radius_top=0.03, radius_bottom=0.03, height=1.0, color=(40, 40, 230), name="axis_x")
    x_cone = make_cone(radius=0.08, height=0.2, color=(40, 40, 230), name="cone_x")
    y_shaft = make_cylinder(radius_top=0.03, radius_bottom=0.03, height=1.0, color=(40, 220, 40), name="axis_y")
    y_cone = make_cone(radius=0.08, height=0.2, color=(40, 220, 40), name="cone_y")
    z_shaft = make_cylinder(radius_top=0.03, radius_bottom=0.03, height=1.0, color=(230, 40, 40), name="axis_z")
    z_cone = make_cone(radius=0.08, height=0.2, color=(230, 40, 40), name="cone_z")

    r20.add_child(ModelNode3D("x_s", mesh=x_shaft, position=np.array([0.5, 0, 0], dtype=np.float32), rotation=np.array([0, 0, -math.pi/2], dtype=np.float32)))
    r20.add_child(ModelNode3D("x_c", mesh=x_cone, position=np.array([1.0, 0, 0], dtype=np.float32), rotation=np.array([0, 0, -math.pi/2], dtype=np.float32)))
    r20.add_child(ModelNode3D("y_s", mesh=y_shaft, position=np.array([0, 0.5, 0], dtype=np.float32)))
    r20.add_child(ModelNode3D("y_c", mesh=y_cone, position=np.array([0, 1.0, 0], dtype=np.float32)))
    r20.add_child(ModelNode3D("z_s", mesh=z_shaft, position=np.array([0, 0, 0.5], dtype=np.float32), rotation=np.array([math.pi/2, 0, 0], dtype=np.float32)))
    r20.add_child(ModelNode3D("z_c", mesh=z_cone, position=np.array([0, 0, 1.0], dtype=np.float32), rotation=np.array([math.pi/2, 0, 0], dtype=np.float32)))
    models.append((Model3D(concept="Coordinate Axes", root=r20),
                   _meta("Coordinate Axes", ["3D axes", "Cartesian coordinate system", "XYZ axes", "spatial coordinates"],
                         "Standard 3D Cartesian coordinate frame with orthogonal X (red), Y (green), and Z (blue) directional vectors.",
                         ["Three orthogonal 90-degree axes", "Standard RGB (XYZ) vector color-coding", "Directional cone arrowheads"], "geometry")))

    return models
