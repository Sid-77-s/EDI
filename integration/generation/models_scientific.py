"""
integration/generation/models_scientific.py — Structured 3D Scientific & Technical Model Templates.

Implements true 3D models with component hierarchies and physical/biological/kinematic animations:
1. Atom (Nucleus + multi-planar electron orbital rings + orbiting electron spheres)
2. Molecule (Ball-and-stick with correct CPK atomic identities, bond angles, and 3D geometry: H2O, CO2, CH4, NaCl, etc.)
3. Solar System (Sun + orbiting planets with Keplerian orbital velocities + Saturn rings)
4. Wave (3D undulating wave surface mesh with traveling oscillation y = A sin(kx - wt))
5. Human Heart (Anatomical ventricles, atria, aorta with rhythmic cardiac contraction pulsation)
6. Mechanical Assembly (Interlocking spur gears + piston crankshaft linkage kinematics)
7. Electrical Circuit (Conductive loop with battery, resistor, LED, and animated circulating electron current)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple
import numpy as np

from integration.generation.geometry_primitives import (
    create_box,
    create_cylinder,
    create_gear,
    create_torus,
    create_uv_sphere,
)
from integration.generation.model3d import Material3D, Mesh3D, Model3D, ModelNode3D

# CPK Atomic Coloring (BGR)
CPK_COLORS: Dict[str, Tuple[int, int, int]] = {
    "H": (245, 245, 245),   # White
    "C": (50, 50, 50),       # Dark Gray
    "N": (220, 50, 50),      # Blue
    "O": (30, 30, 230),      # Red
    "F": (50, 220, 100),     # Green
    "CL": (40, 200, 40),     # Bright Green
    "NA": (180, 50, 180),    # Violet
    "P": (20, 140, 255),     # Orange
    "S": (30, 220, 220),     # Yellow
    "FE": (50, 100, 200),    # Rust / Orange
}


def build_atom_model(concept: str = "atom") -> Model3D:
    """Builds 3D Bohr/quantum atomic model with nucleus and animated electron orbitals."""
    root = ModelNode3D(name="atom_root")

    # 1. Nucleus (cluster of protons and neutrons)
    nucleus_node = ModelNode3D(name="nucleus")
    nucleon_offsets = [
        ([0.0, 0.0, 0.0], (40, 40, 230)),    # Proton (red)
        ([0.15, 0.1, -0.05], (40, 200, 40)), # Neutron (green)
        ([-0.12, 0.12, 0.08], (40, 40, 230)),
        ([0.08, -0.15, 0.1], (40, 200, 40)),
        ([-0.1, -0.1, -0.12], (40, 40, 230)),
        ([0.12, -0.05, -0.15], (40, 200, 40)),
    ]
    for idx, (pos, col) in enumerate(nucleon_offsets):
        sphere = create_uv_sphere(radius=0.15, lat_segs=6, lon_segs=8, color=col, name=f"nucleon_{idx}")
        n_node = ModelNode3D(name=f"nucleon_{idx}", mesh=sphere, position=np.array(pos, dtype=np.float32))
        nucleus_node.add_child(n_node)
    root.add_child(nucleus_node)

    # 2. Orbital Rings & Orbiting Electrons
    # 3 distinct tilted orbital planes
    orbital_angles = [
        (0.0, 0.0, 0.0),
        (math.radians(60), math.radians(30), 0.0),
        (math.radians(-60), math.radians(45), 0.0),
    ]
    electron_nodes: List[ModelNode3D] = []

    for i, rot in enumerate(orbital_angles):
        ring_mesh = create_torus(ring_radius=1.1, tube_radius=0.015, ring_segs=16, tube_segs=4, color=(240, 200, 0), name=f"orbit_ring_{i}")
        ring_node = ModelNode3D(name=f"orbit_{i}", mesh=ring_mesh, rotation=np.array(rot, dtype=np.float32))

        # Electron sphere attached as child
        e_mesh = create_uv_sphere(radius=0.08, lat_segs=6, lon_segs=6, color=(255, 255, 100), name=f"electron_{i}")
        e_node = ModelNode3D(name=f"electron_{i}", mesh=e_mesh, position=np.array([1.1, 0.0, 0.0], dtype=np.float32))
        ring_node.add_child(e_node)
        electron_nodes.append(e_node)

        root.add_child(ring_node)

    # Animation: orbital circulation of electrons
    def _animate_atom(model: Model3D, t: float) -> None:
        speeds = [2.5, 3.2, 2.8]
        for idx, e_node in enumerate(electron_nodes):
            angle = t * speeds[idx] + idx * 1.5
            e_node.position[0] = 1.1 * math.cos(angle)
            e_node.position[2] = 1.1 * math.sin(angle)
        # Nucleus subtle rotation
        nucleus_node.rotation[1] = t * 0.5

    return Model3D(concept=concept, root=root, category="SCIENTIFIC_MODEL", animation_fn=_animate_atom, description="Bohr Atomic Structure with Nucleons and Electron Orbitals")


def build_molecule_model(concept: str = "molecule") -> Model3D:
    """Builds 3D molecular structure with correct geometry (H2O, CO2, CH4, NaCl, or generic)."""
    concept_lower = concept.lower()
    root = ModelNode3D(name="molecule_root")

    atoms_data: List[Tuple[str, np.ndarray, float]] = []  # (symbol, pos, radius)
    bonds_data: List[Tuple[int, int]] = []                # (idx_a, idx_b)

    if any(k in concept_lower for k in ("water", "h2o")):
        # Water: O at center, two H at 104.5 degrees
        angle = math.radians(104.5 * 0.5)
        d = 0.85
        atoms_data = [
            ("O", np.array([0.0, 0.25, 0.0], dtype=np.float32), 0.38),
            ("H", np.array([-d * math.sin(angle), 0.25 - d * math.cos(angle), 0.0], dtype=np.float32), 0.22),
            ("H", np.array([ d * math.sin(angle), 0.25 - d * math.cos(angle), 0.0], dtype=np.float32), 0.22),
        ]
        bonds_data = [(0, 1), (0, 2)]

    elif any(k in concept_lower for k in ("carbon dioxide", "co2")):
        # CO2: Linear O=C=O
        atoms_data = [
            ("C", np.array([0.0, 0.0, 0.0], dtype=np.float32), 0.35),
            ("O", np.array([-0.95, 0.0, 0.0], dtype=np.float32), 0.35),
            ("O", np.array([ 0.95, 0.0, 0.0], dtype=np.float32), 0.35),
        ]
        bonds_data = [(0, 1), (0, 2)]

    elif any(k in concept_lower for k in ("methane", "ch4")):
        # Methane: Tetrahedral 109.5 degrees
        s = 0.65
        atoms_data = [
            ("C", np.array([0.0, 0.0, 0.0], dtype=np.float32), 0.36),
            ("H", np.array([ s,  s,  s], dtype=np.float32), 0.22),
            ("H", np.array([-s, -s,  s], dtype=np.float32), 0.22),
            ("H", np.array([-s,  s, -s], dtype=np.float32), 0.22),
            ("H", np.array([ s, -s, -s], dtype=np.float32), 0.22),
        ]
        bonds_data = [(0, 1), (0, 2), (0, 3), (0, 4)]

    elif any(k in concept_lower for k in ("salt", "nacl", "crystal")):
        # NaCl: Cubic crystal unit
        coords = [-0.5, 0.5]
        idx = 0
        for x in coords:
            for y in coords:
                for z in coords:
                    is_na = ((int(x > 0) + int(y > 0) + int(z > 0)) % 2 == 0)
                    sym = "NA" if is_na else "CL"
                    r = 0.24 if is_na else 0.32
                    atoms_data.append((sym, np.array([x, y, z], dtype=np.float32), r))
                    idx += 1
        # Bonds along cube edges
        bonds_data = [
            (0, 1), (1, 3), (3, 2), (2, 0),
            (4, 5), (5, 7), (7, 6), (6, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]

    else:
        # Default: Ethanol / organic chain (C-C-O-H)
        atoms_data = [
            ("C", np.array([-0.6, 0.0, 0.0], dtype=np.float32), 0.34),
            ("C", np.array([ 0.3, 0.0, 0.0], dtype=np.float32), 0.34),
            ("O", np.array([ 0.9, 0.5, 0.0], dtype=np.float32), 0.32),
            ("H", np.array([ 1.4, 0.2, 0.0], dtype=np.float32), 0.20),
            ("H", np.array([-0.8, -0.6, 0.0], dtype=np.float32), 0.20),
            ("H", np.array([-1.1,  0.4, 0.0], dtype=np.float32), 0.20),
        ]
        bonds_data = [(0, 1), (1, 2), (2, 3), (0, 4), (0, 5)]

    # Build atom spheres
    for i, (sym, pos, r) in enumerate(atoms_data):
        col = CPK_COLORS.get(sym, (180, 180, 180))
        sphere = create_uv_sphere(radius=r, lat_segs=10, lon_segs=12, color=col, name=f"atom_{sym}_{i}")
        node = ModelNode3D(name=f"atom_{i}", mesh=sphere, position=pos)
        root.add_child(node)

    # Build bond cylinders
    for b_idx, (idx_a, idx_b) in enumerate(bonds_data):
        pos_a = atoms_data[idx_a][1]
        pos_b = atoms_data[idx_b][1]
        mid = (pos_a + pos_b) * 0.5
        diff = pos_b - pos_a
        length = float(np.linalg.norm(diff))
        if length > 1e-4:
            cyl = create_cylinder(radius_top=0.07, radius_bottom=0.07, height=length, segs=8, color=(190, 190, 190), name=f"bond_{b_idx}")
            # Rotate cylinder from Y-axis to diff direction
            dir_norm = diff / length
            y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            rot_axis = np.cross(y_axis, dir_norm)
            norm_rot = np.linalg.norm(rot_axis)
            rot_euler = np.zeros(3, dtype=np.float32)
            if norm_rot > 1e-4:
                rot_axis /= norm_rot
                angle = math.acos(max(-1.0, min(1.0, float(np.dot(y_axis, dir_norm)))))
                # Simple Euler approximation
                rot_euler = rot_axis * angle
            bond_node = ModelNode3D(name=f"bond_{b_idx}", mesh=cyl, position=mid, rotation=rot_euler)
            root.add_child(bond_node)

    def _animate_molecule(model: Model3D, t: float) -> None:
        # Thermal vibration & rotation
        root.rotation[1] = t * 0.4
        root.rotation[0] = math.sin(t * 0.8) * 0.15

    return Model3D(concept=concept, root=root, category="SCIENTIFIC_MODEL", animation_fn=_animate_molecule, description="3D Ball-and-Stick Molecular Geometry")


def build_solar_system_model(concept: str = "solar system") -> Model3D:
    """Builds 3D Solar System with Sun, orbiting planets, and planetary rings."""
    root = ModelNode3D(name="solar_root")

    # Central Sun
    sun_mesh = create_uv_sphere(radius=0.48, lat_segs=14, lon_segs=18, color=(30, 210, 255), name="sun")
    sun_node = ModelNode3D(name="sun_node", mesh=sun_mesh)
    root.add_child(sun_node)

    # Planets: (name, radius_from_sun, planet_size, color_bgr, orbital_speed)
    planets = [
        ("Mercury", 0.75, 0.07, (160, 160, 160), 3.2),
        ("Venus",   1.05, 0.11, (100, 210, 240), 2.4),
        ("Earth",   1.40, 0.13, (240, 140,  40), 1.8),
        ("Mars",    1.75, 0.09, ( 40,  80, 220), 1.4),
        ("Jupiter", 2.25, 0.24, (120, 180, 210), 0.9),
        ("Saturn",  2.80, 0.20, (140, 220, 230), 0.7),
    ]

    planet_nodes: List[Tuple[ModelNode3D, float, float]] = []

    for name, dist, sz, col, speed in planets:
        orbit_ring = create_torus(ring_radius=dist, tube_radius=0.008, ring_segs=32, tube_segs=4, color=(80, 80, 80), name=f"{name}_orbit")
        ring_node = ModelNode3D(name=f"{name}_orbit_node", mesh=orbit_ring)
        root.add_child(ring_node)

        p_mesh = create_uv_sphere(radius=sz, lat_segs=8, lon_segs=10, color=col, name=name)
        p_node = ModelNode3D(name=f"{name}_body", mesh=p_mesh, position=np.array([dist, 0.0, 0.0], dtype=np.float32))

        # Saturn 3D rings
        if name == "Saturn":
            s_ring = create_torus(ring_radius=sz * 1.8, tube_radius=0.02, ring_segs=20, tube_segs=4, color=(160, 200, 220), name="saturn_ring")
            s_ring_node = ModelNode3D(name="saturn_ring_node", mesh=s_ring, rotation=np.array([0.4, 0.0, 0.0], dtype=np.float32))
            p_node.add_child(s_ring_node)

        ring_node.add_child(p_node)
        planet_nodes.append((p_node, dist, speed))

    def _animate_solar(model: Model3D, t: float) -> None:
        sun_node.rotation[1] = t * 0.2
        for p_node, dist, speed in planet_nodes:
            angle = t * speed
            p_node.position[0] = dist * math.cos(angle)
            p_node.position[2] = dist * math.sin(angle)

    return Model3D(concept=concept, root=root, category="SCIENTIFIC_MODEL", animation_fn=_animate_solar, description="Heliocentric Solar System with Orbital Kinematics")


def build_wave_model(concept: str = "wave") -> Model3D:
    """Builds 3D undulating sine wave mesh with continuous traveling wave oscillation."""
    root = ModelNode3D(name="wave_root")

    # Construct 3D grid surface: (X x Z)
    nx, nz = 24, 8
    lx, lz = 2.4, 0.8
    xs = np.linspace(-lx * 0.5, lx * 0.5, nx, dtype=np.float32)
    zs = np.linspace(-lz * 0.5, lz * 0.5, nz, dtype=np.float32)

    verts = []
    for z in zs:
        for x in xs:
            y = 0.35 * math.sin(x * 3.5)
            verts.append([x, y, z])
    v_arr = np.array(verts, dtype=np.float32)

    faces = []
    for r in range(nz - 1):
        for c in range(nx - 1):
            i0 = r * nx + c
            i1 = r * nx + c + 1
            i2 = (r + 1) * nx + c
            i3 = (r + 1) * nx + c + 1
            faces.append([i0, i1, i3])
            faces.append([i0, i3, i2])
    f_arr = np.array(faces, dtype=np.int32)

    mat = Material3D(diffuse_color=(255, 200, 40), double_sided=True)
    wave_mesh = Mesh3D(vertices=v_arr, faces=f_arr, material=mat, name="wave_surface")
    wave_mesh.compute_face_normals()

    wave_node = ModelNode3D(name="wave_mesh_node", mesh=wave_mesh)
    root.add_child(wave_node)

    # Reference axis
    axis_mesh = create_cylinder(radius_top=0.015, radius_bottom=0.015, height=2.6, segs=6, color=(140, 140, 140), name="wave_axis")
    axis_node = ModelNode3D(name="axis_node", mesh=axis_mesh, rotation=np.array([0.0, 0.0, math.radians(90)], dtype=np.float32))
    root.add_child(axis_node)

    def _animate_wave(model: Model3D, t: float) -> None:
        idx = 0
        for z in zs:
            for x in xs:
                # Traveling wave: y = A * sin(k*x - w*t)
                wave_mesh.vertices[idx, 1] = 0.35 * math.sin(x * 3.8 - t * 4.0)
                idx += 1
        wave_mesh.compute_face_normals()

    return Model3D(concept=concept, root=root, category="SCIENTIFIC_MODEL", animation_fn=_animate_wave, description="3D Harmonic Traveling Wave Function")


def build_heart_model(concept: str = "heart") -> Model3D:
    """Builds anatomical 3D heart with ventricles, atria, aorta, and rhythmic contraction."""
    root = ModelNode3D(name="heart_root")

    # Ventricle mass (conical rounded lower heart)
    vent_mesh = create_uv_sphere(radius=0.55, lat_segs=12, lon_segs=14, color=(40, 40, 190), name="ventricles")
    # Taper vertices towards bottom
    vent_mesh.vertices[:, 0] *= 0.85
    y_mask = vent_mesh.vertices[:, 1] < 0
    vent_mesh.vertices[y_mask, 0] *= (1.0 + vent_mesh.vertices[y_mask, 1] * 0.4)
    vent_mesh.vertices[y_mask, 2] *= (1.0 + vent_mesh.vertices[y_mask, 1] * 0.4)
    vent_mesh.compute_face_normals()
    vent_node = ModelNode3D(name="ventricle_node", mesh=vent_mesh)
    root.add_child(vent_node)

    # Left and Right Atria (upper lobes)
    atrium_l = create_uv_sphere(radius=0.32, lat_segs=8, lon_segs=10, color=(50, 50, 170), name="atrium_l")
    node_al = ModelNode3D(name="atrium_l_node", mesh=atrium_l, position=np.array([-0.3, 0.45, -0.05], dtype=np.float32))
    root.add_child(node_al)

    atrium_r = create_uv_sphere(radius=0.30, lat_segs=8, lon_segs=10, color=(60, 60, 160), name="atrium_r")
    node_ar = ModelNode3D(name="atrium_r_node", mesh=atrium_r, position=np.array([0.28, 0.45, -0.05], dtype=np.float32))
    root.add_child(node_ar)

    # Aorta arch (curved cylinder top)
    aorta_mesh = create_torus(ring_radius=0.25, tube_radius=0.09, ring_segs=16, tube_segs=6, color=(30, 30, 220), name="aorta")
    aorta_node = ModelNode3D(name="aorta_node", mesh=aorta_mesh, position=np.array([0.0, 0.72, 0.0], dtype=np.float32), rotation=np.array([0.0, 0.0, math.radians(45)], dtype=np.float32))
    root.add_child(aorta_node)

    # Rhythmic systolic/diastolic pulsation animation
    def _animate_heart(model: Model3D, t: float) -> None:
        # Cardiac pulse: two rapid beats then rest (lub-dub)
        pulse = (math.sin(t * 5.0) ** 6) * 0.14
        scl = 1.0 + pulse
        vent_node.scale[:] = [scl, scl * 1.05, scl]
        node_al.scale[:] = [1.0 - pulse * 0.5, 1.0 - pulse * 0.5, 1.0 - pulse * 0.5]
        node_ar.scale[:] = [1.0 - pulse * 0.5, 1.0 - pulse * 0.5, 1.0 - pulse * 0.5]

    return Model3D(concept=concept, root=root, category="SCIENTIFIC_MODEL", animation_fn=_animate_heart, description="Anatomical 3D Heart with Cardiac Pulsation")


def build_mechanical_assembly_model(concept: str = "mechanical assembly") -> Model3D:
    """Builds mechanical assembly with interlocking rotating spur gears and piston linkage."""
    root = ModelNode3D(name="mech_root")

    # Gear 1 (large drive gear)
    gear1_mesh = create_gear(radius=0.65, thickness=0.15, num_teeth=10, color=(160, 160, 160), name="drive_gear")
    gear1_node = ModelNode3D(name="gear1", mesh=gear1_mesh, position=np.array([-0.55, -0.2, 0.0], dtype=np.float32), rotation=np.array([math.radians(90), 0.0, 0.0], dtype=np.float32))
    root.add_child(gear1_node)

    # Gear 2 (smaller pinion gear intermeshed)
    gear2_mesh = create_gear(radius=0.45, thickness=0.15, num_teeth=7, color=(200, 180, 50), name="pinion_gear")
    gear2_node = ModelNode3D(name="gear2", mesh=gear2_mesh, position=np.array([0.48, -0.2, 0.0], dtype=np.float32), rotation=np.array([math.radians(90), 0.0, 0.0], dtype=np.float32))
    root.add_child(gear2_node)

    # Piston cylinder housing
    cyl_mesh = create_cylinder(radius_top=0.25, radius_bottom=0.25, height=0.7, segs=10, color=(100, 100, 100), name="cylinder_block")
    cyl_node = ModelNode3D(name="piston_housing", mesh=cyl_mesh, position=np.array([-0.55, 0.75, 0.0], dtype=np.float32))
    root.add_child(cyl_node)

    # Piston head
    piston_mesh = create_cylinder(radius_top=0.22, radius_bottom=0.22, height=0.3, segs=10, color=(210, 210, 210), name="piston_head")
    piston_node = ModelNode3D(name="piston_head", mesh=piston_mesh, position=np.array([-0.55, 0.65, 0.0], dtype=np.float32))
    root.add_child(piston_node)

    def _animate_mech(model: Model3D, t: float) -> None:
        # Gear 1 rotates at w
        gear1_node.rotation[1] = t * 2.0
        # Gear 2 rotates opposite with gear ratio
        gear2_node.rotation[1] = -t * 2.0 * (10.0 / 7.0) + math.radians(180 / 7)
        # Piston reciprocating stroke
        stroke = 0.22 * math.cos(t * 2.0)
        piston_node.position[1] = 0.65 + stroke

    return Model3D(concept=concept, root=root, category="TECHNICAL_ASSEMBLY", animation_fn=_animate_mech, description="Kinematic Spur Gear Train with Reciprocating Piston")


def build_circuit_model(concept: str = "circuit") -> Model3D:
    """Builds 3D electrical circuit loop with battery, resistor, LED, and flowing electrons."""
    root = ModelNode3D(name="circuit_root")

    # Rectangular wire loop (4 cylindrical segments)
    w_color = (200, 120, 40)  # Copper
    wire_t = create_cylinder(radius_top=0.03, radius_bottom=0.03, height=1.6, segs=6, color=w_color, name="wire_top")
    node_wt = ModelNode3D(name="wire_top_node", mesh=wire_t, position=np.array([0.0, 0.7, 0.0], dtype=np.float32), rotation=np.array([0.0, 0.0, math.radians(90)], dtype=np.float32))
    root.add_child(node_wt)

    wire_b = create_cylinder(radius_top=0.03, radius_bottom=0.03, height=1.6, segs=6, color=w_color, name="wire_bot")
    node_wb = ModelNode3D(name="wire_bot_node", mesh=wire_b, position=np.array([0.0, -0.7, 0.0], dtype=np.float32), rotation=np.array([0.0, 0.0, math.radians(90)], dtype=np.float32))
    root.add_child(node_wb)

    wire_l = create_cylinder(radius_top=0.03, radius_bottom=0.03, height=1.4, segs=6, color=w_color, name="wire_left")
    node_wl = ModelNode3D(name="wire_left_node", mesh=wire_l, position=np.array([-0.8, 0.0, 0.0], dtype=np.float32))
    root.add_child(node_wl)

    wire_r = create_cylinder(radius_top=0.03, radius_bottom=0.03, height=1.4, segs=6, color=w_color, name="wire_right")
    node_wr = ModelNode3D(name="wire_right_node", mesh=wire_r, position=np.array([0.8, 0.0, 0.0], dtype=np.float32))
    root.add_child(node_wr)

    # Battery (on left wire)
    batt_mesh = create_cylinder(radius_top=0.14, radius_bottom=0.14, height=0.45, segs=8, color=(40, 40, 40), name="battery_body")
    batt_node = ModelNode3D(name="battery", mesh=batt_mesh, position=np.array([-0.8, 0.0, 0.0], dtype=np.float32))
    batt_cap = create_cylinder(radius_top=0.06, radius_bottom=0.06, height=0.1, segs=8, color=(40, 200, 240), name="battery_pos")
    batt_cap_node = ModelNode3D(name="batt_cap", mesh=batt_cap, position=np.array([0.0, 0.25, 0.0], dtype=np.float32))
    batt_node.add_child(batt_cap_node)
    root.add_child(batt_node)

    # Resistor (on top wire)
    res_mesh = create_box(width=0.35, height=0.14, depth=0.14, color=(140, 180, 220), name="resistor")
    res_node = ModelNode3D(name="resistor_node", mesh=res_mesh, position=np.array([0.0, 0.7, 0.0], dtype=np.float32))
    root.add_child(res_node)

    # LED Indicator (on right wire)
    led_mesh = create_uv_sphere(radius=0.16, lat_segs=8, lon_segs=10, color=(40, 240, 40), name="led")
    led_node = ModelNode3D(name="led_node", mesh=led_mesh, position=np.array([0.8, 0.0, 0.0], dtype=np.float32))
    root.add_child(led_node)

    # Circulating electrons
    e_spheres: List[ModelNode3D] = []
    for k in range(4):
        e_mesh = create_uv_sphere(radius=0.05, lat_segs=6, lon_segs=6, color=(255, 255, 100), name=f"circ_e_{k}")
        e_node = ModelNode3D(name=f"circ_e_{k}", mesh=e_mesh)
        root.add_child(e_node)
        e_spheres.append(e_node)

    # Perimeter = 2 * (1.6 + 1.4) = 6.0
    def _animate_circuit(model: Model3D, t: float) -> None:
        perim = 6.0
        speed = 1.8
        for k, e_node in enumerate(e_spheres):
            d = (t * speed + k * (perim / 4.0)) % perim
            # Map d along clockwise rectangle: (-0.8, -0.7) -> (-0.8, 0.7) -> (0.8, 0.7) -> (0.8, -0.7) -> (-0.8, -0.7)
            if d < 1.4:  # Up left wire
                x = -0.8
                y = -0.7 + d
            elif d < 3.0: # Across top wire
                x = -0.8 + (d - 1.4)
                y = 0.7
            elif d < 4.4: # Down right wire
                x = 0.8
                y = 0.7 - (d - 3.0)
            else:         # Back across bottom wire
                x = 0.8 - (d - 4.4)
                y = -0.7
            e_node.position[:] = [x, y, 0.0]

    return Model3D(concept=concept, root=root, category="TECHNICAL_ASSEMBLY", animation_fn=_animate_circuit, description="Active 3D Closed Circuit with Electron Drift Velocity")
