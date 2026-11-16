"""
integration/generation/geometry_primitives.py — Fast Procedural 3D Mesh Primitives.

Generates:
- UV Spheres (atoms, celestial bodies, organic features)
- Cylinders & Cones (molecular bonds, stems, structural limbs)
- Boxes & Cuboids (bases, seats, chassis, components)
- Tori & Rings (electron orbits, planetary rings, handles)
- Gears & Cogs (mechanical assemblies, linkages)
- 3D Ribbon Paths (wave functions, electrical traces)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple
import numpy as np

from integration.generation.model3d import Material3D, Mesh3D


def create_uv_sphere(
    radius: float = 1.0,
    lat_segs: int = 12,
    lon_segs: int = 16,
    color: Tuple[int, int, int] = (200, 200, 200),
    name: str = "sphere",
) -> Mesh3D:
    """Generates a UV sphere mesh."""
    verts = []
    # Top pole
    verts.append([0.0, radius, 0.0])

    for i in range(1, lat_segs):
        theta = math.pi * i / lat_segs
        sin_t = math.sin(theta)
        cos_t = math.cos(theta)
        for j in range(lon_segs):
            phi = 2.0 * math.pi * j / lon_segs
            x = radius * sin_t * math.cos(phi)
            y = radius * cos_t
            z = radius * sin_t * math.sin(phi)
            verts.append([x, y, z])

    # Bottom pole
    verts.append([0.0, -radius, 0.0])
    v_arr = np.array(verts, dtype=np.float32)

    faces = []
    # Top pole fan
    for j in range(lon_segs):
        n1 = 1 + j
        n2 = 1 + (j + 1) % lon_segs
        faces.append([0, n1, n2])

    # Middle bands
    for i in range(lat_segs - 2):
        row1 = 1 + i * lon_segs
        row2 = 1 + (i + 1) * lon_segs
        for j in range(lon_segs):
            j_next = (j + 1) % lon_segs
            a = row1 + j
            b = row2 + j
            c = row2 + j_next
            d = row1 + j_next
            faces.append([a, b, c])
            faces.append([a, c, d])

    # Bottom pole fan
    bot_idx = len(verts) - 1
    last_row = 1 + (lat_segs - 2) * lon_segs
    for j in range(lon_segs):
        n1 = last_row + j
        n2 = last_row + (j + 1) % lon_segs
        faces.append([bot_idx, n2, n1])

    f_arr = np.array(faces, dtype=np.int32)
    mat = Material3D(diffuse_color=color)
    mesh = Mesh3D(vertices=v_arr, faces=f_arr, material=mat, name=name)
    mesh.compute_face_normals()
    return mesh


def create_cylinder(
    radius_top: float = 0.5,
    radius_bottom: float = 0.5,
    height: float = 1.0,
    segs: int = 12,
    color: Tuple[int, int, int] = (180, 180, 180),
    name: str = "cylinder",
) -> Mesh3D:
    """Generates a cylinder/cone along Y axis centered at Y=0."""
    verts = []
    half_h = height * 0.5

    # Top ring
    for j in range(segs):
        phi = 2.0 * math.pi * j / segs
        verts.append([radius_top * math.cos(phi), half_h, radius_top * math.sin(phi)])

    # Bottom ring
    for j in range(segs):
        phi = 2.0 * math.pi * j / segs
        verts.append([radius_bottom * math.cos(phi), -half_h, radius_bottom * math.sin(phi)])

    # Top center, bottom center
    top_c = len(verts)
    verts.append([0.0, half_h, 0.0])
    bot_c = len(verts)
    verts.append([0.0, -half_h, 0.0])

    v_arr = np.array(verts, dtype=np.float32)
    faces = []

    for j in range(segs):
        j_next = (j + 1) % segs
        t1 = j
        t2 = j_next
        b1 = segs + j
        b2 = segs + j_next

        # Side quads (2 triangles)
        faces.append([t1, b1, b2])
        faces.append([t1, b2, t2])

        # Caps
        if radius_top > 0:
            faces.append([top_c, t1, t2])
        if radius_bottom > 0:
            faces.append([bot_c, b2, b1])

    f_arr = np.array(faces, dtype=np.int32)
    mat = Material3D(diffuse_color=color)
    mesh = Mesh3D(vertices=v_arr, faces=f_arr, material=mat, name=name)
    mesh.compute_face_normals()
    return mesh


def create_box(
    width: float = 1.0,
    height: float = 1.0,
    depth: float = 1.0,
    color: Tuple[int, int, int] = (180, 180, 180),
    name: str = "box",
) -> Mesh3D:
    """Generates an axis-aligned 3D box centered at origin."""
    hw, hh, hd = width * 0.5, height * 0.5, depth * 0.5
    verts = np.array([
        [-hw, -hh,  hd], [ hw, -hh,  hd], [ hw,  hh,  hd], [-hw,  hh,  hd],  # Front: 0,1,2,3
        [-hw, -hh, -hd], [-hw,  hh, -hd], [ hw,  hh, -hd], [ hw, -hh, -hd],  # Back:  4,5,6,7
    ], dtype=np.float32)

    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # Front
        [7, 4, 5], [7, 5, 6],  # Back
        [4, 0, 3], [4, 3, 5],  # Left
        [1, 7, 6], [1, 6, 2],  # Right
        [3, 2, 6], [3, 6, 5],  # Top
        [4, 7, 1], [4, 1, 0],  # Bottom
    ], dtype=np.int32)

    mat = Material3D(diffuse_color=color)
    mesh = Mesh3D(vertices=verts, faces=faces, material=mat, name=name)
    mesh.compute_face_normals()
    return mesh


def create_torus(
    ring_radius: float = 1.0,
    tube_radius: float = 0.1,
    ring_segs: int = 24,
    tube_segs: int = 8,
    color: Tuple[int, int, int] = (200, 200, 0),
    name: str = "torus",
) -> Mesh3D:
    """Generates a 3D torus in X-Z plane."""
    verts = []
    for i in range(ring_segs):
        u = 2.0 * math.pi * i / ring_segs
        cu, su = math.cos(u), math.sin(u)
        for j in range(tube_segs):
            v = 2.0 * math.pi * j / tube_segs
            cv, sv = math.cos(v), math.sin(v)
            x = (ring_radius + tube_radius * cv) * cu
            y = tube_radius * sv
            z = (ring_radius + tube_radius * cv) * su
            verts.append([x, y, z])

    v_arr = np.array(verts, dtype=np.float32)
    faces = []
    for i in range(ring_segs):
        i_next = (i + 1) % ring_segs
        for j in range(tube_segs):
            j_next = (j + 1) % tube_segs
            a = i * tube_segs + j
            b = i_next * tube_segs + j
            c = i_next * tube_segs + j_next
            d = i * tube_segs + j_next
            faces.append([a, b, c])
            faces.append([a, c, d])

    f_arr = np.array(faces, dtype=np.int32)
    mat = Material3D(diffuse_color=color)
    mesh = Mesh3D(vertices=v_arr, faces=f_arr, material=mat, name=name)
    mesh.compute_face_normals()
    return mesh


def create_gear(
    radius: float = 1.0,
    thickness: float = 0.2,
    num_teeth: int = 8,
    color: Tuple[int, int, int] = (160, 160, 160),
    name: str = "gear",
) -> Mesh3D:
    """Generates a 3D spur gear with teeth."""
    verts = []
    half_t = thickness * 0.5
    total_pts = num_teeth * 4

    for i in range(total_pts):
        angle = 2.0 * math.pi * i / total_pts
        r = radius * 1.2 if (i % 4 in (1, 2)) else radius * 0.85
        x = r * math.cos(angle)
        z = r * math.sin(angle)
        verts.append([x, half_t, z])   # top
        verts.append([x, -half_t, z])  # bottom

    # Center top and bottom
    top_c = len(verts)
    verts.append([0.0, half_t, 0.0])
    bot_c = len(verts)
    verts.append([0.0, -half_t, 0.0])

    v_arr = np.array(verts, dtype=np.float32)
    faces = []

    for i in range(total_pts):
        i_next = (i + 1) % total_pts
        t1 = i * 2
        b1 = t1 + 1
        t2 = i_next * 2
        b2 = t2 + 1

        # Tooth perimeter sides
        faces.append([t1, b1, b2])
        faces.append([t1, b2, t2])

        # Top cap
        faces.append([top_c, t1, t2])
        # Bottom cap
        faces.append([bot_c, b2, b1])

    f_arr = np.array(faces, dtype=np.int32)
    mat = Material3D(diffuse_color=color)
    mesh = Mesh3D(vertices=v_arr, faces=f_arr, material=mat, name=name)
    mesh.compute_face_normals()
    return mesh

