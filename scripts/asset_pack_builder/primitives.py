"""
scripts/asset_pack_builder/primitives.py — Enhanced 3D Mesh Primitives for Asset Library.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple
import numpy as np

from integration.generation.model3d import Material3D, Mesh3D, ModelNode3D


def make_sphere(
    radius: float = 1.0,
    lat_segs: int = 16,
    lon_segs: int = 24,
    color: Tuple[int, int, int] = (200, 200, 200),
    roughness: float = 0.6,
    metallic: float = 0.0,
    opacity: float = 1.0,
    name: str = "sphere",
) -> Mesh3D:
    verts = [[0.0, radius, 0.0]]
    for i in range(1, lat_segs):
        theta = math.pi * i / lat_segs
        sin_t, cos_t = math.sin(theta), math.cos(theta)
        for j in range(lon_segs):
            phi = 2.0 * math.pi * j / lon_segs
            verts.append([radius * sin_t * math.cos(phi), radius * cos_t, radius * sin_t * math.sin(phi)])
    verts.append([0.0, -radius, 0.0])
    v_arr = np.array(verts, dtype=np.float32)

    faces = []
    # Top fan
    for j in range(lon_segs):
        faces.append([0, 1 + j, 1 + (j + 1) % lon_segs])
    # Body quads -> triangles
    for i in range(lat_segs - 2):
        r1 = 1 + i * lon_segs
        r2 = 1 + (i + 1) * lon_segs
        for j in range(lon_segs):
            jn = (j + 1) % lon_segs
            faces.append([r1 + j, r2 + j, r2 + jn])
            faces.append([r1 + j, r2 + jn, r1 + jn])
    # Bottom fan
    bot = len(verts) - 1
    last_r = 1 + (lat_segs - 2) * lon_segs
    for j in range(lon_segs):
        faces.append([bot, last_r + (j + 1) % lon_segs, last_r + j])

    mat = Material3D(diffuse_color=color, roughness=roughness, metallic=metallic, opacity=opacity)
    mesh = Mesh3D(vertices=v_arr, faces=np.array(faces, dtype=np.int32), material=mat, name=name)
    mesh.compute_face_normals()
    return mesh


def make_ellipsoid(
    rx: float = 1.0,
    ry: float = 1.0,
    rz: float = 1.0,
    lat_segs: int = 16,
    lon_segs: int = 24,
    color: Tuple[int, int, int] = (200, 200, 200),
    roughness: float = 0.6,
    metallic: float = 0.0,
    opacity: float = 1.0,
    name: str = "ellipsoid",
) -> Mesh3D:
    s = make_sphere(radius=1.0, lat_segs=lat_segs, lon_segs=lon_segs, color=color,
                    roughness=roughness, metallic=metallic, opacity=opacity, name=name)
    s.vertices[:, 0] *= rx
    s.vertices[:, 1] *= ry
    s.vertices[:, 2] *= rz
    s.compute_face_normals()
    return s


def make_cylinder(
    radius_top: float = 0.5,
    radius_bottom: float = 0.5,
    height: float = 1.0,
    segs: int = 18,
    color: Tuple[int, int, int] = (180, 180, 180),
    roughness: float = 0.5,
    metallic: float = 0.1,
    opacity: float = 1.0,
    name: str = "cylinder",
) -> Mesh3D:
    verts = []
    half_h = height * 0.5
    for j in range(segs):
        phi = 2.0 * math.pi * j / segs
        verts.append([radius_top * math.cos(phi), half_h, radius_top * math.sin(phi)])
    for j in range(segs):
        phi = 2.0 * math.pi * j / segs
        verts.append([radius_bottom * math.cos(phi), -half_h, radius_bottom * math.sin(phi)])
    # Cap centers
    top_c = len(verts)
    verts.append([0.0, half_h, 0.0])
    bot_c = len(verts)
    verts.append([0.0, -half_h, 0.0])

    faces = []
    # Side quads
    for j in range(segs):
        jn = (j + 1) % segs
        t1, t2 = j, jn
        b1, b2 = segs + j, segs + jn
        faces.append([t1, b1, b2])
        faces.append([t1, b2, t2])
    # Caps
    for j in range(segs):
        jn = (j + 1) % segs
        faces.append([top_c, j, jn])
        faces.append([bot_c, segs + jn, segs + j])

    mat = Material3D(diffuse_color=color, roughness=roughness, metallic=metallic, opacity=opacity)
    mesh = Mesh3D(vertices=np.array(verts, dtype=np.float32), faces=np.array(faces, dtype=np.int32), material=mat, name=name)
    mesh.compute_face_normals()
    return mesh


def make_box(
    dx: float = 1.0,
    dy: float = 1.0,
    dz: float = 1.0,
    color: Tuple[int, int, int] = (180, 180, 180),
    roughness: float = 0.7,
    metallic: float = 0.0,
    opacity: float = 1.0,
    name: str = "box",
) -> Mesh3D:
    hx, hy, hz = dx * 0.5, dy * 0.5, dz * 0.5
    verts = [
        [-hx, -hy,  hz], [ hx, -hy,  hz], [ hx,  hy,  hz], [-hx,  hy,  hz],
        [-hx, -hy, -hz], [-hx,  hy, -hz], [ hx,  hy, -hz], [ hx, -hy, -hz],
    ]
    faces = [
        [0, 1, 2], [0, 2, 3],  # Front
        [7, 4, 5], [7, 5, 6],  # Back
        [3, 2, 6], [3, 6, 5],  # Top
        [4, 1, 0], [4, 7, 1],  # Bottom
        [1, 7, 6], [1, 6, 2],  # Right
        [4, 0, 3], [4, 3, 5],  # Left
    ]
    mat = Material3D(diffuse_color=color, roughness=roughness, metallic=metallic, opacity=opacity)
    mesh = Mesh3D(vertices=np.array(verts, dtype=np.float32), faces=np.array(faces, dtype=np.int32), material=mat, name=name)
    mesh.compute_face_normals()
    return mesh


def make_torus(
    r_major: float = 1.0,
    r_minor: float = 0.25,
    segs_major: int = 24,
    segs_minor: int = 12,
    color: Tuple[int, int, int] = (200, 200, 200),
    roughness: float = 0.5,
    metallic: float = 0.0,
    name: str = "torus",
) -> Mesh3D:
    verts = []
    for i in range(segs_major):
        u = 2.0 * math.pi * i / segs_major
        cu, su = math.cos(u), math.sin(u)
        for j in range(segs_minor):
            v = 2.0 * math.pi * j / segs_minor
            cv, sv = math.cos(v), math.sin(v)
            x = (r_major + r_minor * cv) * cu
            y = r_minor * sv
            z = (r_major + r_minor * cv) * su
            verts.append([x, y, z])

    faces = []
    for i in range(segs_major):
        i_next = (i + 1) % segs_major
        for j in range(segs_minor):
            j_next = (j + 1) % segs_minor
            a = i * segs_minor + j
            b = i_next * segs_minor + j
            c = i_next * segs_minor + j_next
            d = i * segs_minor + j_next
            faces.append([a, b, c])
            faces.append([a, c, d])

    mat = Material3D(diffuse_color=color, roughness=roughness, metallic=metallic)
    mesh = Mesh3D(vertices=np.array(verts, dtype=np.float32), faces=np.array(faces, dtype=np.int32), material=mat, name=name)
    mesh.compute_face_normals()
    return mesh


def make_ring_disc(
    r_inner: float = 1.3,
    r_outer: float = 2.2,
    segs: int = 36,
    color: Tuple[int, int, int] = (190, 180, 140),
    opacity: float = 0.9,
    name: str = "ring_disc",
) -> Mesh3D:
    """Flat concentric ring disc in the XZ plane (for planetary rings)."""
    verts = []
    for j in range(segs):
        phi = 2.0 * math.pi * j / segs
        c, s = math.cos(phi), math.sin(phi)
        verts.append([r_inner * c, 0.0, r_inner * s])
        verts.append([r_outer * c, 0.0, r_outer * s])

    faces = []
    for j in range(segs):
        jn = (j + 1) % segs
        in1, out1 = 2 * j, 2 * j + 1
        in2, out2 = 2 * jn, 2 * jn + 1
        # Top face
        faces.append([in1, out1, out2])
        faces.append([in1, out2, in2])
        # Bottom face
        faces.append([in1, out2, out1])
        faces.append([in1, in2, out2])

    mat = Material3D(diffuse_color=color, roughness=0.6, opacity=opacity, double_sided=True)
    mesh = Mesh3D(vertices=np.array(verts, dtype=np.float32), faces=np.array(faces, dtype=np.int32), material=mat, name=name)
    mesh.compute_face_normals()
    return mesh


def make_cone(
    radius: float = 0.5,
    height: float = 1.0,
    segs: int = 18,
    color: Tuple[int, int, int] = (180, 180, 180),
    roughness: float = 0.5,
    metallic: float = 0.1,
    opacity: float = 1.0,
    name: str = "cone",
) -> Mesh3D:
    return make_cylinder(
        radius_top=0.001,
        radius_bottom=radius,
        height=height,
        segs=segs,
        color=color,
        roughness=roughness,
        metallic=metallic,
        opacity=opacity,
        name=name,
    )


def make_helix(
    radius: float = 0.4,
    pitch: float = 0.4,
    turns: float = 3.0,
    tube_radius: float = 0.06,
    segs_total: int = 60,
    color: Tuple[int, int, int] = (80, 140, 220),
    roughness: float = 0.4,
    metallic: float = 0.1,
    opacity: float = 1.0,
    name: str = "helix",
) -> Mesh3D:
    """Continuous 3D tube wound into a helical coil."""
    verts = []
    faces = []
    cross_segs = 8

    # Generate backbone points
    total_angle = 2.0 * math.pi * turns
    angles = np.linspace(0, total_angle, segs_total, endpoint=True)
    ys = np.linspace(-turns * pitch * 0.5, turns * pitch * 0.5, segs_total)

    ring_verts = []
    for k, (ang, y) in enumerate(zip(angles, ys)):
        cx, cz = radius * math.cos(ang), radius * math.sin(ang)
        center = np.array([cx, y, cz], dtype=np.float32)
        # Normal and binormal vectors for circular cross section
        tangent = np.array([-radius * math.sin(ang), pitch / (2.0 * math.pi), radius * math.cos(ang)], dtype=np.float32)
        tangent /= np.linalg.norm(tangent)
        radial = np.array([math.cos(ang), 0.0, math.sin(ang)], dtype=np.float32)
        binormal = np.cross(tangent, radial)
        binormal /= np.linalg.norm(binormal)

        ring = []
        for c_idx in range(cross_segs):
            c_ang = 2.0 * math.pi * c_idx / cross_segs
            offset = (radial * math.cos(c_ang) + binormal * math.sin(c_ang)) * tube_radius
            pt = center + offset
            ring.append(len(verts))
            verts.append(pt.tolist())
        ring_verts.append(ring)

    for k in range(segs_total - 1):
        r1 = ring_verts[k]
        r2 = ring_verts[k + 1]
        for c in range(cross_segs):
            cn = (c + 1) % cross_segs
            faces.append([r1[c], r2[c], r2[cn]])
            faces.append([r1[c], r2[cn], r1[cn]])

    mat = Material3D(diffuse_color=color, roughness=roughness, metallic=metallic, opacity=opacity)
    mesh = Mesh3D(vertices=np.array(verts, dtype=np.float32), faces=np.array(faces, dtype=np.int32), material=mat, name=name)
    mesh.compute_face_normals()
    return mesh


def make_platonic_solid(
    solid_type: str = "tetrahedron",
    radius: float = 1.0,
    color: Tuple[int, int, int] = (150, 180, 220),
    roughness: float = 0.4,
    metallic: float = 0.1,
    opacity: float = 1.0,
    name: str = "platonic",
) -> Mesh3D:
    kind = solid_type.lower()
    if "tetra" in kind:
        # Regular tetrahedron
        a = 1.0 / math.sqrt(3.0)
        v = np.array([[a, a, a], [-a, -a, a], [-a, a, -a], [a, -a, -a]], dtype=np.float32) * radius
        f = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32)
    elif "octa" in kind:
        # Regular octahedron
        v = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=np.float32) * radius
        f = np.array([
            [2, 0, 4], [2, 4, 1], [2, 1, 5], [2, 5, 0],
            [3, 4, 0], [3, 1, 4], [3, 5, 1], [3, 0, 5],
        ], dtype=np.int32)
    elif "icosa" in kind:
        # Regular icosahedron (golden ratio phi)
        phi = (1.0 + math.sqrt(5.0)) * 0.5
        v = np.array([
            [-1,  phi, 0], [ 1,  phi, 0], [-1, -phi, 0], [ 1, -phi, 0],
            [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
            [ phi, 0, -1], [ phi, 0,  1], [-phi, 0, -1], [-phi, 0,  1],
        ], dtype=np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        v *= radius
        f = np.array([
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ], dtype=np.int32)
    else:
        # Dodecahedron (subdivided/dual of icosahedron or approximated)
        return make_sphere(radius=radius, lat_segs=8, lon_segs=10, color=color, roughness=roughness, metallic=metallic, opacity=opacity, name=name)

    mat = Material3D(diffuse_color=color, roughness=roughness, metallic=metallic, opacity=opacity)
    mesh = Mesh3D(vertices=v, faces=f, material=mat, name=name)
    mesh.compute_face_normals()
    return mesh
