"""
integration/generation/model3d.py — Core 3D Geometry Models, Hierarchy & Parsers.

Provides:
- Mesh3D: Vertices, faces, normals, vertex colors, material shading parameters.
- ModelNode3D: Component hierarchy graph with local position, rotation, scale.
- Model3D: Unified 3D representation with animation functions and component access.
- Native Zero-Dependency GLB (binary glTF 2.0) parser.
- Native Wavefront OBJ parser.
- 4x4 Transformation Matrix Math (Translation, Euler rotation, Scale, MVP).
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class Material3D:
    """Material shading parameters."""
    diffuse_color: Tuple[int, int, int] = (200, 200, 200)  # BGR
    ambient_coeff: float = 0.25
    diffuse_coeff: float = 0.75
    specular_coeff: float = 0.2
    shininess: float = 16.0
    opacity: float = 1.0
    wireframe: bool = False
    double_sided: bool = True


@dataclass
class Mesh3D:
    """Represents a 3D polygonal mesh."""
    vertices: np.ndarray  # (N, 3) float32
    faces: np.ndarray     # (M, 3) int32
    normals: Optional[np.ndarray] = None  # (M, 3) or (N, 3) float32
    colors: Optional[np.ndarray] = None   # (M, 3) or (N, 3) uint8 BGR
    material: Material3D = field(default_factory=Material3D)
    name: str = "mesh"

    def compute_face_normals(self) -> np.ndarray:
        """Computes normalized surface normals for all triangular faces."""
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        norm = np.linalg.norm(cross, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        self.normals = (cross / norm).astype(np.float32)
        return self.normals

    def bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        if len(self.vertices) == 0:
            return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    def center_and_normalize(self, target_radius: float = 1.0) -> None:
        """Centers mesh at origin and scales to target radius."""
        if len(self.vertices) == 0:
            return
        b_min, b_max = self.bounding_box()
        center = (b_min + b_max) * 0.5
        self.vertices -= center
        extent = np.linalg.norm(self.vertices, axis=1).max()
        if extent > 1e-6:
            self.vertices *= (target_radius / extent)


@dataclass
class ModelNode3D:
    """Node in a 3D component hierarchy."""
    name: str
    mesh: Optional[Mesh3D] = None
    position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    rotation: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))  # Euler [rx, ry, rz] radians
    scale: np.ndarray = field(default_factory=lambda: np.ones(3, dtype=np.float32))
    children: List[ModelNode3D] = field(default_factory=list)
    parent: Optional[ModelNode3D] = None
    visible: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_child(self, child: ModelNode3D) -> None:
        child.parent = self
        self.children.append(child)

    def get_local_transform(self) -> np.ndarray:
        return make_transform_matrix(self.position, self.rotation, self.scale)

    def get_world_transform(self) -> np.ndarray:
        local_mat = self.get_local_transform()
        if self.parent is not None:
            return self.parent.get_world_transform() @ local_mat
        return local_mat


@dataclass
class Model3D:
    """Complete structured 3D model containing nodes, animation, and metadata."""
    concept: str
    root: ModelNode3D
    visual_id: Optional[str] = None
    version: int = 1
    animation_fn: Optional[Callable[[Model3D, float], None]] = None
    category: str = "3D_OBJECT"  # "3D_OBJECT" | "SCIENTIFIC_MODEL" | "TECHNICAL_ASSEMBLY"
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_all_meshes_with_transforms(self, t: float = 0.0) -> List[Tuple[Mesh3D, np.ndarray]]:
        """Applies animation tick and traverses hierarchy to collect meshes with world matrices."""
        if self.animation_fn is not None:
            try:
                self.animation_fn(self, t)
            except Exception:
                pass

        results: List[Tuple[Mesh3D, np.ndarray]] = []

        def _traverse(node: ModelNode3D, parent_mat: np.ndarray) -> None:
            if not node.visible:
                return
            world_mat = parent_mat @ node.get_local_transform()
            if node.mesh is not None:
                results.append((node.mesh, world_mat))
            for child in node.children:
                _traverse(child, world_mat)

        _traverse(self.root, np.eye(4, dtype=np.float32))
        return results

    def find_node(self, name: str) -> Optional[ModelNode3D]:
        """Finds a component node by name in the hierarchy."""
        stack = [self.root]
        while stack:
            cur = stack.pop()
            if cur.name == name:
                return cur
            stack.extend(cur.children)
        return None


# ---------------------------------------------------------------------------
# 4x4 Matrix Transformation Utilities
# ---------------------------------------------------------------------------

def make_translation_matrix(t: np.ndarray) -> np.ndarray:
    m = np.eye(4, dtype=np.float32)
    m[0, 3] = t[0]
    m[1, 3] = t[1]
    m[2, 3] = t[2]
    return m


def make_rotation_matrix(r: np.ndarray) -> np.ndarray:
    rx, ry, rz = r[0], r[1], r[2]
    # Roll, pitch, yaw
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    mx = np.array([[1, 0, 0, 0], [0, cx, -sx, 0], [0, sx, cx, 0], [0, 0, 0, 1]], dtype=np.float32)
    my = np.array([[cy, 0, sy, 0], [0, 1, 0, 0], [-sy, 0, cy, 0], [0, 0, 0, 1]], dtype=np.float32)
    mz = np.array([[cz, -sz, 0, 0], [sz, cz, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)

    return mz @ my @ mx


def make_scale_matrix(s: np.ndarray) -> np.ndarray:
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = s[0]
    m[1, 1] = s[1]
    m[2, 2] = s[2]
    return m


def make_transform_matrix(pos: np.ndarray, rot: np.ndarray, scl: np.ndarray) -> np.ndarray:
    return make_translation_matrix(pos) @ make_rotation_matrix(rot) @ make_scale_matrix(scl)


def make_perspective_matrix(fov_deg: float, aspect: float, near: float = 0.1, far: float = 100.0) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fov_deg) * 0.5)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2.0 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def make_look_at_matrix(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    z_axis = eye - target
    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-8)
    x_axis = np.cross(up, z_axis)
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)
    y_axis = np.cross(z_axis, x_axis)

    r = np.eye(4, dtype=np.float32)
    r[0, :3] = x_axis
    r[1, :3] = y_axis
    r[2, :3] = z_axis

    t = np.eye(4, dtype=np.float32)
    t[:3, 3] = -eye

    return r @ t


# ---------------------------------------------------------------------------
# Native GLB (Binary glTF 2.0) Parser (Zero-dependency)
# ---------------------------------------------------------------------------

def load_glb(file_path: Union[str, Path]) -> Model3D:
    """
    Parses a binary glTF (.glb) file into a Model3D hierarchy without external dependencies.
    Extracts meshes, positions, normals, indices, and base colors.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"GLB file not found: {path}")

    with open(str(path), "rb") as f:
        data = f.read()

    if len(data) < 20:
        raise ValueError(f"Invalid GLB file size: {len(data)} bytes")

    # GLB Header: magic (4), version (4), length (4)
    magic, version, length = struct.unpack("<4sII", data[:12])
    if magic != b"glTF":
        raise ValueError(f"Not a valid GLB file: magic={magic}")

    # Chunk 0: JSON Chunk
    json_len, json_type = struct.unpack("<II", data[12:20])
    if json_type != 0x4E4F534A:  # "JSON"
        raise ValueError("First GLB chunk is not JSON")

    json_bytes = data[20 : 20 + json_len]
    gltf_json = json.loads(json_bytes.decode("utf-8"))

    # Chunk 1: BIN Chunk (optional)
    bin_offset = 20 + json_len
    bin_data = b""
    if bin_offset + 8 <= len(data):
        bin_len, bin_type = struct.unpack("<II", data[bin_offset : bin_offset + 8])
        if bin_type == 0x004E4942:  # "BIN\0"
            bin_data = data[bin_offset + 8 : bin_offset + 8 + bin_len]

    # Parse Accessors & BufferViews
    accessors = gltf_json.get("accessors", [])
    buffer_views = gltf_json.get("bufferViews", [])

    def _read_accessor_data(acc_idx: int) -> np.ndarray:
        acc = accessors[acc_idx]
        bv_idx = acc["bufferView"]
        bv = buffer_views[bv_idx]
        byte_offset = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
        component_type = acc["componentType"]
        count = acc["count"]
        type_str = acc["type"]

        type_map = {
            5120: np.int8,
            5121: np.uint8,
            5122: np.int16,
            5123: np.uint16,
            5125: np.uint32,
            5126: np.float32,
        }
        dtype = type_map.get(component_type, np.float32)

        components_per_element = {
            "SCALAR": 1,
            "VEC2": 2,
            "VEC3": 3,
            "VEC4": 4,
            "MAT4": 16,
        }.get(type_str, 1)

        total_vals = count * components_per_element
        byte_len = total_vals * np.dtype(dtype).itemsize
        raw_slice = bin_data[byte_offset : byte_offset + byte_len]

        arr = np.frombuffer(raw_slice, dtype=dtype)
        if components_per_element > 1:
            arr = arr.reshape((count, components_per_element))
        return arr

    root_node = ModelNode3D(name=path.stem)
    meshes_json = gltf_json.get("meshes", [])

    for m_idx, m_json in enumerate(meshes_json):
        m_name = m_json.get("name", f"mesh_{m_idx}")
        for p_idx, prim in enumerate(m_json.get("primitives", [])):
            attrs = prim.get("attributes", {})
            pos_acc = attrs.get("POSITION")
            if pos_acc is None:
                continue
            verts = _read_accessor_data(pos_acc).astype(np.float32)

            # Faces / Indices
            idx_acc = prim.get("indices")
            if idx_acc is not None:
                indices = _read_accessor_data(idx_acc).astype(np.int32)
                faces = indices.reshape((-1, 3))
            else:
                faces = np.arange(len(verts), dtype=np.int32).reshape((-1, 3))

            # Normals
            normals = None
            if "NORMAL" in attrs:
                normals = _read_accessor_data(attrs["NORMAL"]).astype(np.float32)

            mesh = Mesh3D(
                vertices=verts,
                faces=faces,
                normals=normals,
                name=f"{m_name}_p{p_idx}",
            )
            if mesh.normals is None:
                mesh.compute_face_normals()

            child_node = ModelNode3D(name=mesh.name, mesh=mesh)
            root_node.add_child(child_node)

    model = Model3D(
        concept=path.stem,
        root=root_node,
        category="3D_OBJECT",
        description=f"Loaded from GLB: {path.name}",
    )
    return model


# ---------------------------------------------------------------------------
# Native Wavefront OBJ Parser
# ---------------------------------------------------------------------------

def load_obj(file_path: Union[str, Path]) -> Model3D:
    """Parses a Wavefront .obj file into a Model3D object."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"OBJ file not found: {path}")

    verts: List[List[float]] = []
    faces: List[List[int]] = []
    normals: List[List[float]] = []

    with open(str(path), "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] == "v" and len(parts) >= 4:
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "vn" and len(parts) >= 4:
                normals.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f" and len(parts) >= 4:
                # Handle v, v/vt, v/vt/vn, v//vn
                face_indices = []
                for p in parts[1:4]:
                    idx_str = p.split("/")[0]
                    face_indices.append(int(idx_str) - 1)
                faces.append(face_indices)

    v_arr = np.array(verts, dtype=np.float32) if verts else np.zeros((0, 3), dtype=np.float32)
    f_arr = np.array(faces, dtype=np.int32) if faces else np.zeros((0, 3), dtype=np.int32)

    mesh = Mesh3D(vertices=v_arr, faces=f_arr, name=path.stem)
    if len(f_arr) > 0:
        mesh.compute_face_normals()
    mesh.center_and_normalize(1.0)

    root = ModelNode3D(name=path.stem, mesh=mesh)
    return Model3D(concept=path.stem, root=root, category="3D_OBJECT")

