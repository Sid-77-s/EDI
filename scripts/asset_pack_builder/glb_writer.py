"""
scripts/asset_pack_builder/glb_writer.py — Native GLB v2 Exporter & Validator for Magic Mirror.
"""
from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from integration.generation.model3d import Material3D, Mesh3D, Model3D, ModelNode3D, load_glb


def model_to_glb(model: Model3D, path: Union[str, Path]) -> int:
    """Exports a Model3D hierarchy to a 100% compliant binary GLB 2.0 file."""
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    meshes: List[Dict[str, Any]] = []
    materials: List[Dict[str, Any]] = []
    nodes: List[Dict[str, Any]] = []
    bin_chunks: List[bytes] = []
    buffer_views: List[Dict[str, Any]] = []
    accessors: List[Dict[str, Any]] = []
    mat_map: Dict[Tuple, int] = {}

    def get_mat_idx(mat: Material3D) -> int:
        key = (mat.diffuse_color, mat.roughness, mat.metallic, mat.opacity, mat.double_sided)
        if key in mat_map:
            return mat_map[key]
        idx = len(materials)
        b, g, r = mat.diffuse_color
        rgba = [float(r) / 255.0, float(g) / 255.0, float(b) / 255.0, float(mat.opacity)]
        materials.append({
            "pbrMetallicRoughness": {
                "baseColorFactor": rgba,
                "roughnessFactor": float(mat.roughness),
                "metallicFactor": float(mat.metallic),
            },
            "doubleSided": bool(mat.double_sided),
        })
        mat_map[key] = idx
        return idx

    def add_buffer_data(data_bytes: bytes) -> int:
        offset = sum(len(b) for b in bin_chunks)
        length = len(data_bytes)
        bin_chunks.append(data_bytes)
        pad = (4 - (length % 4)) % 4
        if pad > 0:
            bin_chunks.append(b"\x00" * pad)
        view_idx = len(buffer_views)
        buffer_views.append({"buffer": 0, "byteOffset": offset, "byteLength": length})
        return view_idx

    def process_node(node: ModelNode3D, node_tracker: List[int]) -> int:
        current_idx = node_tracker[0]
        node_tracker[0] += 1

        node_dict: Dict[str, Any] = {"name": node.name}
        if not np.allclose(node.position, [0, 0, 0], atol=1e-5):
            node_dict["translation"] = [float(x) for x in node.position]
        if not np.allclose(node.scale, [1, 1, 1], atol=1e-5):
            node_dict["scale"] = [float(x) for x in node.scale]
        if not np.allclose(node.rotation, [0, 0, 0], atol=1e-5):
            # Euler angles to quaternion [x, y, z, w]
            rx, ry, rz = float(node.rotation[0]), float(node.rotation[1]), float(node.rotation[2])
            cx, cy, cz = math.cos(rx * 0.5), math.cos(ry * 0.5), math.cos(rz * 0.5)
            sx, sy, sz = math.sin(rx * 0.5), math.sin(ry * 0.5), math.sin(rz * 0.5)
            qx = sx * cy * cz - cx * sy * sz
            qy = cx * cy * cz + sx * cy * sz
            qz = cx * cy * sz - sx * sy * cz
            qw = cx * cy * cz + sx * sy * sz
            node_dict["rotation"] = [float(qx), float(qy), float(qz), float(qw)]

        if node.mesh is not None:
            m = node.mesh
            verts = m.vertices.astype(np.float32)
            faces = m.faces.astype(np.uint32)
            normals = m.normals if m.normals is not None else m.compute_face_normals()

            if len(normals) == len(faces):
                v_norms = np.zeros_like(verts)
                counts = np.zeros(len(verts), dtype=np.float32)
                for f_idx, f in enumerate(faces):
                    v_norms[f] += normals[f_idx]
                    counts[f] += 1.0
                counts[counts == 0] = 1.0
                v_norms /= counts[:, None]
                n_norm = np.linalg.norm(v_norms, axis=1, keepdims=True)
                n_norm[n_norm == 0] = 1.0
                v_norms /= n_norm
                normals = v_norms.astype(np.float32)
            else:
                normals = normals.astype(np.float32)

            vb = verts.tobytes()
            nb = normals.tobytes()
            fb = faces.flatten().tobytes()

            v_view = add_buffer_data(vb)
            n_view = add_buffer_data(nb)
            f_view = add_buffer_data(fb)

            v_acc = len(accessors)
            accessors.append({
                "bufferView": v_view,
                "byteOffset": 0,
                "componentType": 5126,  # FLOAT
                "count": len(verts),
                "type": "VEC3",
                "min": [float(x) for x in verts.min(axis=0)],
                "max": [float(x) for x in verts.max(axis=0)],
            })
            n_acc = len(accessors)
            accessors.append({
                "bufferView": n_view,
                "byteOffset": 0,
                "componentType": 5126,  # FLOAT
                "count": len(normals),
                "type": "VEC3",
                "min": [float(x) for x in normals.min(axis=0)],
                "max": [float(x) for x in normals.max(axis=0)],
            })
            f_acc = len(accessors)
            accessors.append({
                "bufferView": f_view,
                "byteOffset": 0,
                "componentType": 5125,  # UNSIGNED_INT
                "count": len(faces) * 3,
                "type": "SCALAR",
                "min": [int(faces.min())],
                "max": [int(faces.max())],
            })

            mat_idx = get_mat_idx(m.material)
            mesh_idx = len(meshes)
            meshes.append({
                "name": m.name,
                "primitives": [{
                    "attributes": {"POSITION": v_acc, "NORMAL": n_acc},
                    "indices": f_acc,
                    "material": mat_idx,
                }],
            })
            node_dict["mesh"] = mesh_idx

        nodes.append(node_dict)

        children_indices: List[int] = []
        for child in node.children:
            ch_idx = process_node(child, node_tracker)
            children_indices.append(ch_idx)
        if children_indices:
            node_dict["children"] = children_indices

        return current_idx

    node_tracker = [0]
    root_idx = process_node(model.root, node_tracker)

    bin_data = b"".join(bin_chunks)
    doc = {
        "asset": {"version": "2.0", "generator": "MagicMirror Asset Library Generator"},
        "buffers": [{"byteLength": len(bin_data)}],
        "bufferViews": buffer_views,
        "accessors": accessors,
        "materials": materials,
        "meshes": meshes,
        "nodes": nodes,
        "scenes": [{"nodes": [root_idx]}],
        "scene": 0,
    }

    json_bytes = json.dumps(doc, separators=(",", ":")).encode("utf-8")
    pad = (4 - (len(json_bytes) % 4)) % 4
    json_bytes += b" " * pad

    total_len = 12 + 8 + len(json_bytes) + 8 + len(bin_data)
    hdr = struct.pack("<4sII", b"glTF", 2, total_len)
    ch0 = struct.pack("<II", len(json_bytes), 0x4E4F534A)  # JSON
    ch1 = struct.pack("<II", len(bin_data), 0x004E4942)   # BIN

    total_payload = hdr + ch0 + json_bytes + ch1 + bin_data
    target_path.write_bytes(total_payload)
    return len(total_payload)


def write_asset_sidecar(glb_path: Union[str, Path], metadata: Dict[str, Any]) -> Path:
    """Writes the metadata sidecar <model>.json next to <model>.glb."""
    glb_p = Path(glb_path)
    sidecar_p = glb_p.with_suffix(".json")
    sidecar_p.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return sidecar_p


def validate_glb_asset(glb_path: Union[str, Path]) -> Dict[str, Any]:
    """Validates an exported GLB file using Magic Mirror's native loader."""
    p = Path(glb_path)
    if not p.is_file():
        return {"valid": False, "error": f"File not found: {p}"}

    file_size = p.stat().st_size
    if file_size < 20:
        return {"valid": False, "error": "File too small for GLB header"}

    try:
        model = load_glb(p)
        mesh_instances = model.get_all_meshes_with_transforms()
        mesh_count = len(mesh_instances)
        triangle_count = sum(len(m.faces) for m, _ in mesh_instances)

        if mesh_count == 0 or triangle_count == 0:
            return {"valid": False, "error": "No meshes or triangles found"}

        # Validate bounding box and finite coordinates
        for mesh, _ in mesh_instances:
            if not np.all(np.isfinite(mesh.vertices)):
                return {"valid": False, "error": "Non-finite vertex coordinates"}
            if not np.all(np.isfinite(mesh.faces)):
                return {"valid": False, "error": "Non-finite face indices"}

        return {
            "valid": True,
            "mesh_count": mesh_count,
            "triangle_count": triangle_count,
            "file_size_bytes": file_size,
        }
    except Exception as exc:
        return {"valid": False, "error": str(exc)}
