import json
import struct
from pathlib import Path

import numpy as np

from integration.generation.model3d import load_glb


def _write_fixture(path: Path) -> None:
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype='<f4').tobytes()
    times = np.array([0, 1], dtype='<f4').tobytes()
    translations = np.array([[0, 0, 0], [2, 0, 0]], dtype='<f4').tobytes()
    binary = positions + times + translations
    doc = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 36},
            {"buffer": 0, "byteOffset": 36, "byteLength": 8},
            {"buffer": 0, "byteOffset": 44, "byteLength": 24},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5126, "count": 2, "type": "SCALAR"},
            {"bufferView": 2, "componentType": 5126, "count": 2, "type": "VEC3"},
        ],
        "materials": [{"pbrMetallicRoughness": {"baseColorFactor": [0.8, 0.4, 0.2, 0.5]}}],
        "meshes": [{"name": "triangle", "primitives": [{"attributes": {"POSITION": 0}, "material": 0}]}],
        "nodes": [{"name": "animated", "mesh": 0, "translation": [1, 0, 0]}],
        "scenes": [{"nodes": [0]}], "scene": 0,
        "animations": [{"samplers": [{"input": 1, "output": 2, "interpolation": "LINEAR"}], "channels": [{"sampler": 0, "target": {"node": 0, "path": "translation"}}]}],
    }
    json_bytes = json.dumps(doc, separators=(',', ':')).encode()
    json_bytes += b' ' * ((-len(json_bytes)) % 4)
    binary += b'\0' * ((-len(binary)) % 4)
    total = 12 + 8 + len(json_bytes) + 8 + len(binary)
    path.write_bytes(struct.pack('<4sII', b'glTF', 2, total) + struct.pack('<II', len(json_bytes), 0x4E4F534A) + json_bytes + struct.pack('<II', len(binary), 0x004E4942) + binary)


def test_glb_hierarchy_materials_and_trs_animation(tmp_path):
    fixture = tmp_path / 'phase2.glb'
    _write_fixture(fixture)
    model = load_glb(fixture)

    assert model.animation_fn is not None
    assert model.find_node('animated') is not None
    mesh, transform = model.get_all_meshes_with_transforms(t=0)[0]
    assert np.allclose(transform[:3, 3], [0, 0, 0])
    assert mesh.material.diffuse_color == (51, 102, 204)
    assert mesh.material.opacity == 0.5

    _, animated_transform = model.get_all_meshes_with_transforms(t=0.5)[0]
    assert np.allclose(animated_transform[:3, 3], [1, 0, 0])
    _, animated_transform = model.get_all_meshes_with_transforms(t=1)[0]
    assert np.allclose(animated_transform[:3, 3], [2, 0, 0])

