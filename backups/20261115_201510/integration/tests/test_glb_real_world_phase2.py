from pathlib import Path

import numpy as np

from integration.generation.model3d import load_glb


FIXTURES = Path(__file__).parent / "fixtures" / "glb_real_world"


def _nodes(root):
    stack = [(root, 0)]
    while stack:
        node, depth = stack.pop()
        yield node, depth
        stack.extend((child, depth + 1) for child in reversed(node.children))


def test_real_static_textured_box_loads_geometry_and_reports_texture_limit():
    model = load_glb(FIXTURES / "BoxTextured.glb")
    rendered_meshes = model.get_all_meshes_with_transforms()

    assert len(rendered_meshes) == 1
    mesh, _ = rendered_meshes[0]
    assert len(mesh.vertices) > 0
    assert len(mesh.faces) > 0
    assert model.animation_fn is None
    # GLB is valid and its embedded image is detected, but Model3D has no
    # texture slot; retain this as an explicit current importer limitation.
    assert model.metadata["unsupported_textures"] == 1
    assert mesh.colors is None


def test_real_milk_truck_preserves_multi_mesh_nested_hierarchy_and_transforms():
    model = load_glb(FIXTURES / "CesiumMilkTruck.glb")
    nodes = list(_nodes(model.root))
    rendered_meshes = model.get_all_meshes_with_transforms()

    assert len(rendered_meshes) >= 5
    assert any(node.name == "Wheels" for node, _ in nodes)
    assert max(depth for _, depth in nodes) >= 5

    positioned = model.find_node("Node")
    assert positioned is not None
    world = positioned.get_world_transform()
    # Source quaternion at Yup2Zup rotates the truck-local translation to this world point.\n    assert np.allclose(world[:3, 3], [0.0, 0.427722, 1.43267], atol=1e-4)


def test_real_animated_box_exposes_changing_node_animation():
    model = load_glb(FIXTURES / "BoxAnimated.glb")
    assert model.animation_fn is not None

    at_start = model.get_all_meshes_with_transforms(t=0.0)
    midway = model.get_all_meshes_with_transforms(t=0.5)
    at_end = model.get_all_meshes_with_transforms(t=1.0)
    assert at_start and len(at_start) == len(midway) == len(at_end)
    assert any(
        not np.allclose(start_transform, end_transform)
        for (_, start_transform), (_, end_transform) in zip(at_start, at_end)
    )

