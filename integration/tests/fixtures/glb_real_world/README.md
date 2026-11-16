# Real-world GLB validation fixtures

These three individual samples are from the official Khronos glTF Sample Assets repository. They are stored here solely as a small regression set, not as a downloaded dataset. Source files were downloaded from the repository's `main` branch on 2026-11-15.

| Fixture | Coverage | License and attribution | Source |
|---|---|---|---|
| `BoxTextured.glb` | Static mesh, base color/material and embedded texture detection | CC BY 4.0; Cesium (2017, 2015). Preserve attribution; the sample also contains a Cesium mark. | [Asset folder](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/BoxTextured), [license metadata](https://github.com/KhronosGroup/glTF-Sample-Assets/blob/main/REUSE.toml) |
| `CesiumMilkTruck.glb` | Multiple mesh primitives, deep node hierarchy, nested translation transforms | CC BY 4.0; Cesium (2017, 2015). Preserve attribution; the sample also contains a Cesium mark. | [Asset folder](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/CesiumMilkTruck), [license metadata](https://github.com/KhronosGroup/glTF-Sample-Assets/blob/main/REUSE.toml) |
| `BoxAnimated.glb` | Node rotation and translation animation channels | CC BY 4.0; Cesium (2017). Preserve attribution. | [Asset folder](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/BoxAnimated), [license metadata](https://github.com/KhronosGroup/glTF-Sample-Assets/blob/main/REUSE.toml) |

The BoxTextured and CesiumMilkTruck assets' REUSE entries also identify `LicenseRef-LegalMark-Cesium` for the Cesium trademark/logo. Their CC BY license does not grant trademark rights. For this test corpus, retain the full upstream license records and do not use the marks as project branding.

The importer does not represent image textures in `Model3D` or the renderer. The texture case therefore validates that the GLB and geometry load and that the importer reports its unsupported texture count; it does not claim rendered texture fidelity. These tests exercise the real importer data structures, not visual output in the renderer. The generated fixture in `test_glb_importer_phase2.py` remains a separate synthetic feature test.
