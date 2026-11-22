"""
integration/generation/renderer3d.py — Real-time 3D Software Rasterizer & Viewport Renderer.

Provides:
- Native, zero-dependency 3D mesh rasterization into OpenCV image buffers.
- Perspective projection, viewport transformation inside VisualOpportunity bounding boxes.
- Diffuse Lambertian lighting + specular highlight shading.
- Back-face culling and depth sorting (Painter's algorithm with stable depth).
- Component hierarchy evaluation and smooth continuous animation over time parameter t.
- Operates at 30+ FPS directly in the camera frame rendering loop.
"""

from __future__ import annotations

import math
import time
from typing import List, Optional, Tuple
import cv2
import numpy as np

from integration.generation.model3d import (
    Material3D,
    Mesh3D,
    Model3D,
    make_look_at_matrix,
    make_perspective_matrix,
    make_rotation_matrix,
)


class SoftwareRenderer3D:
    """High-performance 3D software rasterizer rendering into 2D camera frames."""

    def __init__(
        self,
        fov_deg: float = 45.0,
        light_dir: Optional[np.ndarray] = None,
        ambient_intensity: float = 0.48,
        diffuse_intensity: float = 0.82,
    ) -> None:
        self.fov_deg = fov_deg
        ld = light_dir if light_dir is not None else np.array([0.4, 0.7, 0.8], dtype=np.float32)
        self.light_dir = ld / (np.linalg.norm(ld) + 1e-8)
        self.ambient_intensity = ambient_intensity
        self.diffuse_intensity = diffuse_intensity

        # Fixed camera setup looking at origin from z=3.0
        self.eye = np.array([0.0, 0.0, 3.2], dtype=np.float32)
        self.target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.view_matrix = make_look_at_matrix(self.eye, self.target, self.up)

    def render(
        self,
        canvas: np.ndarray,
        model: Model3D,
        bbox: Tuple[float, float, float, float],
        t: float = 0.0,
        manual_rotation: Optional[Tuple[float, float, float]] = None,
        pad: int = 4,
    ) -> bool:
        """
        Renders the 3D model directly into the canvas within the normalized bbox.
        Maintains aspect ratio inside box interior.
        """
        if canvas is None or canvas.size == 0 or model is None:
            return False

        h, w = canvas.shape[:2]
        x1 = max(0, min(w - 1, int(round(bbox[0] * w))))
        y1 = max(0, min(h - 1, int(round(bbox[1] * h))))
        x2 = max(x1 + 10, min(w, int(round(bbox[2] * w))))
        y2 = max(y1 + 10, min(h, int(round(bbox[3] * h))))

        # Interior viewport bounds
        vx1 = x1 + pad
        vy1 = y1 + pad
        vx2 = x2 - pad
        vy2 = y2 - pad
        vw = vx2 - vx1
        vh = vy2 - vy1

        if vw <= 8 or vh <= 8:
            return False

        aspect = float(vw) / float(vh)
        proj_matrix = make_perspective_matrix(self.fov_deg, aspect, near=0.1, far=50.0)
        vp_matrix = proj_matrix @ self.view_matrix

        # Base model rotation (e.g. gentle continuous turntable + manual pitch/yaw)
        base_rot_arr = np.array([0.25, t * 0.8, 0.0], dtype=np.float32)
        if manual_rotation is not None:
            base_rot_arr += np.array(manual_rotation, dtype=np.float32)
        base_model_rot = make_rotation_matrix(base_rot_arr)

        # Half vector for Blinn-Phong specular calculation (constant per frame)
        view_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        half_vector = self.light_dir + view_dir
        half_vector /= (np.linalg.norm(half_vector) + 1e-8)

        # Collect all active meshes with node hierarchy transforms
        mesh_instances = model.get_all_meshes_with_transforms(t=t)
        if not mesh_instances:
            return False

        # Triangles to sort across all meshes: list of (avg_z, screen_pts_3x2, color_bgr, alpha)
        render_queue: List[Tuple[float, np.ndarray, Tuple[int, int, int], float]] = []

        for mesh, local_world_mat in mesh_instances:
            if mesh.vertices.size == 0 or mesh.faces.size == 0:
                continue

            full_world = base_model_rot @ local_world_mat
            mvp = vp_matrix @ full_world

            # Transform vertices to homogeneous coordinates: (N, 4)
            verts_h = np.hstack([mesh.vertices, np.ones((len(mesh.vertices), 1), dtype=np.float32)])
            clip_pts = (mvp @ verts_h.T).T  # (N, 4)

            w_clip = clip_pts[:, 3:4]
            # Avoid divide-by-zero
            w_safe = np.where(np.abs(w_clip) < 1e-6, 1e-6, w_clip)
            ndc = clip_pts[:, :3] / w_safe

            # Screen coordinates inside viewport
            sx = vx1 + (ndc[:, 0] * 0.5 + 0.5) * vw
            # Flip Y (screen Y goes down, NDC goes up)
            sy = vy1 + (-ndc[:, 1] * 0.5 + 0.5) * vh
            screen_coords = np.column_stack([sx, sy])

            # Vectorized backface and depth culling BEFORE expensive lighting
            faces = mesh.faces
            p0 = screen_coords[faces[:, 0]]
            p1 = screen_coords[faces[:, 1]]
            p2 = screen_coords[faces[:, 2]]

            cross_2d = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])
            front_mask = (cross_2d > 0) if not mesh.material.double_sided else np.ones(len(faces), dtype=bool)

            z_depths = (clip_pts[faces[:, 0], 2] + clip_pts[faces[:, 1], 2] + clip_pts[faces[:, 2], 2]) * (1.0 / 3.0)
            valid_mask = front_mask & (z_depths > 0.05) & (z_depths < 50.0)

            valid_indices = np.where(valid_mask)[0]
            if len(valid_indices) == 0:
                continue

            # Transform surface normals only for visible faces
            mesh_normals = mesh.normals
            if mesh_normals is None or len(mesh_normals) != len(mesh.faces):
                mesh_normals = mesh.compute_face_normals()

            vis_normals = mesh_normals[valid_indices]
            norm_mat = full_world[:3, :3]
            world_normals = (norm_mat @ vis_normals.T).T
            norm_lengths = np.linalg.norm(world_normals, axis=1, keepdims=True)
            norm_lengths[norm_lengths == 0] = 1.0
            world_normals /= norm_lengths

            # Key light (directional, upper-right front)
            key_dots = np.clip(np.sum(world_normals * self.light_dir, axis=1), 0.0, 1.0)
            # Fill light (camera view direction, ensures visible unlit surfaces remain richly illuminated)
            fill_dots = np.clip(np.sum(world_normals * view_dir, axis=1), 0.0, 1.0)

            amb = mesh.material.ambient_coeff
            diff = mesh.material.diffuse_coeff
            ambient_fill = max(0.52, amb * self.ambient_intensity)
            diffuse_fill = self.diffuse_intensity * diff * key_dots
            fill_light = 0.35 * fill_dots

            spec_dot = np.clip(np.sum(world_normals * half_vector, axis=1), 0.0, 1.0)
            roughness = float(np.clip(mesh.material.roughness, 0.04, 1.0))
            spec_power = max(4.0, min(float(mesh.material.shininess), 96.0) * (1.0 - 0.55 * roughness))
            spec = (np.power(spec_dot, spec_power) * mesh.material.specular_coeff *
                    (1.0 - 0.55 * roughness))
            intensities = np.clip(ambient_fill + diffuse_fill + fill_light + spec, 0.52, 1.35)

            mat_color = np.array(mesh.material.diffuse_color, dtype=np.float32)
            face_colors = mesh.colors if mesh.colors is not None and len(mesh.colors) == len(mesh.faces) else None
            opacity = mesh.material.opacity

            # Vectorized color calculation for visible faces
            if face_colors is not None:
                base_c = face_colors[valid_indices].astype(np.float32)
            else:
                base_c = mat_color

            int_vals = intensities[:, None]
            lit_colors = np.clip(base_c * int_vals, 0, 255).astype(np.uint8)

            p0_v = p0[valid_indices]
            p1_v = p1[valid_indices]
            p2_v = p2[valid_indices]
            z_v = z_depths[valid_indices]

            # Vectorized polygon construction and bounds check
            polys = np.stack([p0_v, p1_v, p2_v], axis=1).astype(np.int32)
            min_x = np.min(polys[:, :, 0], axis=1)
            max_x = np.max(polys[:, :, 0], axis=1)
            min_y = np.min(polys[:, :, 1], axis=1)
            max_y = np.max(polys[:, :, 1], axis=1)
            in_bounds = (max_x >= vx1) & (min_x <= vx2) & (max_y >= vy1) & (min_y <= vy2)

            if not np.any(in_bounds):
                continue

            polys = polys[in_bounds]
            lit_colors = lit_colors[in_bounds]
            z_v = z_v[in_bounds]

            for k in range(len(polys)):
                col = (int(lit_colors[k, 0]), int(lit_colors[k, 1]), int(lit_colors[k, 2]))
                render_queue.append((float(z_v[k]), polys[k], col, opacity))

        if not render_queue:
            return False

        # Sort depth from back to front (Painter's algorithm)
        render_queue.sort(key=lambda item: item[0], reverse=True)

        # Rasterize triangles
        roi = canvas[vy1:vy2, vx1:vx2]
        needs_alpha = any(item[3] < 0.99 for item in render_queue)

        # Contact shadow beneath object to ground it gently without opaque black blocks
        all_points = np.concatenate([item[1] for item in render_queue], axis=0)
        object_left = int(np.clip(all_points[:, 0].min() - vx1, 0, vw - 1))
        object_right = int(np.clip(all_points[:, 0].max() - vx1, 0, vw - 1))
        object_bottom = int(np.clip(all_points[:, 1].max() - vy1, 0, vh - 1))
        shadow_center = (vx1 + (object_left + object_right) // 2, min(vy2 - 1, vy1 + object_bottom + max(2, vh // 36)))
        shadow_axes = (max(4, (object_right - object_left) // 3), max(2, vh // 32))

        sy1 = max(0, shadow_center[1] - shadow_axes[1] * 2)
        sy2 = min(h, shadow_center[1] + shadow_axes[1] * 2 + 1)
        sx1 = max(0, shadow_center[0] - shadow_axes[0] * 2)
        sx2 = min(w, shadow_center[0] + shadow_axes[0] * 2 + 1)
        if sy2 > sy1 and sx2 > sx1:
            s_patch = np.zeros((sy2 - sy1, sx2 - sx1), dtype=np.uint8)
            cv2.ellipse(s_patch, (shadow_center[0] - sx1, shadow_center[1] - sy1), shadow_axes, 0, 0, 360, 255, -1, cv2.LINE_AA)
            s_patch = cv2.GaussianBlur(s_patch, (0, 0), sigmaX=max(1.5, vw / 90.0))
            s_alpha = (s_patch.astype(np.float32) / 255.0 * 0.28)[..., None]
            canvas[sy1:sy2, sx1:sx2] = (canvas[sy1:sy2, sx1:sx2].astype(np.float32) * (1.0 - s_alpha)).astype(np.uint8)

        if needs_alpha:
            for _, poly, color, alpha in render_queue:
                poly_local = poly - np.array([vx1, vy1], dtype=np.int32)
                if alpha >= 0.99:
                    cv2.fillConvexPoly(roi, poly_local, color, cv2.LINE_8)
                else:
                    bx1 = max(0, int(poly_local[:, 0].min()))
                    by1 = max(0, int(poly_local[:, 1].min()))
                    bx2 = min(vw, int(poly_local[:, 0].max()) + 1)
                    by2 = min(vh, int(poly_local[:, 1].max()) + 1)
                    if bx2 > bx1 and by2 > by1:
                        p_sub = poly_local - np.array([bx1, by1], dtype=np.int32)
                        sub_mask = np.zeros((by2 - by1, bx2 - bx1), dtype=np.uint8)
                        cv2.fillConvexPoly(sub_mask, p_sub, 255, cv2.LINE_8)
                        m = sub_mask > 0
                        sub_roi = roi[by1:by2, bx1:bx2]
                        c_arr = np.array(color, dtype=np.float32)
                        sub_roi[m] = (c_arr * alpha + sub_roi[m].astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
        else:
            for _, poly, color, _ in render_queue:
                cv2.fillConvexPoly(canvas, poly, color, cv2.LINE_8)

        return True
