"""Pinhole projection from CARLA world coordinates to image pixels.

Pure numpy + math — no CARLA, no image. The caller extracts the 8 bbox
vertices from the actor (``actor.bounding_box.get_world_vertices(actor.get_transform())``)
and passes the camera's 4×4 world transform together with intrinsics.

Axis convention: CARLA/UE4 is left-handed with X forward, Y right, Z up.
Standard CV camera is right-handed with X right, Y down, Z forward. The
remap ``(cam_x, cam_y, cam_z) → (cam_y, -cam_z, cam_x)`` applied here
matches the CARLA PythonAPI ``client_bounding_boxes.py`` example.
"""

from __future__ import annotations

import math

import numpy as np


def build_camera_matrix(width: int, height: int, fov_deg: float) -> np.ndarray:
    """Pinhole intrinsics for a CARLA camera. FOV is horizontal; pixels are square."""
    focal = width / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    K = np.eye(3)
    K[0, 0] = focal
    K[1, 1] = focal
    K[0, 2] = width / 2.0
    K[1, 2] = height / 2.0
    return K


def project_points(
    points_world: np.ndarray,
    world_to_camera: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    """Project (N, 3) world points to (N, 2) pixel coordinates.

    Points behind the camera (z_cv ≤ 0) are returned as NaN — caller filters.
    """
    if points_world.ndim != 2 or points_world.shape[1] != 3:
        raise ValueError(f"points_world must be (N, 3), got {points_world.shape}")
    n = points_world.shape[0]
    pts_h = np.concatenate([points_world, np.ones((n, 1))], axis=1).T  # (4, N)
    pts_cam = world_to_camera @ pts_h                                   # (4, N)
    # UE4 → CV frame remap: (x, y, z) → (y, -z, x)
    x_cv = pts_cam[1]
    y_cv = -pts_cam[2]
    z_cv = pts_cam[0]
    behind = z_cv <= 1e-6
    with np.errstate(invalid="ignore", divide="ignore"):
        u = (K[0, 0] * x_cv) / z_cv + K[0, 2]
        v = (K[1, 1] * y_cv) / z_cv + K[1, 2]
    u = np.where(behind, np.nan, u)
    v = np.where(behind, np.nan, v)
    return np.stack([u, v], axis=1)


def world_bbox_to_image(
    vertices_world: np.ndarray,
    world_to_camera: np.ndarray,
    K: np.ndarray,
    image_size: tuple[int, int],
) -> tuple[float, float, float, float] | None:
    """Project 8 world-space bbox vertices to an axis-aligned image bbox.

    ``image_size`` is ``(height, width)`` to match numpy/OpenCV convention.
    Returns ``None`` if every vertex is behind the camera or the resulting
    rectangle is empty after clipping to image bounds.
    """
    h, w = image_size
    uv = project_points(vertices_world, world_to_camera, K)
    valid = ~np.isnan(uv).any(axis=1)
    if int(valid.sum()) == 0:
        return None
    us = uv[valid, 0]
    vs = uv[valid, 1]
    x1 = float(np.clip(us.min(), 0, w - 1))
    y1 = float(np.clip(vs.min(), 0, h - 1))
    x2 = float(np.clip(us.max(), 0, w - 1))
    y2 = float(np.clip(vs.max(), 0, h - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def pixel_to_world_on_ground(
    u: float,
    v: float,
    K: np.ndarray,
    camera_to_world: np.ndarray,
    ground_z: float = 0.0,
) -> tuple[float, float] | None:
    """Inverse of ``project_points`` restricted to a ground plane.

    Shoot a ray from the camera origin through pixel ``(u, v)`` and intersect
    it with the plane ``z = ground_z``. Returns the world-space ``(x, y)`` of
    the intersection, or ``None`` if the ray doesn't hit (parallel to the
    plane, or intersection behind the camera).

    This is the closed-form inverse used by the flight PID — given the locked
    track's bbox centre pixel, it recovers the suspect's world position
    without a small-angle approximation or altitude-proxy shortcut. Works at
    any camera pitch/yaw/altitude.

    ``camera_to_world`` is CARLA's ``camera.get_transform().get_matrix()``
    (4×4). Intrinsics ``K`` come from ``build_camera_matrix``.
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Ray direction in standard CV camera frame (X right, Y down, Z forward).
    d_cv = np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=np.float64)

    # Undo the UE4 → CV remap from ``project_points``:
    #   cv_x =  cam_y     →  cam_y =  cv_x
    #   cv_y = -cam_z     →  cam_z = -cv_y
    #   cv_z =  cam_x     →  cam_x =  cv_z
    d_ue4 = np.array([d_cv[2], d_cv[0], -d_cv[1]], dtype=np.float64)

    # Rotate the direction into world frame.
    R = camera_to_world[:3, :3]
    d_world = R @ d_ue4

    # Ray origin = camera position in world.
    o_world = camera_to_world[:3, 3]

    # Intersect with z = ground_z.
    if abs(d_world[2]) < 1e-9:
        return None  # ray parallel to ground plane
    t = (ground_z - o_world[2]) / d_world[2]
    if t <= 0.0:
        return None  # intersection behind the camera
    p = o_world + t * d_world
    return (float(p[0]), float(p[1]))


def iou_xyxy(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """Axis-aligned bbox IoU. Boxes in (x1, y1, x2, y2) format."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0
