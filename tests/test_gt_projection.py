"""Unit tests for skycop.cv.gt_projection — pure math, no CARLA."""

import math

import numpy as np
import pytest

from skycop.cv.gt_projection import (
    build_camera_matrix,
    iou_xyxy,
    project_points,
    world_bbox_to_image,
)


def _camera_world_transform(x: float, y: float, z: float, pitch_deg: float = 0.0) -> np.ndarray:
    """Build a 4x4 camera-to-world transform with pitch around Y (CARLA convention).

    Identity orientation puts CARLA camera X forward, Y right, Z up. A
    negative pitch tilts the camera downward (nose toward ground).
    """
    t = np.eye(4)
    p = math.radians(pitch_deg)
    # Rotation around Y axis (pitch in CARLA, left-handed): x' = cos*x + sin*z, z' = -sin*x + cos*z
    c, s = math.cos(p), math.sin(p)
    t[0, 0] = c
    t[0, 2] = s
    t[2, 0] = -s
    t[2, 2] = c
    t[:3, 3] = [x, y, z]
    return t


def test_build_camera_matrix_focal_and_principal_point():
    K = build_camera_matrix(1280, 720, 90.0)
    # 90° horizontal FOV → focal = W / (2 * tan(45°)) = W / 2 = 640
    assert abs(K[0, 0] - 640.0) < 1e-6
    assert abs(K[1, 1] - 640.0) < 1e-6
    assert K[0, 2] == 640.0
    assert K[1, 2] == 360.0


def test_project_single_point_ahead_of_camera_goes_to_principal_point():
    # Camera at origin, looking down +X (CARLA). Point on the +X axis projects
    # to the principal point (image center).
    K = build_camera_matrix(640, 480, 90.0)
    cam_world = np.eye(4)   # identity = at origin, X forward
    world_to_cam = np.linalg.inv(cam_world)
    points = np.array([[10.0, 0.0, 0.0]])  # 10 m in front, on axis
    uv = project_points(points, world_to_cam, K)
    assert not np.isnan(uv).any()
    assert abs(uv[0, 0] - K[0, 2]) < 1e-6
    assert abs(uv[0, 1] - K[1, 2]) < 1e-6


def test_project_point_offset_right_goes_right_of_center():
    K = build_camera_matrix(640, 480, 90.0)
    cam_world = np.eye(4)
    world_to_cam = np.linalg.inv(cam_world)
    # Shifted +Y (right in CARLA) at same forward distance — should land right of centre.
    points = np.array([[10.0, 1.0, 0.0]])
    uv = project_points(points, world_to_cam, K)
    assert uv[0, 0] > K[0, 2]
    # y unchanged (z_up is zero) → image row == principal
    assert abs(uv[0, 1] - K[1, 2]) < 1e-6


def test_project_point_above_goes_above_center():
    K = build_camera_matrix(640, 480, 90.0)
    cam_world = np.eye(4)
    world_to_cam = np.linalg.inv(cam_world)
    # Shifted +Z (up in CARLA) — should project ABOVE image center (lower v value).
    points = np.array([[10.0, 0.0, 1.0]])
    uv = project_points(points, world_to_cam, K)
    assert abs(uv[0, 0] - K[0, 2]) < 1e-6
    assert uv[0, 1] < K[1, 2]


def test_project_points_behind_camera_are_nan():
    K = build_camera_matrix(640, 480, 90.0)
    cam_world = np.eye(4)
    world_to_cam = np.linalg.inv(cam_world)
    # Behind = -X in CARLA
    points = np.array([[-5.0, 0.0, 0.0]])
    uv = project_points(points, world_to_cam, K)
    assert np.isnan(uv).all()


def test_world_bbox_to_image_builds_axis_aligned_rect():
    K = build_camera_matrix(640, 480, 90.0)
    cam_world = np.eye(4)
    world_to_cam = np.linalg.inv(cam_world)
    # 8 vertices of a 2x2x2 m cube centred at (10, 0, 0)
    cx, cy, cz = 10.0, 0.0, 0.0
    verts = np.array([
        [cx - 1, cy - 1, cz - 1], [cx + 1, cy - 1, cz - 1],
        [cx - 1, cy + 1, cz - 1], [cx + 1, cy + 1, cz - 1],
        [cx - 1, cy - 1, cz + 1], [cx + 1, cy - 1, cz + 1],
        [cx - 1, cy + 1, cz + 1], [cx + 1, cy + 1, cz + 1],
    ])
    bbox = world_bbox_to_image(verts, world_to_cam, K, image_size=(480, 640))
    assert bbox is not None
    x1, y1, x2, y2 = bbox
    # Cube straddles principal point in X and Y
    assert x1 < K[0, 2] < x2
    assert y1 < K[1, 2] < y2


def test_world_bbox_to_image_all_behind_returns_none():
    K = build_camera_matrix(640, 480, 90.0)
    cam_world = np.eye(4)
    world_to_cam = np.linalg.inv(cam_world)
    verts = np.array([[-3, -1, -1], [-3, 1, -1], [-3, -1, 1], [-3, 1, 1]] * 2)
    bbox = world_bbox_to_image(verts, world_to_cam, K, image_size=(480, 640))
    assert bbox is None


def test_project_points_wrong_shape_raises():
    K = build_camera_matrix(640, 480, 90.0)
    with pytest.raises(ValueError):
        project_points(np.array([1.0, 2.0, 3.0]), np.eye(4), K)


def test_iou_xyxy_identical_boxes_is_one():
    box = (10.0, 10.0, 50.0, 50.0)
    assert iou_xyxy(box, box) == 1.0


def test_iou_xyxy_disjoint_boxes_is_zero():
    a = (0.0, 0.0, 10.0, 10.0)
    b = (100.0, 100.0, 110.0, 110.0)
    assert iou_xyxy(a, b) == 0.0


def test_iou_xyxy_half_overlap():
    a = (0.0, 0.0, 10.0, 10.0)
    b = (5.0, 0.0, 15.0, 10.0)
    # overlap 5x10=50, union = 100+100-50=150 → IoU = 50/150 ≈ 0.333
    assert abs(iou_xyxy(a, b) - 1.0 / 3.0) < 1e-6
