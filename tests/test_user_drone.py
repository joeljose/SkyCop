"""Unit tests for skycop.control.user_drone — pure-pose controller."""

from __future__ import annotations

import math

import pytest

from skycop.control.user_drone import (
    Pose,
    UserDroneConfig,
    UserDroneController,
)


def _ctrl(**overrides) -> UserDroneController:
    cfg = UserDroneConfig(**overrides)
    return UserDroneController(cfg=cfg)


def _pose(x=0.0, y=0.0, z=25.0, yaw_rad=0.0) -> Pose:
    return Pose(x=x, y=y, z=z, yaw_rad=yaw_rad)


# ── Null input ────────────────────────────────────────────────────

def test_no_keys_returns_unchanged_pose():
    c = _ctrl()
    p0 = _pose(x=1.0, y=2.0, z=25.0, yaw_rad=0.5)
    p1 = c.step(p0, frozenset(), dt=0.05)
    assert p1.x == p0.x
    assert p1.y == p0.y
    assert p1.z == p0.z
    assert p1.yaw_rad == p0.yaw_rad


def test_rejects_non_positive_dt():
    c = _ctrl()
    with pytest.raises(ValueError):
        c.step(_pose(), {"w"}, dt=0.0)
    with pytest.raises(ValueError):
        c.step(_pose(), {"w"}, dt=-0.05)


# ── Forward / back (drone heading follow) ─────────────────────────

def test_w_moves_body_forward_at_yaw_zero():
    c = _ctrl(max_speed_mps=10.0)
    p1 = c.step(_pose(yaw_rad=0.0), {"w"}, dt=0.1)
    # Yaw=0 → body-forward is world +X. 10 m/s * 0.1s = 1 m.
    assert abs(p1.x - 1.0) < 1e-9
    assert abs(p1.y) < 1e-9


def test_w_at_yaw_90_moves_world_plus_y():
    c = _ctrl(max_speed_mps=10.0)
    p1 = c.step(_pose(yaw_rad=math.pi / 2), {"w"}, dt=0.1)
    # Yaw=90° → body-forward is world +Y.
    assert abs(p1.x) < 1e-9
    assert abs(p1.y - 1.0) < 1e-9


def test_s_moves_body_back():
    c = _ctrl(max_speed_mps=10.0)
    p1 = c.step(_pose(yaw_rad=0.0), {"s"}, dt=0.1)
    assert abs(p1.x - (-1.0)) < 1e-9


# ── Strafe ────────────────────────────────────────────────────────

def test_d_strafes_body_right():
    c = _ctrl(max_speed_mps=10.0)
    # Yaw 0 → body +Y (right) is world -Y (CARLA left-handed convention
    # our convention puts right = +sin(yaw + 90°) = +cos(yaw); we mapped
    # D → (+right) → world (-sin(yaw), +cos(yaw)). At yaw 0: (0, +1)).
    p1 = c.step(_pose(yaw_rad=0.0), {"d"}, dt=0.1)
    assert abs(p1.x) < 1e-9
    assert abs(p1.y - 1.0) < 1e-9


def test_a_strafes_body_left():
    c = _ctrl(max_speed_mps=10.0)
    p1 = c.step(_pose(yaw_rad=0.0), {"a"}, dt=0.1)
    assert abs(p1.x) < 1e-9
    assert abs(p1.y - (-1.0)) < 1e-9


# ── Diagonal normalisation ────────────────────────────────────────

def test_diagonal_wd_is_speed_clamped():
    c = _ctrl(max_speed_mps=10.0)
    p1 = c.step(_pose(yaw_rad=0.0), {"w", "d"}, dt=0.1)
    # W + D → body (1, 1) normalised to unit + scaled by speed = (0.707, 0.707) * 10 * 0.1
    disp = math.hypot(p1.x, p1.y)
    assert abs(disp - 1.0) < 1e-6


def test_opposing_keys_cancel():
    c = _ctrl()
    p1 = c.step(_pose(), {"w", "s"}, dt=0.1)
    assert abs(p1.x) < 1e-9
    assert abs(p1.y) < 1e-9


# ── Yaw ───────────────────────────────────────────────────────────

def test_q_rotates_ccw():
    c = _ctrl(yaw_rate_deg_s=90.0)
    p1 = c.step(_pose(yaw_rad=0.0), {"q"}, dt=1.0)
    # Q = yaw left. 90°/s * 1s → -π/2.
    assert abs(p1.yaw_rad - (-math.pi / 2)) < 1e-6


def test_e_rotates_cw():
    c = _ctrl(yaw_rate_deg_s=90.0)
    p1 = c.step(_pose(yaw_rad=0.0), {"e"}, dt=1.0)
    assert abs(p1.yaw_rad - (math.pi / 2)) < 1e-6


def test_qe_together_cancel():
    c = _ctrl(yaw_rate_deg_s=90.0)
    p1 = c.step(_pose(yaw_rad=0.3), {"q", "e"}, dt=1.0)
    assert abs(p1.yaw_rad - 0.3) < 1e-9


def test_yaw_wraps_to_negative_pi_plus_pi():
    c = _ctrl(yaw_rate_deg_s=360.0)
    p1 = c.step(_pose(yaw_rad=math.pi - 0.1), {"e"}, dt=1.0)
    # yaw + 2π wraps; result should be inside (-π, π]
    assert -math.pi < p1.yaw_rad <= math.pi


# ── Altitude ──────────────────────────────────────────────────────

def test_shift_ascends():
    c = _ctrl(altitude_rate_mps=5.0)
    p1 = c.step(_pose(z=25.0), {"shift"}, dt=0.2)
    assert abs(p1.z - 26.0) < 1e-9


def test_ctrl_descends():
    c = _ctrl(altitude_rate_mps=5.0)
    p1 = c.step(_pose(z=25.0), {"control"}, dt=0.2)
    assert abs(p1.z - 24.0) < 1e-9


def test_altitude_clamped_at_floor():
    c = _ctrl(altitude_rate_mps=5.0, min_altitude_m=2.0)
    p1 = c.step(_pose(z=3.0), {"control"}, dt=1.0)
    # Would descend to -2 m but clamp at 2 m.
    assert p1.z == 2.0


# ── Case-insensitive + ignores unknowns ──────────────────────────

def test_uppercase_keys_accepted():
    c = _ctrl(max_speed_mps=10.0)
    p1 = c.step(_pose(), {"W"}, dt=0.1)
    assert abs(p1.x - 1.0) < 1e-9


def test_unknown_keys_ignored():
    c = _ctrl()
    p1 = c.step(_pose(), {"x", "space", "tab"}, dt=0.1)
    assert p1.x == 0.0 and p1.y == 0.0 and p1.z == 25.0


# ── Integration: keep walking forward for 1 s ────────────────────

def test_walking_forward_for_1s_covers_max_speed_metres():
    c = _ctrl(max_speed_mps=20.0)
    p = _pose()
    dt = 0.05
    for _ in range(20):
        p = c.step(p, {"w"}, dt=dt)
    assert abs(p.x - 20.0) < 1e-6
    assert abs(p.y) < 1e-9
