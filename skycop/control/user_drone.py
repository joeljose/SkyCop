"""User drone controller — body-frame snappy WASD-QE flight.

Pure-logic module; no CARLA. Given a pose (x, y, z, yaw) and a set of
currently-pressed keys, computes the commanded pose for the next tick.

Convention (per v0c grill — option B "drone-heading follow"):

- ``yaw_rad`` — drone body heading in world frame. Camera yaw follows.
- **W** → drone-forward  (world Δ = (cos yaw, sin yaw) · speed).
- **S** → drone-back.
- **A / D** → strafe left / right in body frame.
- **Q / E** → yaw left / right at ``yaw_rate_deg_s``.
- **Shift / Ctrl** → altitude up / down at ``altitude_rate_mps``.

Snappy feel (no momentum): pressed key → max velocity that axis, released
→ zero. Velocities clamp per-axis at ``max_speed_mps``. Collision handling
is out of scope — the mission loop applies raycast slide-along after this
returns.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

_TRACKED_KEYS = frozenset(["w", "a", "s", "d", "q", "e", "shift", "control"])


@dataclass(frozen=True)
class UserDroneConfig:
    max_speed_mps: float = 20.0             # body-frame horizontal cap
    altitude_rate_mps: float = 5.0          # vertical
    yaw_rate_deg_s: float = 60.0            # deg per second while Q / E held
    min_altitude_m: float = 1.0             # ground clearance floor


@dataclass
class Pose:
    x: float
    y: float
    z: float
    yaw_rad: float


@dataclass
class UserDroneController:
    cfg: UserDroneConfig = field(default_factory=UserDroneConfig)

    # ── Public API ──────────────────────────────────────────────────

    def step(
        self,
        pose: Pose,
        pressed: frozenset[str] | set[str],
        dt: float,
    ) -> Pose:
        """Return the commanded pose after ``dt`` seconds of input."""
        if dt <= 0:
            raise ValueError(f"dt must be > 0, got {dt}")

        pressed = {k.lower() for k in pressed} & set(_TRACKED_KEYS)

        # ── Yaw rate ──
        yaw_rate_rad = math.radians(self.cfg.yaw_rate_deg_s)
        dyaw = 0.0
        if "q" in pressed:
            dyaw -= yaw_rate_rad * dt
        if "e" in pressed:
            dyaw += yaw_rate_rad * dt
        new_yaw = self._wrap_rad(pose.yaw_rad + dyaw)

        # ── Body-frame WASD → world velocity ──
        # Body +X = forward, body +Y = right (CARLA left-handed convention).
        fwd = 0.0
        right = 0.0
        if "w" in pressed:
            fwd += 1.0
        if "s" in pressed:
            fwd -= 1.0
        if "d" in pressed:
            right += 1.0
        if "a" in pressed:
            right -= 1.0
        mag = math.hypot(fwd, right)
        if mag > 1.0:
            fwd /= mag
            right /= mag
        # Use the *current* yaw (not the post-rotation one) for motion —
        # keeps the drone moving in the direction it was actually facing
        # at the start of the tick.
        vx_world = (fwd * math.cos(pose.yaw_rad) - right * math.sin(pose.yaw_rad)) * self.cfg.max_speed_mps
        vy_world = (fwd * math.sin(pose.yaw_rad) + right * math.cos(pose.yaw_rad)) * self.cfg.max_speed_mps
        new_x = pose.x + vx_world * dt
        new_y = pose.y + vy_world * dt

        # ── Altitude ──
        vz = 0.0
        if "shift" in pressed:
            vz += self.cfg.altitude_rate_mps
        if "control" in pressed:
            vz -= self.cfg.altitude_rate_mps
        new_z = max(self.cfg.min_altitude_m, pose.z + vz * dt)

        return Pose(x=new_x, y=new_y, z=new_z, yaw_rad=new_yaw)

    # ── Helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _wrap_rad(a: float) -> float:
        """Wrap ``a`` to (-π, π]."""
        return (a + math.pi) % (2 * math.pi) - math.pi
