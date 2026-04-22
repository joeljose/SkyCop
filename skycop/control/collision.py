"""Slide-along collision for the user-mode drone (v0c).

CARLA integration lives here, not in user_drone.py, so the pure pose
controller stays test-only-pure. Each tick:

1. UserDroneController computes an *intended* pose from key input.
2. This module casts a ray from the current drone position toward the
   intended position; if the ray hits a non-Road / non-Vehicle surface,
   it decomposes the motion into (parallel-to-surface, kept) +
   (perpendicular-to-surface, blocked) using the hit normal.
3. Returns the safe pose to commit.

If the ray hit's label is outside a small *pass-through* allow-list
(Road / RoadLines / Vehicles / Sky / NONE), we treat it as solid.

Ground floor handled by ``user_drone.UserDroneController`` via
``min_altitude_m`` — this module only deals with vertical structures.
"""

from __future__ import annotations

import carla

from skycop.control.user_drone import Pose

# Ray labels we treat as passable (drone is non-collision aerial camera,
# but we mark these as "ok to fly through" explicitly).
_PASSABLE_LABELS: frozenset[str] = frozenset(
    {"Roads", "RoadLines", "Vehicles", "NONE", "Sky", "Terrain"}
)


def apply_slide_along(
    world: carla.World,
    current: Pose,
    intended: Pose,
    drone_radius_m: float = 0.75,
) -> Pose:
    """Project ``intended`` onto the allowed slide plane if it would collide.

    If the cast from ``current`` to ``intended`` is clear, returns
    ``intended`` unchanged. Otherwise decomposes the motion using the
    hit normal; if the normal cannot be recovered (CARLA returns only
    a label + point on ``cast_ray`` in 0.9.16), falls back to *soft gate*
    — cancel horizontal motion this tick, keep vertical.
    """
    # Altitude-only move: skip the cast (ground floor handled upstream).
    dx = intended.x - current.x
    dy = intended.y - current.y
    if dx == 0.0 and dy == 0.0:
        return intended

    # Extend the ray by the drone radius so we stop *before* touching.
    mag = (dx * dx + dy * dy) ** 0.5
    ext = (drone_radius_m / mag) if mag > 0 else 0.0
    start = carla.Location(current.x, current.y, current.z)
    end = carla.Location(
        intended.x + dx * ext,
        intended.y + dy * ext,
        intended.z,
    )
    hits = world.cast_ray(start, end)
    blocker = _first_blocker(hits)
    if blocker is None:
        return intended

    # CARLA's LabelledPoint (0.9.16) gives us label + location. No normal.
    # Estimate one from hit geometry: the normal lies in the horizontal plane
    # pointing from the hit back toward the current position (AABB approx).
    hx = blocker.location.x
    hy = blocker.location.y
    nx = current.x - hx
    ny = current.y - hy
    nmag = (nx * nx + ny * ny) ** 0.5
    if nmag < 1e-6:
        # Degenerate — we're right on top of the hit. Soft-gate.
        return Pose(x=current.x, y=current.y, z=intended.z, yaw_rad=intended.yaw_rad)
    nx /= nmag
    ny /= nmag

    # Decompose the requested motion into (parallel, perpendicular) w.r.t. normal.
    # perp = (v · n) n,  parallel = v - perp.  We keep only the parallel component.
    v_dot_n = dx * nx + dy * ny
    # If motion is away from the wall (v · n > 0), there's no conflict — let it through.
    if v_dot_n > 0:
        return intended
    slide_x = dx - v_dot_n * nx
    slide_y = dy - v_dot_n * ny
    return Pose(
        x=current.x + slide_x,
        y=current.y + slide_y,
        z=intended.z,
        yaw_rad=intended.yaw_rad,
    )


def _first_blocker(hits):
    """Return the first ray hit whose label isn't in the pass-through set."""
    for h in hits:
        label_name = str(h.label).split(".")[-1]
        if label_name not in _PASSABLE_LABELS:
            return h
    return None
