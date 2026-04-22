"""Smoke tests for the vendored CARLA agents subset.

No CARLA server required — we only verify the imports wire up correctly
and the small hand-vendored helpers (RoadOption enum, vector helper)
match upstream values.
"""

from __future__ import annotations

from skycop.vendor.carla_agents import GlobalRoutePlanner, RoadOption
from skycop.vendor.carla_agents._misc import vector


def test_road_option_values_match_upstream():
    # Pinned to the upstream enum (local_planner.RoadOption) — a silent
    # drift here would change GlobalRoutePlanner's output.
    assert RoadOption.VOID == -1
    assert RoadOption.LEFT == 1
    assert RoadOption.RIGHT == 2
    assert RoadOption.STRAIGHT == 3
    assert RoadOption.LANEFOLLOW == 4
    assert RoadOption.CHANGELANELEFT == 5
    assert RoadOption.CHANGELANERIGHT == 6


def test_vector_is_unit_and_correct_direction():
    class _L:
        def __init__(self, x: float, y: float, z: float) -> None:
            self.x, self.y, self.z = x, y, z
    v = vector(_L(0.0, 0.0, 0.0), _L(3.0, 4.0, 0.0))
    assert abs(v[0] - 0.6) < 1e-9
    assert abs(v[1] - 0.8) < 1e-9
    assert abs(v[2]) < 1e-9


def test_global_route_planner_is_importable():
    # Construction requires a carla.Map — skip that, just verify the class
    # object is accessible through the public import path.
    assert GlobalRoutePlanner.__name__ == "GlobalRoutePlanner"
