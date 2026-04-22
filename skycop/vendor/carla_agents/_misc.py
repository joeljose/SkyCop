"""Minimal subset of CARLA's agents.tools.misc — only the helpers the
vendored GlobalRoutePlanner + LocalPlanner + VehiclePIDController use.

Upstream: https://github.com/carla-simulator/carla/blob/master/PythonAPI/carla/agents/tools/misc.py
License: MIT.

We vendor ``vector`` (used by GlobalRoutePlanner), ``get_speed`` (used by
LocalPlanner + VehiclePIDController), and ``draw_waypoints`` (used by
LocalPlanner debug-draw). The upstream file also contains
``is_within_distance`` and a few others that only BasicAgent uses — those
are intentionally skipped.
"""

import math

import carla
import numpy as np


def vector(location_1, location_2):
    """Unit vector from ``location_1`` to ``location_2`` (carla.Location)."""
    x = location_2.x - location_1.x
    y = location_2.y - location_1.y
    z = location_2.z - location_1.z
    norm = np.linalg.norm([x, y, z]) + np.finfo(float).eps
    return [x / norm, y / norm, z / norm]


def get_speed(vehicle):
    """Compute the speed of a CARLA vehicle in km/h."""
    vel = vehicle.get_velocity()
    return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)


def draw_waypoints(world, waypoints, z=0.5):
    """Draw a list of waypoints at height ``z`` — useful for debug visuals."""
    for wpt in waypoints:
        wpt_t = wpt.transform
        begin = wpt_t.location + carla.Location(z=z)
        angle = math.radians(wpt_t.rotation.yaw)
        end = begin + carla.Location(x=math.cos(angle), y=math.sin(angle))
        world.debug.draw_arrow(begin, end, arrow_size=0.3, life_time=1.0)
