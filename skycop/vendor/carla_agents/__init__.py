"""Vendored CARLA agents subset — just enough to run GlobalRoutePlanner.

Upstream: https://github.com/carla-simulator/carla/tree/master/PythonAPI/carla/agents
License: MIT (see LICENSE_CARLA in this directory).

Why vendored: ``pip install carla==0.9.16`` does not ship the ``agents``
module — only the binary/source tarball does. We only need
``GlobalRoutePlanner`` (graph-search router over the map's waypoint graph)
to drive the suspect to a chosen parking destination when the FSM enters
PARKING, because ``TrafficManager.set_path()`` is known-broken in 0.9.13
onward.

Scope: this subset deliberately skips BasicAgent / BehaviorAgent (depend
on shapely for obstacle polygons, which we don't need), LocalPlanner
(uses CARLA's own PID — we drive via our lighter WaypointFollower), and
the full tools/misc module (only the ``vector`` helper is actually
needed by the router).
"""

from skycop.vendor.carla_agents.controller import VehiclePIDController
from skycop.vendor.carla_agents.global_route_planner import GlobalRoutePlanner
from skycop.vendor.carla_agents.local_planner import LocalPlanner
from skycop.vendor.carla_agents.road_option import RoadOption

__all__ = [
    "GlobalRoutePlanner",
    "LocalPlanner",
    "RoadOption",
    "VehiclePIDController",
]
