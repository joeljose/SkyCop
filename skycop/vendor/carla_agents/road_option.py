"""RoadOption enum extracted from CARLA's agents.navigation.local_planner.

Vendored to avoid pulling the whole LocalPlanner chain (controller.py,
tools/misc.py) when we only need the enum to talk to GlobalRoutePlanner.

Upstream: https://github.com/carla-simulator/carla/blob/master/PythonAPI/carla/agents/navigation/local_planner.py
License: MIT.
"""

from enum import IntEnum


class RoadOption(IntEnum):
    """Topological transitions between lane segments — see upstream."""
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6
